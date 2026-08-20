"""Minimal scheduler-neutral reaction run and synchronous ASE executor."""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.trajectory import Trajectory

from .candidates import ReactionCandidate, ReactionTracker
from .detection import BondChangeDetector, BondDetectorConfig, assign_atom_ids, atom_ids
from .pathway import CalculatorProvider, PathwayConfig, PathwayOutcome, refine_pathway
from .restart import ExactRestartSnapshot
from .runtime import ExactDynamicsRuntime, ExactRuntimeProvider
from .segments import ResumeToken, SegmentGeneration, SegmentStore
from .store import OccurrenceRecord, OccurrenceStore

DynamicsFactory = Callable[[Atoms], Any]

_LOGGER = logging.getLogger(__name__)
_STRUCTURAL_RESTART_NOTICE = (
    "Exact ASE dynamics state was not restored; continuing generation %d from a structural "
    "checkpoint. The caller will provide fresh RNG, thermostat, barostat, integrator, and "
    "calculator runtime state."
)


@dataclass(frozen=True, slots=True)
class ReactionRunConfig:
    """Controls shared by manual and synchronous ReactionRun execution."""

    observation_interval: int = 100
    detector: BondDetectorConfig = field(default_factory=BondDetectorConfig)
    candidate_stability_frames: int = 3
    pathway: PathwayConfig = field(default_factory=PathwayConfig)

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_interval", self.observation_interval),
            ("candidate_stability_frames", self.candidate_stability_frames),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_interval": int(self.observation_interval),
            "detector": self.detector.to_dict(),
            "candidate_stability_frames": int(self.candidate_stability_frames),
            "pathway": asdict(self.pathway),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReactionRunConfig:
        return cls(
            observation_interval=value["observation_interval"],
            detector=BondDetectorConfig.from_dict(value["detector"]),
            candidate_stability_frames=value["candidate_stability_frames"],
            pathway=PathwayConfig(**value["pathway"]),
        )


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Small durable-run summary returned by ``run_ase``."""

    phase: str
    generation: int
    global_step: int
    global_frame: int
    occurrences: int
    pathways: int


def _transport(atoms: Atoms) -> Atoms:
    snapshot = atoms.copy()
    snapshot.calc = None
    snapshot.info["atom_ids"] = list(atom_ids(snapshot))
    return snapshot


def _same_atomic_state(first: Atoms, second: Atoms) -> bool:
    return (
        set(first.arrays) == set(second.arrays)
        and all(np.array_equal(first.arrays[name], second.arrays[name]) for name in first.arrays)
        and np.array_equal(first.cell.array, second.cell.array)
        and np.array_equal(first.pbc, second.pbc)
    )


class ReactionRun:
    """Connect detection, candidate storage, pathways, and segment checkpoints."""

    def __init__(self, root: Path, config: ReactionRunConfig) -> None:
        self.root = root.resolve()
        self.config = config
        self.state_path = self.root / "state.json"
        self.pathways = self.root / "pathways"
        self.runtime_checkpoints = self.root / "runtime-checkpoints"
        self.pathways.mkdir(parents=True, exist_ok=True)
        self.runtime_checkpoints.mkdir(parents=True, exist_ok=True)
        self.occurrences = OccurrenceStore(self.root)
        self.segments = SegmentStore(self.root)
        self._phase = "new"
        self._generation = 0
        self._global_step = 0
        self._global_frame = 0
        self._pending: list[str] = []
        self._failure: dict[str, str] | None = None
        self._segment: SegmentGeneration | None = None
        self._detector: BondChangeDetector | None = None
        self._tracker: ReactionTracker | None = None
        self._active_checkpoint: Path | None = None
        self._exact_snapshot: ExactRestartSnapshot | None = None

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        config: ReactionRunConfig | None = None,
    ) -> ReactionRun:
        """Create a new durable run directory."""

        path = Path(root).resolve()
        path.mkdir(parents=True, exist_ok=True)
        if (path / "state.json").exists():
            raise FileExistsError(path / "state.json")
        run = cls(path, config or ReactionRunConfig())
        run._write_state()
        return run

    @classmethod
    def open(cls, root: str | Path) -> ReactionRun:
        """Open a run, including one interrupted after a complete checkpoint."""

        path = Path(root).resolve()
        value = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if value.get("schema_version") != 1:
            raise ValueError("unsupported ReactionRun state")
        run = cls(path, ReactionRunConfig.from_dict(value["config"]))
        run._phase = str(value["phase"])
        run._generation = int(value["generation"])
        run._global_step = int(value["global_step"])
        run._global_frame = int(value["global_frame"])
        run._pending = list(map(str, value["pending_pathway_ids"]))
        failure = value.get("failure")
        run._failure = None if failure is None else dict(failure)
        detector_state = value.get("detector_state")
        if detector_state is not None:
            run._detector = BondChangeDetector.from_state(
                detector_state,
                config=run.config.detector,
            )
        active_checkpoint = value.get("active_checkpoint")
        if active_checkpoint is not None:
            if not isinstance(active_checkpoint, str):
                raise ValueError("active runtime checkpoint path must be a string")
            candidate = (run.root / active_checkpoint).resolve()
            if not candidate.is_relative_to(run.runtime_checkpoints):
                raise ValueError("active runtime checkpoint escapes the run directory")
            run._active_checkpoint = candidate

        token_path = run._token_path
        if run._phase == "checkpoint_pending" and token_path.is_file():
            ResumeToken.read(token_path)
            run._phase = "refining"
            run._write_state()
        if run._phase == "refining" and not run._pending:
            run._phase = "resume_ready"
            run._write_state()
        if run._phase == "running" and run._active_checkpoint is not None:
            run._load_runtime_checkpoint()
        elif run._phase == "running" and run._generation > 0:
            trajectory = run.root / f"segments/{run._generation:04d}/trajectory.traj"
            if not trajectory.exists():
                token = ResumeToken.read(
                    run.root / f"segments/{run._generation - 1:04d}/checkpoint/resume.json"
                )
                run._segment = run.segments.resume(token, recover_empty=True)
                run._restore_observers(run._segment.atoms)
                _LOGGER.warning(_STRUCTURAL_RESTART_NOTICE, run._generation)
        return run

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def global_frame(self) -> int:
        return self._global_frame

    @property
    def pending_pathway_ids(self) -> tuple[str, ...]:
        return tuple(self._pending)

    @property
    def failure(self) -> Mapping[str, str] | None:
        return None if self._failure is None else dict(self._failure)

    @property
    def current_segment(self) -> SegmentGeneration | None:
        return self._segment

    @property
    def _token_path(self) -> Path:
        return self.root / f"segments/{self._generation:04d}/checkpoint/resume.json"

    def _state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase": self._phase,
            "generation": self._generation,
            "global_step": self._global_step,
            "global_frame": self._global_frame,
            "pending_pathway_ids": list(self._pending),
            "failure": self._failure,
            "config": self.config.to_dict(),
            "detector_state": (None if self._detector is None else self._detector.export_state()),
            "active_checkpoint": (
                None
                if self._active_checkpoint is None
                else str(self._active_checkpoint.relative_to(self.root))
            ),
        }

    def _write_state(self) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._state(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _publish_runtime_checkpoint(
        self,
        snapshot: ExactRestartSnapshot,
        *,
        write_state: bool,
    ) -> Path:
        if self._segment is None or self._detector is None or self._tracker is None:
            raise RuntimeError("an active segment and monitor are required for an exact checkpoint")
        if not _same_atomic_state(self._segment.atoms, snapshot.atoms):
            raise ValueError("exact snapshot atoms do not match the active trajectory")
        name = (
            f"g{self._generation:04d}-s{self._global_step:012d}-"
            f"f{self._global_frame:012d}-{uuid4().hex}"
        )
        final = self.runtime_checkpoints / name
        temporary = self.runtime_checkpoints / f".{name}.tmp"
        temporary.mkdir()
        try:
            snapshot.write(temporary / "exact-restart")
            self._tracker.write_checkpoint(temporary / "tracker")
            manifest = {
                "schema_version": 1,
                "generation": self._generation,
                "global_step": self._global_step,
                "global_frame": self._global_frame,
                "detector_state": self._detector.export_state(),
            }
            (temporary / "checkpoint.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self._active_checkpoint = final
        self._exact_snapshot = snapshot
        if write_state:
            self._write_state()
        return final

    def _load_runtime_checkpoint(self) -> None:
        assert self._active_checkpoint is not None
        manifest = json.loads(
            (self._active_checkpoint / "checkpoint.json").read_text(encoding="utf-8")
        )
        expected = {
            "schema_version": 1,
            "generation": self._generation,
            "global_step": self._global_step,
            "global_frame": self._global_frame,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("active runtime checkpoint does not match the durable run state")
        detector = BondChangeDetector.from_state(
            manifest["detector_state"],
            config=self.config.detector,
        )
        if self._detector is not None and detector.export_state() != self._detector.export_state():
            raise ValueError("runtime checkpoint detector conflicts with the run state")
        tracker = ReactionTracker.read_checkpoint(
            self._active_checkpoint / "tracker",
            stability_frames=self.config.candidate_stability_frames,
        )
        if detector.last_frame != self._global_frame or tracker.last_frame != self._global_frame:
            raise ValueError("runtime checkpoint monitor does not match the resume boundary")
        snapshot = ExactRestartSnapshot.read(self._active_checkpoint / "exact-restart")
        self._detector = detector
        self._tracker = tracker
        self._exact_snapshot = snapshot
        self._segment = self.segments.attach(
            self._generation,
            snapshot.atoms,
            global_step=self._global_step,
            global_frame=self._global_frame,
        )

    def _record_failure(self, stage: str, error: Exception) -> None:
        self._phase = "failed"
        self._failure = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        }
        self._write_state()

    def _require(self, *phases: str) -> None:
        if self._phase not in phases:
            expected = " or ".join(phases)
            raise RuntimeError(f"ReactionRun phase is {self._phase!r}, expected {expected}")

    def _seed_observers(self, atoms: Atoms) -> None:
        self._detector = BondChangeDetector(self.config.detector)
        self._tracker = ReactionTracker(stability_frames=self.config.candidate_stability_frames)
        self._detector.process(atoms, frame=self._global_frame)
        self._tracker.process(
            atoms,
            frame=self._global_frame,
            stable_bonds=self._detector.stable_bonds,
            pending_bonds=self._detector.pending_bonds,
        )

    def _restore_observers(self, atoms: Atoms) -> None:
        if self._detector is None or self._detector.last_frame != self._global_frame:
            raise ValueError("the detector checkpoint does not match the resume boundary")
        if self._detector.pending_bonds is not None:
            raise ValueError("cannot resume from a detector with pending changes")
        self._tracker = ReactionTracker(stability_frames=self.config.candidate_stability_frames)
        self._tracker.process(
            atoms,
            frame=self._global_frame,
            stable_bonds=self._detector.stable_bonds,
            pending_bonds=None,
        )

    def start(self, atoms: Atoms) -> SegmentGeneration:
        """Start generation zero and seed its detector/tracker baseline."""

        self._require("new")
        try:
            self._segment = self.segments.start(atoms)
            self._generation = self._segment.generation
            self._global_step = self._segment.global_step
            self._global_frame = self._segment.global_frame
            self._seed_observers(self._segment.atoms)
            self._phase = "running"
            self._write_state()
            return self._segment
        except Exception as error:
            self._record_failure("start", error)
            raise

    def _register(
        self,
        candidates: tuple[ReactionCandidate, ...],
        *,
        label: str,
    ) -> tuple[OccurrenceRecord, ...]:
        records: list[OccurrenceRecord] = []
        for index, candidate in enumerate(candidates):
            occurrence_id = f"segment-{self._generation:04d}-frame-{label}-{index:04d}"
            record, inserted = self.occurrences.register(
                occurrence_id,
                candidate,
                detector_config=self.config.detector,
            )
            records.append(record)
            if inserted and record.is_representative:
                self._pending.append(record.occurrence_id)
        return tuple(records)

    def observe(
        self,
        atoms: Atoms,
        *,
        global_step: int,
        global_frame: int,
    ) -> tuple[OccurrenceRecord, ...]:
        """Observe one safe boundary and register every completed occurrence."""

        return self._observe(
            atoms,
            global_step=global_step,
            global_frame=global_frame,
            persist=True,
        )

    def _observe(
        self,
        atoms: Atoms,
        *,
        global_step: int,
        global_frame: int,
        persist: bool,
    ) -> tuple[OccurrenceRecord, ...]:
        """Update the monitor, optionally deferring state publication to a checkpoint."""

        self._require("running")
        if global_step < self._global_step or global_frame <= self._global_frame:
            raise ValueError("observation counters must move forward")
        assert self._detector is not None and self._tracker is not None
        try:
            self._detector.process(atoms, frame=global_frame)
            candidates = self._tracker.process(
                atoms,
                frame=global_frame,
                stable_bonds=self._detector.stable_bonds,
                pending_bonds=self._detector.pending_bonds,
            )
            records = self._register(candidates, label=f"{global_frame:08d}")
            self._global_step = int(global_step)
            self._global_frame = int(global_frame)
            if self._pending:
                self._phase = "checkpoint_pending"
            if persist:
                self._write_state()
            return records
        except Exception as error:
            self._record_failure("observe", error)
            raise

    def checkpoint(
        self,
        atoms: Atoms,
        *,
        exact_restart: ExactRestartSnapshot | None = None,
    ) -> ResumeToken:
        """Publish the current structural or exact checkpoint before pathway work."""

        self._require("checkpoint_pending")
        if self._segment is None:
            raise RuntimeError("the live segment is unavailable; reopen only after checkpointing")
        try:
            token = self.segments.checkpoint(
                self._segment,
                atoms,
                global_step=self._global_step,
                global_frame=self._global_frame,
                exact_restart=exact_restart,
            )
            self._phase = "refining"
            self._write_state()
            return token
        except Exception as error:
            self._record_failure("checkpoint", error)
            raise

    def _record_for(self, occurrence_id: str) -> OccurrenceRecord:
        for record in self.occurrences.records():
            if record.occurrence_id == occurrence_id:
                return record
        raise KeyError(occurrence_id)

    def _write_images(self, path: Path, images: tuple[Atoms, ...]) -> None:
        trajectory = Trajectory(path, "w")
        try:
            for image in images:
                trajectory.write(_transport(image))
        finally:
            trajectory.close()

    def _publish_outcome(
        self,
        occurrence_id: str,
        outcome: PathwayOutcome,
    ) -> None:
        final = self.pathways / occurrence_id
        if final.exists():
            return
        record = self._record_for(occurrence_id)
        temporary = self.pathways / f".{occurrence_id}-{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            self._write_images(temporary / "images.traj", outcome.images)
            result = {
                "schema_version": 1,
                "occurrence_id": occurrence_id,
                "class_id": record.class_id,
                "status": outcome.status,
                "barrier_eV": outcome.barrier,
                "energies_eV": list(outcome.energies),
                "message": outcome.message,
            }
            (temporary / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _load_outcome(self, occurrence_id: str) -> PathwayOutcome:
        directory = self.pathways / occurrence_id
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if result.get("schema_version") != 1 or result.get("occurrence_id") != occurrence_id:
            raise ValueError(f"unsupported pathway result for {occurrence_id!r}")
        images = read(directory / "images.traj", ":")
        for image in images:
            atom_ids(image)
            assign_atom_ids(image)
            image.info.pop("atom_ids", None)
            image.calc = None
        barrier = result.get("barrier_eV")
        return PathwayOutcome(
            status=str(result["status"]),
            barrier=None if barrier is None else float(barrier),
            energies=tuple(map(float, result.get("energies_eV", []))),
            images=tuple(images),
            message=str(result.get("message", "")),
        )

    def refine_pending(
        self,
        calculator_provider: CalculatorProvider,
    ) -> tuple[PathwayOutcome, ...]:
        """Refine each queued representative serially and publish its result."""

        self._require("refining", "resume_ready")
        if self._phase == "resume_ready":
            return ()
        outcomes: list[PathwayOutcome] = []
        try:
            while self._pending:
                occurrence_id = self._pending[0]
                directory = self.pathways / occurrence_id
                if directory.is_dir():
                    outcome = self._load_outcome(occurrence_id)
                else:
                    outcome = refine_pathway(
                        self.occurrences.load(occurrence_id),
                        calculator_provider=calculator_provider,
                        config=self.config.pathway,
                        detector_config=self.occurrences.load_detector_config(occurrence_id),
                    )
                    self._publish_outcome(occurrence_id, outcome)
                outcomes.append(outcome)
                self._pending.pop(0)
                self._write_state()
            self._phase = "resume_ready"
            self._write_state()
            return tuple(outcomes)
        except Exception as error:
            self._record_failure("refine", error)
            raise

    def resume_segment(self) -> SegmentGeneration:
        """Resume the completed checkpoint into a fresh trajectory generation."""

        self._require("resume_ready")
        if self._pending:
            raise RuntimeError("pathways remain pending")
        try:
            token = ResumeToken.read(self._token_path)
            self._segment = self.segments.resume(token, recover_empty=True)
            self._generation = self._segment.generation
            self._global_step = self._segment.global_step
            self._global_frame = self._segment.global_frame
            self._restore_observers(self._segment.atoms)
            self._phase = "running"
            self._write_state()
            _LOGGER.warning(_STRUCTURAL_RESTART_NOTICE, self._generation)
            return self._segment
        except Exception as error:
            self._record_failure("resume", error)
            raise

    def resume_exact_segment(self) -> SegmentGeneration:
        """Resume an exact checkpoint into a fresh trajectory generation."""

        self._require("resume_ready")
        if self._pending:
            raise RuntimeError("pathways remain pending")
        try:
            token = ResumeToken.read(self._token_path)
            snapshot = self.segments.read_exact(token)
            self._segment = self.segments.resume(token, recover_empty=True)
            self._segment = self.segments.bind(self._segment, snapshot.atoms)
            self._generation = self._segment.generation
            self._global_step = self._segment.global_step
            self._global_frame = self._segment.global_frame
            self._restore_observers(self._segment.atoms)
            self._phase = "running"
            self._active_checkpoint = None
            self._exact_snapshot = snapshot
            self._write_state()
            return self._segment
        except Exception as error:
            self._record_failure("resume_exact", error)
            raise

    def complete(self) -> RunSummary:
        """Drain unresolved terminal candidates and mark the run complete."""

        self._require("running", "resume_ready", "completed")
        if self._phase == "completed":
            return self.summary()
        try:
            if self._phase == "running":
                assert self._tracker is not None
                self._register(
                    self._tracker.finish(),
                    label=f"{self._global_frame:08d}-terminal",
                )
            self._phase = "completed"
            self._write_state()
            return self.summary()
        except Exception as error:
            self._record_failure("complete", error)
            raise

    def summary(self) -> RunSummary:
        return RunSummary(
            phase=self._phase,
            generation=self._generation,
            global_step=self._global_step,
            global_frame=self._global_frame,
            occurrences=len(self.occurrences.records()),
            pathways=sum(
                1
                for path in self.pathways.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ),
        )

    def _run_segment(
        self,
        *,
        total_steps: int,
        md_calculator_provider: CalculatorProvider,
        dynamics_factory: DynamicsFactory,
    ) -> None:
        assert self._segment is not None
        atoms = self._segment.atoms
        trajectory = Trajectory(self._segment.trajectory_path, "w")
        try:
            trajectory.write(_transport(atoms))
            with md_calculator_provider("md") as calculator:
                atoms.calc = calculator
                try:
                    dynamics = dynamics_factory(atoms)
                    while self._global_step < total_steps and self._phase == "running":
                        requested = min(
                            self.config.observation_interval,
                            total_steps - self._global_step,
                        )
                        before = int(dynamics.nsteps)
                        dynamics.run(steps=requested)
                        advanced = int(dynamics.nsteps) - before
                        if advanced < 1:
                            raise RuntimeError("ASE dynamics did not advance")
                        next_step = self._global_step + advanced
                        next_frame = self._global_frame + 1
                        trajectory.write(_transport(atoms))
                        self.observe(
                            atoms,
                            global_step=next_step,
                            global_frame=next_frame,
                        )
                finally:
                    atoms.calc = None
        finally:
            atoms.calc = None
            trajectory.close()

    def _snapshot_runtime(self, runtime: ExactDynamicsRuntime) -> ExactRestartSnapshot:
        if isinstance(runtime.nsteps, bool) or not isinstance(runtime.nsteps, Integral):
            raise TypeError("exact runtime nsteps must be an integer")
        if int(runtime.nsteps) != self._global_step:
            raise ValueError("exact runtime step counter does not match the durable run")
        snapshot = runtime.snapshot()
        if not _same_atomic_state(runtime.atoms, snapshot.atoms):
            raise ValueError("exact runtime snapshot does not match its live atoms")
        return snapshot

    def _write_exact_boundary(self, trajectory: Trajectory, atoms: Atoms) -> None:
        snapshot = _transport(atoms)
        snapshot.info["reactionflow_global_step"] = self._global_step
        snapshot.info["reactionflow_global_frame"] = self._global_frame
        trajectory.write(snapshot)

    def _run_exact_segment(
        self,
        *,
        total_steps: int,
        runtime_provider: ExactRuntimeProvider,
    ) -> None:
        assert self._segment is not None
        manager = (
            runtime_provider.start(self._segment.atoms)
            if self._exact_snapshot is None
            else runtime_provider.restore(self._exact_snapshot)
        )
        with manager as runtime:
            self._segment = self.segments.bind(self._segment, runtime.atoms)
            initial = self._snapshot_runtime(runtime)
            if self._active_checkpoint is None:
                self._publish_runtime_checkpoint(initial, write_state=True)

            trajectory_exists = self._segment.trajectory_path.exists()
            write_initial = True
            if trajectory_exists:
                last = read(self._segment.trajectory_path, -1)
                marker = last.info.get("reactionflow_global_frame")
                write_initial = marker is None or int(marker) < self._global_frame
                if marker is not None and int(marker) > self._global_frame:
                    raise ValueError("trajectory is ahead of its exact runtime checkpoint")
            mode = "a" if trajectory_exists else "w"
            trajectory = Trajectory(self._segment.trajectory_path, mode)
            try:
                if write_initial:
                    self._write_exact_boundary(trajectory, runtime.atoms)

                while self._global_step < total_steps and self._phase == "running":
                    requested = min(
                        self.config.observation_interval,
                        total_steps - self._global_step,
                    )
                    before = int(runtime.nsteps)
                    runtime.run(requested)
                    advanced = int(runtime.nsteps) - before
                    if advanced != requested:
                        raise RuntimeError(
                            f"exact runtime advanced {advanced} steps; expected {requested}"
                        )
                    next_step = self._global_step + advanced
                    next_frame = self._global_frame + 1
                    self._observe(
                        runtime.atoms,
                        global_step=next_step,
                        global_frame=next_frame,
                        persist=False,
                    )
                    snapshot = self._snapshot_runtime(runtime)
                    self._publish_runtime_checkpoint(snapshot, write_state=False)
                    if self._phase == "checkpoint_pending":
                        self.checkpoint(runtime.atoms, exact_restart=snapshot)
                    else:
                        self._write_state()
                    self._write_exact_boundary(trajectory, runtime.atoms)
            finally:
                trajectory.close()

    def run_exact(
        self,
        atoms: Atoms | None = None,
        *,
        runtime_provider: ExactRuntimeProvider,
        pathway_calculator_provider: CalculatorProvider,
        total_steps: int,
    ) -> RunSummary:
        """Run live detection, serial NEB/CI-NEB, and exact MD continuation."""

        if isinstance(total_steps, bool) or not isinstance(total_steps, Integral):
            raise ValueError("total_steps must be a non-negative integer")
        if total_steps < self._global_step:
            raise ValueError("total_steps cannot precede the durable run counter")
        if self._phase == "new":
            if atoms is None:
                raise ValueError("initial atoms are required for a new run")
            self.start(atoms)
        elif atoms is not None:
            raise ValueError("initial atoms may only be supplied to a new run")

        try:
            while self._phase != "completed":
                if self._phase == "failed":
                    raise RuntimeError(f"ReactionRun failed: {self._failure}")
                if self._phase == "checkpoint_pending":
                    raise RuntimeError(
                        "exact runtime was interrupted before its reaction checkpoint completed"
                    )
                if self._phase == "refining":
                    self.refine_pending(pathway_calculator_provider)
                    continue
                if self._phase == "resume_ready":
                    if self._global_step >= total_steps:
                        self.complete()
                    else:
                        self.resume_exact_segment()
                    continue
                if self._phase == "running":
                    if self._global_step >= total_steps:
                        self.complete()
                        continue
                    if self._segment is None:
                        raise RuntimeError("active exact runtime checkpoint is unavailable")
                    self._run_exact_segment(
                        total_steps=int(total_steps),
                        runtime_provider=runtime_provider,
                    )
                    continue
                raise RuntimeError(f"unsupported ReactionRun phase {self._phase!r}")
            return self.summary()
        except Exception as error:
            if self._phase != "failed":
                self._record_failure("run_exact", error)
            raise

    def run_ase(
        self,
        atoms: Atoms | None = None,
        *,
        md_calculator_provider: CalculatorProvider,
        pathway_calculator_provider: CalculatorProvider,
        dynamics_factory: DynamicsFactory,
        total_steps: int,
    ) -> RunSummary:
        """Run synchronous ASE MD, refinement, and structural resume to a step target."""

        if isinstance(total_steps, bool) or not isinstance(total_steps, Integral):
            raise ValueError("total_steps must be a non-negative integer")
        if total_steps < self._global_step:
            raise ValueError("total_steps cannot precede the durable run counter")
        if self._phase == "new":
            if atoms is None:
                raise ValueError("initial atoms are required for a new run")
            self.start(atoms)
        elif atoms is not None:
            raise ValueError("initial atoms may only be supplied to a new run")

        try:
            while self._phase != "completed":
                if self._phase == "failed":
                    raise RuntimeError(f"ReactionRun failed: {self._failure}")
                if self._phase == "checkpoint_pending":
                    if self._segment is None:
                        raise RuntimeError("checkpoint was not completed before interruption")
                    self.checkpoint(self._segment.atoms)
                    continue
                if self._phase == "refining":
                    self.refine_pending(pathway_calculator_provider)
                    continue
                if self._phase == "resume_ready":
                    if self._global_step >= total_steps:
                        self.complete()
                    else:
                        self.resume_segment()
                    continue
                if self._phase == "running":
                    if self._global_step >= total_steps:
                        self.complete()
                        continue
                    if self._segment is None:
                        raise RuntimeError(
                            "active MD cannot be structurally recovered without a checkpoint"
                        )
                    self._run_segment(
                        total_steps=int(total_steps),
                        md_calculator_provider=md_calculator_provider,
                        dynamics_factory=dynamics_factory,
                    )
                    continue
                raise RuntimeError(f"unsupported ReactionRun phase {self._phase!r}")
            return self.summary()
        except Exception as error:
            if self._phase != "failed":
                self._record_failure("run_ase", error)
            raise


__all__ = [
    "DynamicsFactory",
    "ReactionRun",
    "ReactionRunConfig",
    "RunSummary",
]
