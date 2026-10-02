"""One trajectory: exact MD, live bond monitoring, and serial pathway refinement."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory

from ._durable import ensure_directory, publish, sync_directory
from ._version import __version__
from .candidates import ReactionCandidate, ReactionTracker
from .detection import BondChangeDetector, BondDetectorConfig, assign_atom_ids, atom_ids
from .pathway import CalculatorProvider, PathwayConfig, PathwayOutcome, refine_pathway
from .restart import ExactRestartSnapshot
from .runtime import ExactDynamicsRuntime, ExactRuntimeProvider
from .store import OccurrenceRecord, OccurrenceStore

_PHASES = {"new", "running", "refining", "completed", "failed"}


@dataclass(frozen=True, slots=True)
class ReactionRunConfig:
    """Observation cadence, detector, candidate stability, and pathway controls."""

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
    """Small durable-run summary returned by ``ReactionRun.run``."""

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


def _last_frame(path: Path) -> int:
    """Global frame of the last trajectory entry, or -1 when none has been written."""

    if not path.exists() or path.stat().st_size == 0:
        return -1
    with Trajectory(path) as trajectory:
        if len(trajectory) == 0:
            return -1
        return int(trajectory[-1].info["reactionflow_global_frame"])


def _same_atomic_state(first: Atoms, second: Atoms) -> bool:
    return (
        set(first.arrays) == set(second.arrays)
        and all(np.array_equal(first.arrays[name], second.arrays[name]) for name in first.arrays)
        and np.array_equal(first.cell.array, second.cell.array)
        and np.array_equal(first.pbc, second.pbc)
    )


class ReactionRun:
    """Connect detection, candidate storage, pathways, and exact MD checkpoints.

    Every observation publishes one exact checkpoint: the dynamics and calculator snapshot plus
    the reaction tracker, named by ``state.json`` together with the detector state. A confirmed
    reaction releases the MD runtime, refines the queued pathways, and restores that same
    checkpoint into the next trajectory generation, exactly as a resubmitted job does.
    """

    def __init__(self, root: Path, config: ReactionRunConfig) -> None:
        self.root = root.resolve()
        self.config = config
        self.state_path = self.root / "state.json"
        self.pathways = self.root / "pathways"
        self.runtime_checkpoints = self.root / "runtime-checkpoints"
        ensure_directory(self.pathways)
        ensure_directory(self.runtime_checkpoints)
        self.occurrences = OccurrenceStore(self.root)
        self._phase = "new"
        self._generation = 0
        self._global_step = 0
        self._global_frame = 0
        self._pending: list[str] = []
        self._failure: dict[str, str] | None = None
        self._initial: Atoms | None = None
        self._detector: BondChangeDetector | None = None
        self._tracker: ReactionTracker | None = None
        self._active_checkpoint: Path | None = None
        self._snapshot: ExactRestartSnapshot | None = None
        self._created_with: str | None = __version__

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        config: ReactionRunConfig | None = None,
    ) -> ReactionRun:
        """Create a new durable run directory."""

        path = Path(root).resolve()
        ensure_directory(path)
        if (path / "state.json").exists():
            raise FileExistsError(path / "state.json")
        run = cls(path, config or ReactionRunConfig())
        run._write_state()
        return run

    @classmethod
    def open(cls, root: str | Path) -> ReactionRun:
        """Open a run at its last durable observation boundary."""

        path = Path(root).resolve()
        value = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if value.get("schema_version") != 2 or value["phase"] not in _PHASES:
            raise ValueError("unsupported ReactionRun state")
        run = cls(path, ReactionRunConfig.from_dict(value["config"]))
        run._phase = value["phase"]
        run._generation = int(value["generation"])
        run._global_step = int(value["global_step"])
        run._global_frame = int(value["global_frame"])
        run._pending = list(map(str, value["pending_pathway_ids"]))
        run._created_with = value.get("created_with_reactionflow")
        run._failure = value.get("failure")
        if value.get("detector_state") is not None:
            run._detector = BondChangeDetector.from_state(
                value["detector_state"],
                config=run.config.detector,
            )
        if value.get("active_checkpoint") is not None:
            run._active_checkpoint = run.root / value["active_checkpoint"]
        if run._phase in {"running", "refining"}:
            run._load_checkpoint()
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

    def _write_state(self) -> None:
        state = {
            "schema_version": 2,
            "created_with_reactionflow": self._created_with,
            "phase": self._phase,
            "generation": self._generation,
            "global_step": self._global_step,
            "global_frame": self._global_frame,
            "pending_pathway_ids": list(self._pending),
            "failure": self._failure,
            "config": self.config.to_dict(),
            "detector_state": None if self._detector is None else self._detector.export_state(),
            "active_checkpoint": (
                None
                if self._active_checkpoint is None
                else str(self._active_checkpoint.relative_to(self.root))
            ),
        }
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        publish(temporary, self.state_path)
        # Prune only after both the new state and its directory entry are durable.
        # A failed publication must leave the previous checkpoint available for recovery.
        keep = None if self._active_checkpoint is None else self._active_checkpoint.name
        for path in self.runtime_checkpoints.iterdir():
            if path.name != keep:
                shutil.rmtree(path, ignore_errors=True)

    def _checkpoint(self, runtime: ExactDynamicsRuntime) -> None:
        """Publish the exact runtime and tracker at this boundary, then the state naming them."""

        assert self._tracker is not None
        if int(runtime.nsteps) != self._global_step:
            raise ValueError("exact runtime step counter does not match the durable run")
        snapshot = runtime.snapshot()
        if not _same_atomic_state(runtime.atoms, snapshot.atoms):
            raise ValueError("exact runtime snapshot does not match its live atoms")
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
            # The previous checkpoint is removed once state.json names this one, so this one must
            # already be on disk if the node fails.
            publish(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self._active_checkpoint = final
        self._snapshot = snapshot
        self._write_state()

    def _load_checkpoint(self) -> None:
        if self._active_checkpoint is None or self._detector is None:
            raise ValueError("the run has no exact checkpoint to resume from")
        tracker = ReactionTracker.read_checkpoint(
            self._active_checkpoint / "tracker",
            stability_frames=self.config.candidate_stability_frames,
        )
        if not self._detector.last_frame == tracker.last_frame == self._global_frame:
            raise ValueError("runtime checkpoint monitor does not match the resume boundary")
        self._tracker = tracker
        self._snapshot = ExactRestartSnapshot.read(self._active_checkpoint / "exact-restart")

    @contextmanager
    def _fatal(self, stage: str) -> Iterator[None]:
        """Mark the run failed when its durable record cannot be maintained."""

        try:
            yield
        except Exception as error:
            self._phase = "failed"
            self._failure = {"stage": stage, "type": type(error).__name__, "message": str(error)}
            self._write_state()
            raise

    def _start(self, atoms: Atoms) -> None:
        """Assign stable IDs and seed the detector/tracker baseline."""

        with self._fatal("start"):
            initial = assign_atom_ids(atoms.copy())
            initial.calc = None
            initial.info.pop("atom_ids", None)
            self._detector = BondChangeDetector(self.config.detector)
            self._tracker = ReactionTracker(stability_frames=self.config.candidate_stability_frames)
            self._detector.process(initial, frame=0)
            self._tracker.process(
                initial,
                frame=0,
                stable_bonds=self._detector.stable_bonds,
                pending_bonds=self._detector.pending_bonds,
            )
            self._initial = initial
            self._phase = "running"

    def _queue_pathway(self, record: OccurrenceRecord, candidate: ReactionCandidate) -> None:
        if not candidate.resolved or (self.pathways / record.occurrence_id).is_dir():
            return
        for previous in self.occurrences.records():
            if previous.class_id != record.class_id:
                continue
            if previous.occurrence_id in self._pending:
                return
            result_path = self.pathways / previous.occurrence_id / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result["status"] == "ci_neb_converged":
                    return
        self._pending.append(record.occurrence_id)

    def _register(self, candidates: tuple[ReactionCandidate, ...], *, label: str) -> None:
        # Registration is idempotent, so an exact replay of an observation that was interrupted
        # before its checkpoint queues the same pathways again.
        for index, candidate in enumerate(candidates):
            occurrence_id = f"segment-{self._generation:04d}-frame-{label}-{index:04d}"
            record, _ = self.occurrences.register(
                occurrence_id,
                candidate,
                detector_config=self.config.detector,
            )
            self._queue_pathway(record, candidate)

    def _observe(self, atoms: Atoms, global_step: int) -> None:
        """Advance the monitor one frame and register every completed occurrence."""

        assert self._detector is not None and self._tracker is not None
        frame = self._global_frame + 1
        with self._fatal("observe"):
            self._detector.process(atoms, frame=frame)
            candidates = self._tracker.process(
                atoms,
                frame=frame,
                stable_bonds=self._detector.stable_bonds,
                pending_bonds=self._detector.pending_bonds,
            )
            self._register(candidates, label=f"{frame:08d}")
            self._global_step = global_step
            self._global_frame = frame
            if self._pending:
                self._phase = "refining"

    def _write_frame(self, trajectory: Trajectory, atoms: Atoms, written: int) -> None:
        """Append this boundary's frame unless an interrupted attempt already wrote it."""

        if self._global_frame <= written:
            return
        frame = _transport(atoms)
        frame.info["reactionflow_global_step"] = self._global_step
        frame.info["reactionflow_global_frame"] = self._global_frame
        trajectory.write(frame)

    def _run_segment(self, total_steps: int, runtime_provider: ExactRuntimeProvider) -> None:
        """Run MD from the current checkpoint until a reaction is confirmed or the run ends."""

        manager = (
            runtime_provider.start(self._initial)
            if self._snapshot is None
            else runtime_provider.restore(self._snapshot)
        )
        with manager as runtime:
            if self._snapshot is None:
                self._checkpoint(runtime)
            elif int(runtime.nsteps) != self._global_step:
                raise ValueError("exact runtime step counter does not match the durable run")

            path = self.root / f"segments/{self._generation:04d}/trajectory.traj"
            ensure_directory(path.parent)
            # Each frame is written before its checkpoint, so an interrupted attempt may already
            # hold this boundary's frame or the next one, which the exact replay reproduces.
            written = _last_frame(path)
            if written > self._global_frame + 1:
                raise ValueError("trajectory is ahead of its exact runtime checkpoint")
            with Trajectory(path, "w" if written < 0 else "a") as trajectory:
                self._write_frame(trajectory, runtime.atoms, written)
                while self._global_step < total_steps and self._phase == "running":
                    steps = min(self.config.observation_interval, total_steps - self._global_step)
                    before = int(runtime.nsteps)
                    runtime.run(steps)
                    advanced = int(runtime.nsteps) - before
                    if advanced != steps:
                        raise RuntimeError(
                            f"exact runtime advanced {advanced} steps; expected {steps}"
                        )
                    self._observe(runtime.atoms, self._global_step + steps)
                    self._write_frame(trajectory, runtime.atoms, written)
                    self._checkpoint(runtime)

    def _publish_outcome(self, occurrence_id: str, outcome: PathwayOutcome) -> None:
        final = self.pathways / occurrence_id
        temporary = self.pathways / f".{occurrence_id}-{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            with Trajectory(temporary / "images.traj", "w") as trajectory:
                for image in outcome.images:
                    trajectory.write(_transport(image))
            result = {
                "schema_version": 1,
                "occurrence_id": occurrence_id,
                "reactionflow_version": __version__,
                "class_id": self.occurrences.record(occurrence_id).class_id,
                "status": outcome.status,
                "barrier_eV": outcome.barrier,
                "energies_eV": list(outcome.energies),
                "method": outcome.method,
                "pressure_GPa": outcome.pressure_GPa,
                "barrier_quantity": outcome.barrier_quantity,
                "enthalpies_eV": list(outcome.enthalpies),
                "volumes_A3": list(outcome.volumes),
                "message": outcome.message,
                "frequency_validation": (
                    None
                    if outcome.frequency_validation is None
                    else asdict(outcome.frequency_validation)
                ),
                "connectivity": (
                    None if outcome.connectivity is None else asdict(outcome.connectivity)
                ),
            }
            (temporary / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            publish(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _refine_pending(
        self,
        calculator_provider: CalculatorProvider,
        pressure_GPa: float | None,
    ) -> None:
        """Refine each queued occurrence serially and publish its immutable result."""

        with self._fatal("refine"):
            while self._pending:
                occurrence_id = self._pending[0]
                if (self.pathways / occurrence_id).is_dir():
                    # Published before an interruption; make its name durable and move on.
                    sync_directory(self.pathways)
                else:
                    outcome = refine_pathway(
                        self.occurrences.load(occurrence_id),
                        calculator_provider=calculator_provider,
                        config=self.config.pathway,
                        detector_config=self.occurrences.load_detector_config(occurrence_id),
                        pressure_GPa=pressure_GPa,
                    )
                    self._publish_outcome(occurrence_id, outcome)
                self._pending.pop(0)
                self._write_state()

    def _complete(self) -> None:
        """Drain unresolved terminal candidates and mark the run complete."""

        assert self._tracker is not None
        with self._fatal("complete"):
            self._register(self._tracker.finish(), label=f"{self._global_frame:08d}-terminal")
            self._phase = "completed"
            self._write_state()

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

    def run(
        self,
        atoms: Atoms | None = None,
        *,
        runtime_provider: ExactRuntimeProvider,
        pathway_calculator_provider: CalculatorProvider,
        total_steps: int,
        pressure_GPa: float | None = None,
    ) -> RunSummary:
        """Run or resume live detection, serial NEB/CI-NEB, and exact MD to a step target."""

        if isinstance(total_steps, bool) or not isinstance(total_steps, Integral):
            raise ValueError("total_steps must be a non-negative integer")
        if total_steps < self._global_step:
            raise ValueError("total_steps cannot precede the durable run counter")
        if self._phase == "new":
            if atoms is None:
                raise ValueError("initial atoms are required for a new run")
            self._start(atoms)
        elif atoms is not None:
            raise ValueError("initial atoms may only be supplied to a new run")

        if self._phase == "refining":
            # A reopened adapter has not checked its model against the saved MD state yet.
            # Verify before any pathway result can be published, including terminal work
            # that will never resume MD. Read a separate snapshot so validation cannot
            # mutate the run's checkpoint.
            assert self._active_checkpoint is not None
            snapshot = ExactRestartSnapshot.read(self._active_checkpoint / "exact-restart")
            with runtime_provider.restore(snapshot):
                pass

        # Only failures to maintain the durable record mark the run failed. Any other error, such
        # as a runtime that this environment refuses to restore, leaves the last exact checkpoint
        # in place, so the run resumes once the cause is fixed.
        while self._phase != "completed":
            if self._phase == "failed":
                raise RuntimeError(f"ReactionRun failed: {self._failure}")
            if self._pending:
                self._refine_pending(pathway_calculator_provider, pressure_GPa)
            elif self._global_step >= total_steps:
                self._complete()
            else:
                if self._phase == "refining":
                    # Refinement is finished; MD continues in a new trajectory generation.
                    self._generation += 1
                    self._phase = "running"
                self._run_segment(int(total_steps), runtime_provider)
        return self.summary()


__all__ = ["ReactionRun", "ReactionRunConfig", "RunSummary"]
