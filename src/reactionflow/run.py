"""One trajectory: exact MD, live bond monitoring, and serial pathway refinement."""

from __future__ import annotations

import json
import shutil
import time
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory

from ._durable import ensure_directory, flush_to_disk, publish, publishing, sync_directory
from ._version import __version__
from .candidates import ReactionCandidate, ReactionTracker
from .detection import BondChangeDetector, BondDetectorConfig, canonical_copy, transport_copy
from .pathway import (
    BarrierTimeLimit,
    CalculatorProvider,
    PathwayConfig,
    PathwayOutcome,
    refine_pathway,
)
from .restart import ExactRestartSnapshot
from .runtime import ExactDynamicsRuntime, ExactRuntimeProvider
from .store import OccurrenceStore

_PHASES = {"new", "running", "refining", "completed", "failed"}

# By file times, refinements that reached a connected saddle took 1.1 and 1.8 hours (288-atom
# acetonitrile, SSNEB, UMA on one GPU), and none of the nine longest, from 8.5 h, reached a saddle.
DEFAULT_MAX_HOURS_PER_BARRIER = 4.0


@dataclass(frozen=True, slots=True)
class ReactionRunConfig:
    """Observation cadence, detector, candidate stability, and pathway controls."""

    observation_interval: int = 100
    detector: BondDetectorConfig = field(default_factory=BondDetectorConfig)
    candidate_stability_frames: int = 3
    candidate_region_radius: int | None = 2
    pathway: PathwayConfig = field(default_factory=PathwayConfig)

    def __post_init__(self) -> None:
        for name in ("observation_interval", "candidate_stability_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        radius = self.candidate_region_radius
        if radius is not None and (
            isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 0
        ):
            raise ValueError("candidate_region_radius must be a non-negative integer or null")

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_interval": int(self.observation_interval),
            "detector": self.detector.to_dict(),
            "candidate_stability_frames": int(self.candidate_stability_frames),
            "candidate_region_radius": self.candidate_region_radius,
            "pathway": asdict(self.pathway),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReactionRunConfig:
        return cls(
            observation_interval=value["observation_interval"],
            detector=BondDetectorConfig.from_dict(value["detector"]),
            candidate_stability_frames=value["candidate_stability_frames"],
            # Runs started before local regions took whole bonded components, and runs started
            # before step-by-step refinement relaxed the tracker's reactant frame in one band.
            candidate_region_radius=value.get("candidate_region_radius"),
            pathway=PathwayConfig.from_dict(
                {"reactant_lookback": 0, "split_steps": False, **value["pathway"]}
            ),
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


def _last_frame(path: Path) -> int:
    """Global frame of the last trajectory entry, or -1 when none has been written."""

    if not path.exists() or path.stat().st_size == 0:
        return -1
    with Trajectory(path) as trajectory:
        if len(trajectory) == 0:
            return -1
        return int(trajectory[-1].info["reactionflow_global_frame"])


def _readable_span(path: Path) -> tuple[int, int]:
    """The first global frame of a segment and the last one left to read in it."""

    with Trajectory(path) as trajectory:
        first = int(trajectory[0].info["reactionflow_global_frame"])
        for index in range(len(trajectory) - 1, 0, -1):
            try:
                return first, int(trajectory[index].info["reactionflow_global_frame"])
            except Exception:  # a cut tail
                continue
        return first, first


def missing_frames(root: Path, last_frame: int) -> list[list[int]]:
    """Ranges of global frames up to ``last_frame`` that no segment of this run holds.

    Each frame is on disk before the checkpoint that counts it, except the very first frame of a
    run, so a healthy run reports nothing however it was stopped. This reads each segment's ends,
    so it finds a segment that is gone or cut; damage inside one shows only when ``scan`` reads it.
    """

    if last_frame < 1:
        return []
    segments = [
        path for path in root.glob("segments/*/trajectory.traj") if path.parent.name.isdigit()
    ]
    missing, expected = [], 0
    for path in sorted(segments, key=lambda path: int(path.parent.name)):
        try:
            span = _readable_span(path)
        except Exception:  # an empty or cut file raises one of several types
            continue
        if span[0] > expected:
            missing.append([expected, span[0] - 1])
        expected = max(expected, span[1] + 1)
    if expected <= last_frame:
        missing.append([expected, last_frame])
    return missing


def frame_ranges(ranges: list[list[int]]) -> str:
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def _same_atomic_state(first: Atoms, second: Atoms) -> bool:
    return (
        set(first.arrays) == set(second.arrays)
        and all(np.array_equal(first.arrays[name], second.arrays[name]) for name in first.arrays)
        and np.array_equal(first.cell.array, second.cell.array)
        and np.array_equal(first.pbc, second.pbc)
    )


def _require_finite(snapshot: ExactRestartSnapshot) -> None:
    """Refuse a non-finite MD state, so that it reaches neither a frame nor a checkpoint.

    The integrator's arrays count too: NaN forces or stress on the first evaluation, or NaN stress
    on the last step of an interval, leave the atoms finite and would poison every later step.
    """

    atoms = snapshot.atoms
    arrays = [atoms.positions, atoms.get_momenta(), atoms.cell.array]
    arrays += [a for a in snapshot.dynamics.arrays.values() if np.issubdtype(a.dtype, np.inexact)]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(
            "MD state is non-finite, usually from NaN or inf in the model's output or a blown-up "
            "trajectory; nothing was saved for it, so the next job repeats it"
        )


def _write_result(directory: Path, outcome: PathwayOutcome, identity: Mapping[str, Any]) -> None:
    with Trajectory(directory / "images.traj", "w") as trajectory:
        for image in outcome.images:
            trajectory.write(transport_copy(image))
    result = {**identity, **outcome.to_dict()}
    (directory / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class ReactionRun:
    """Connect detection, candidate storage, pathways, and exact MD checkpoints.

    Every observation publishes one exact checkpoint: the dynamics and calculator snapshot plus
    the reaction tracker, named by ``state.json`` together with the detector state. A confirmed
    reaction releases the MD runtime, refines the queued pathways, and restores that same
    checkpoint into the next trajectory generation, exactly as a resubmitted job does. With
    ``refine_after_md``, confirmed reactions instead wait in the queue until MD reaches its target.
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
        self._barrier_seconds = 0.0
        self._barrier_clock = 0.0
        self._initial: Atoms | None = None
        self._detector: BondChangeDetector | None = None
        self._tracker: ReactionTracker | None = None
        self._active_checkpoint: Path | None = None
        self._snapshot: ExactRestartSnapshot | None = None
        self._created_with: str | None = __version__
        self._refine_after_md = False

    @classmethod
    def create(cls, root: str | Path, *, config: ReactionRunConfig | None = None) -> ReactionRun:
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
        run._barrier_seconds = float(value.get("barrier_seconds", 0.0))
        if value.get("detector_state") is not None:
            run._detector = BondChangeDetector.from_state(
                value["detector_state"], config=run.config.detector
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
            "barrier_seconds": self._barrier_seconds,
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
        # The previous checkpoint is removed once state.json names this one, so this one must
        # already be on disk if the node fails.
        with publishing(temporary, final):
            snapshot.write(temporary / "exact-restart")
            self._tracker.write_checkpoint(temporary / "tracker")
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
        if tracker.region_radius != self.config.candidate_region_radius:
            raise ValueError("reaction-tracker checkpoint configuration does not match")
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
            failure = {"stage": stage, "type": type(error).__name__, "message": str(error)}
            self._failure = self._failure or failure  # the first is why MD stopped
            self._write_state()
            raise

    def _start(self, atoms: Atoms) -> None:
        """Assign stable IDs and seed the detector/tracker baseline."""

        with self._fatal("start"):
            initial = canonical_copy(atoms)
            self._detector = BondChangeDetector(self.config.detector)
            self._tracker = ReactionTracker(
                stability_frames=self.config.candidate_stability_frames,
                region_radius=self.config.candidate_region_radius,
            )
            self._detector.process(initial, frame=0)
            self._tracker.process(
                initial,
                frame=0,
                stable_bonds=self._detector.stable_bonds,
                pending_bonds=self._detector.pending_bonds,
            )
            self._initial = initial
            self._phase = "running"

    def _register(self, candidates: tuple[ReactionCandidate, ...], *, label: str) -> None:
        # Registration is idempotent, so an exact replay of an observation that was interrupted
        # before its checkpoint queues the same pathways again.
        for index, candidate in enumerate(candidates):
            occurrence_id = f"segment-{self._generation:04d}-frame-{label}-{index:04d}"
            self.occurrences.register(
                occurrence_id, candidate, detector_config=self.config.detector
            )
            # Every resolved occurrence is refined, also one that repeats an earlier reaction,
            # because the reactions around it can change its barrier.
            if (
                candidate.resolved
                and occurrence_id not in self._pending
                and not (self.pathways / occurrence_id).is_dir()
            ):
                self._pending.append(occurrence_id)

    def _observe(self, atoms: Atoms, global_step: int, total_steps: int) -> None:
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
            if self._pending and (not self._refine_after_md or global_step >= total_steps):
                self._phase = "refining"

    def _write_frame(self, trajectory: Trajectory, atoms: Atoms, written: int) -> None:
        """Append this boundary's frame unless an interrupted attempt already wrote it."""

        if self._global_frame <= written:
            return
        frame = transport_copy(atoms)
        frame.info["reactionflow_global_step"] = self._global_step
        frame.info["reactionflow_global_frame"] = self._global_frame
        trajectory.write(frame)

    def _run_segment(self, total_steps: int, runtime_provider: ExactRuntimeProvider) -> None:
        """Run MD from the current checkpoint until refinement is due or the run ends."""

        manager = (
            runtime_provider.start(self._initial)
            if self._snapshot is None
            else runtime_provider.restore(self._snapshot)
        )
        with manager as runtime:
            if self._snapshot is None:
                _require_finite(runtime.snapshot())
                self._checkpoint(runtime)
            elif int(runtime.nsteps) != self._global_step:
                raise ValueError("exact runtime step counter does not match the durable run")
            else:
                self._record_environment_change(runtime.snapshot())

            path = self.root / f"segments/{self._generation:04d}/trajectory.traj"
            ensure_directory(path.parent)
            # Each frame is written before its checkpoint, so an interrupted attempt may already
            # hold this boundary's frame or the next one, which the exact replay reproduces.
            way_out = (
                "Restore it from a copy, or move it aside to continue in a new file without its "
                "frames, which every later run then reports as missing"
            )
            try:
                written = _last_frame(path)
            except Exception as error:  # a cut file raises one of several types
                raise ValueError(
                    f"{path} cannot be read, so this run cannot append to it "
                    f"({type(error).__name__}: {error}). {way_out}"
                ) from error
            if written > self._global_frame + 1:
                raise ValueError("trajectory is ahead of its exact runtime checkpoint")
            if 0 <= written < self._global_frame:
                raise ValueError(
                    f"trajectory is missing frames before its exact runtime checkpoint: {path} "
                    f"ends at frame {written}, and the checkpoint is at {self._global_frame}. "
                    f"{way_out}"
                )
            with Trajectory(path, "w" if written < 0 else "a") as trajectory:
                self._write_frame(trajectory, runtime.atoms, written)
                sync_directory(path.parent)
                while self._global_step < total_steps and self._phase == "running":
                    steps = min(self.config.observation_interval, total_steps - self._global_step)
                    before = int(runtime.nsteps)
                    runtime.run(steps)
                    _require_finite(runtime.snapshot())
                    advanced = int(runtime.nsteps) - before
                    if advanced != steps:
                        raise RuntimeError(
                            f"exact runtime advanced {advanced} steps; expected {steps}"
                        )
                    self._observe(runtime.atoms, self._global_step + steps, total_steps)
                    self._write_frame(trajectory, runtime.atoms, written)
                    # A checkpoint survives a crash, so the frames written before it must too.
                    flush_to_disk(path)
                    self._checkpoint(runtime)

    def _record_environment_change(self, resumed: ExactRestartSnapshot) -> None:
        """Log a restore under other software or hardware than the checkpoint recorded."""

        assert self._snapshot is not None
        before, after = _environment(self._snapshot), _environment(resumed)
        changed = {
            key: [before.get(key), after.get(key)]
            for key in sorted(before.keys() | after.keys())
            if before.get(key) != after.get(key)
        }
        if not changed:
            return
        record = {"generation": self._generation, "step": self._global_step, "changed": changed}
        path = self.root / "restart-environment.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        flush_to_disk(path)
        warnings.warn(
            f"resumed at step {self._global_step} with a different environment, so the run is "
            f"no longer bitwise exact: {', '.join(changed)} (recorded in {path.name})",
            stacklevel=2,
        )

    def _saved_frames(self, first: int, last: int) -> dict[int, Atoms]:
        """The saved observations from global frame ``first`` to ``last`` that are on disk."""

        frames: dict[int, Atoms] = {}
        for path in sorted(self.root.glob("segments/*/trajectory.traj")):
            if path.stat().st_size == 0:
                continue
            try:
                trajectory = Trajectory(path)
            except Exception:  # cut, which run() reports: refinement can do without it
                continue
            with trajectory:
                try:
                    # A generation holds one frame per observation, from the one it resumed at.
                    start = int(trajectory[0].info["reactionflow_global_frame"])
                except Exception:
                    continue
                for frame in range(max(first, start), min(last, start + len(trajectory) - 1) + 1):
                    try:
                        atoms = trajectory[frame - start]
                    except Exception:
                        break
                    if int(atoms.info["reactionflow_global_frame"]) != frame:
                        raise ValueError(f"{path} does not hold one frame per observation")
                    frames.setdefault(frame, canonical_copy(atoms))
        return frames

    def _publish_outcome(self, occurrence_id: str, outcome: PathwayOutcome) -> None:
        final = self.pathways / occurrence_id
        temporary = self.pathways / f".{occurrence_id}-{uuid4().hex}.tmp"
        with publishing(temporary, final):
            identity = {
                "schema_version": 1,
                "occurrence_id": occurrence_id,
                "reactionflow_version": __version__,
            }
            class_id = self.occurrences.record(occurrence_id).class_id
            _write_result(temporary, outcome, {**identity, "class_id": class_id})
            for number, step in enumerate(outcome.steps, 1):
                bonds = {
                    "reactant_bonds": sorted(map(list, step.reactant_bonds)),
                    "product_bonds": sorted(map(list, step.product_bonds)),
                }
                folder = temporary / f"step-{number}"
                folder.mkdir()
                _write_result(folder, step.outcome, {**identity, "step": number, **bonds})

    def _time_limited(
        self, provider: CalculatorProvider, max_hours: float | None
    ) -> CalculatorProvider:
        """A provider that ends a refinement as ``time_limit`` once it has run for ``max_hours``.

        The time is checked before each model evaluation and added up in ``state.json``, because an
        interrupted refinement starts over when it resumes: a job shorter than the refinement would
        otherwise repeat it forever.
        """

        def check() -> None:
            before = self._barrier_seconds
            now = time.monotonic()
            self._barrier_seconds += now - self._barrier_clock
            self._barrier_clock = now
            if max_hours is not None and self._barrier_seconds >= max_hours * 3600:
                raise BarrierTimeLimit(f"stopped at the limit of {max_hours:g} h per barrier")
            if self._barrier_seconds // 60 > before // 60:  # save once a minute
                with suppress(OSError):  # this runs inside the refinement, whose result it would be
                    self._write_state()

        @contextmanager
        def lease(stage: str) -> Iterator[Any]:
            check()
            with provider(stage) as calculator:
                calculate = calculator.calculate

                def checked(*args: Any, **kwargs: Any) -> Any:
                    check()
                    return calculate(*args, **kwargs)

                calculator.calculate = checked
                try:
                    yield calculator
                finally:
                    calculator.calculate = calculate

        return lease

    def _refine_pending(
        self, calculator_provider: CalculatorProvider, pressure_GPa: float | None
    ) -> None:
        """Refine each queued occurrence serially and publish its immutable result.

        Returns early after a refinement with no barrier, so that the caller can check the model.
        """

        with self._fatal("refine"):
            while self._pending:
                occurrence_id = self._pending[0]
                no_barrier = False
                if (self.pathways / occurrence_id).is_dir():
                    # Published before an interruption; make its name durable and move on.
                    sync_directory(self.pathways)
                else:
                    self._barrier_clock = time.monotonic()
                    candidate = self.occurrences.load(occurrence_id)
                    outcome = refine_pathway(
                        candidate,
                        calculator_provider=calculator_provider,
                        config=self.config.pathway,
                        detector_config=self.occurrences.load_detector_config(occurrence_id),
                        pressure_GPa=pressure_GPa,
                        frames=self._saved_frames(
                            candidate.reactant_frame - self.config.pathway.reactant_lookback,
                            candidate.product_frame,
                        ),
                    )
                    self._publish_outcome(occurrence_id, outcome)
                    no_barrier = outcome.barrier is None
                self._pending.pop(0)
                self._barrier_seconds = 0.0
                self._write_state()
                if no_barrier:
                    return

    def _check_model(self, runtime_provider: ExactRuntimeProvider) -> None:
        """Restore the last checkpoint, which loads and evaluates the model, or raise.

        Reads a separate snapshot so that validation cannot mutate the run's checkpoint.
        """

        assert self._active_checkpoint is not None
        snapshot = ExactRestartSnapshot.read(self._active_checkpoint / "exact-restart")
        with runtime_provider.restore(snapshot) as runtime:
            if runtime.atoms.calc is not None and not np.isfinite(runtime.atoms.get_forces()).all():
                raise ValueError("the model returned non-finite forces at the last checkpoint")

    def _refine_due(self) -> bool:
        """Whether to refine now: the queue waits only for running MD under ``refine_after_md``."""

        return bool(self._pending) and (self._phase != "running" or not self._refine_after_md)

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
        max_hours_per_barrier: float | None = DEFAULT_MAX_HOURS_PER_BARRIER,
        total_steps: int,
        pressure_GPa: float | None = None,
        refine_after_md: bool = False,
    ) -> RunSummary:
        """Run or resume live detection, serial NEB/CI-NEB, and exact MD to a step target.

        Refining one event stops after ``max_hours_per_barrier`` hours, or never with ``None``.
        """

        if isinstance(total_steps, bool) or not isinstance(total_steps, Integral):
            raise ValueError("total_steps must be a non-negative integer")
        if total_steps < self._global_step:
            raise ValueError("total_steps cannot precede the durable run counter")
        pathway_calculator_provider = self._time_limited(
            pathway_calculator_provider, max_hours_per_barrier
        )
        if self._phase == "new":
            if atoms is None:
                raise ValueError("initial atoms are required for a new run")
            self._start(atoms)
        elif atoms is not None:
            raise ValueError("initial atoms may only be supplied to a new run")
        self._refine_after_md = refine_after_md
        lost = missing_frames(self.root, self._global_frame)
        if lost:
            warnings.warn(
                f"saved frames {frame_ranges(lost)} of {self.root.name} are missing or unreadable. "
                "The MD state comes from the checkpoint and is not affected, but scan cannot "
                "replay those frames and refinement cannot use them",
                stacklevel=2,
            )

        if self._phase == "refining" or self._refine_due():
            # A reopened adapter has not checked its model against the saved MD state yet.
            # Verify before any pathway result can be published, including terminal work
            # that will never resume MD.
            self._check_model(runtime_provider)

        # Only failures to maintain the durable record mark the run failed. Any other error, such
        # as a runtime that this environment refuses to restore, leaves the last exact checkpoint
        # in place, so the run resumes once the cause is fixed.
        while self._phase != "completed":
            if self._refine_due():
                self._refine_pending(pathway_calculator_provider, pressure_GPa)
                if self._pending:
                    # No barrier. If the model is gone, stop before it fails the rest of the queue.
                    self._check_model(runtime_provider)
            elif self._phase == "failed":
                raise RuntimeError(f"ReactionRun failed: {self._failure}")
            elif self._global_step >= total_steps:
                self._complete()
            else:
                if self._phase == "refining":
                    # Refinement is finished; MD continues in a new trajectory generation.
                    self._generation += 1
                    self._phase = "running"
                self._run_segment(int(total_steps), runtime_provider)
        return self.summary()


def _environment(snapshot: ExactRestartSnapshot) -> dict[str, Any]:
    versions = ("ase_version", "numpy_version")
    dynamics = {f"dynamics_{key}": snapshot.dynamics.metadata.get(key) for key in versions}
    return {**snapshot.calculator.metadata, **dynamics}


__all__ = ["ReactionRun", "ReactionRunConfig", "RunSummary"]
