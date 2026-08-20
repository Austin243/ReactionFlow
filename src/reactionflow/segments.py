"""Structural checkpoints and immutable trajectory generations."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
from ase import Atoms
from ase.io import read, write

from .detection import assign_atom_ids, atom_ids
from .restart import ExactRestartSnapshot


@dataclass(frozen=True, slots=True)
class SegmentGeneration:
    """One calculator-free starting state and its dedicated trajectory path."""

    generation: int
    directory: Path
    trajectory_path: Path
    atoms: Atoms
    global_step: int
    global_frame: int
    _identity: tuple[tuple[int, int], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ResumeToken:
    """Pointer to one complete structural or exact checkpoint."""

    path: Path
    source_generation: int
    global_step: int
    global_frame: int
    fidelity: str = "structural"

    @property
    def checkpoint_path(self) -> Path:
        return self.path.parent / "atoms.traj"

    @property
    def next_generation(self) -> int:
        return self.source_generation + 1

    @property
    def exact_restart_path(self) -> Path | None:
        if self.fidelity != "exact":
            return None
        return self.path.parent / "exact-restart"

    @classmethod
    def read(cls, path: str | Path) -> ResumeToken:
        path = Path(path).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        fidelity = value.get("fidelity")
        if (
            value.get("schema_version") != 1
            or not isinstance(fidelity, str)
            or fidelity not in {"structural", "exact"}
        ):
            raise ValueError("unsupported resume token")
        source = _counter(value.get("source_generation"), "source_generation")
        step = _counter(value.get("global_step"), "global_step")
        frame = _counter(value.get("global_frame"), "global_frame")
        return cls(path, source, step, frame, fidelity)


def _counter(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _identity(atoms: Atoms) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(zip(atom_ids(atoms), map(int, atoms.numbers), strict=True)))


def _canonical_snapshot(atoms: Atoms) -> Atoms:
    snapshot = atoms.copy()
    snapshot.calc = None
    snapshot.info.pop("atom_ids", None)
    return snapshot


def _transport_snapshot(atoms: Atoms) -> Atoms:
    snapshot = _canonical_snapshot(atoms)
    snapshot.info["atom_ids"] = list(atom_ids(snapshot))
    return snapshot


def _same_atomic_state(first: Atoms, second: Atoms) -> bool:
    if set(first.arrays) != set(second.arrays):
        return False
    return (
        all(np.array_equal(first.arrays[name], second.arrays[name]) for name in first.arrays)
        and np.array_equal(first.cell.array, second.cell.array)
        and np.array_equal(first.pbc, second.pbc)
    )


def _write_token(path: Path, token: ResumeToken) -> None:
    value = {
        "schema_version": 1,
        "fidelity": token.fidelity,
        "source_generation": token.source_generation,
        "global_step": token.global_step,
        "global_frame": token.global_frame,
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SegmentStore:
    """Create structural checkpoints without reopening prior trajectories."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.segments = self.root / "segments"
        self.segments.mkdir(parents=True, exist_ok=True)

    def _generation(
        self,
        generation: int,
        atoms: Atoms,
        global_step: int,
        global_frame: int,
        *,
        recover_empty: bool = False,
    ) -> SegmentGeneration:
        directory = self.segments / f"{generation:04d}"
        try:
            directory.mkdir()
        except FileExistsError:
            if not recover_empty or any(directory.iterdir()):
                raise
        return SegmentGeneration(
            generation=generation,
            directory=directory,
            trajectory_path=directory / "trajectory.traj",
            atoms=atoms,
            global_step=global_step,
            global_frame=global_frame,
            _identity=_identity(atoms),
        )

    def start(
        self,
        atoms: Atoms,
        *,
        global_step: int = 0,
        global_frame: int = 0,
    ) -> SegmentGeneration:
        """Claim generation zero for a new run."""

        step = _counter(global_step, "global_step")
        frame = _counter(global_frame, "global_frame")
        initial = atoms.copy()
        assign_atom_ids(initial)
        return self._generation(0, _canonical_snapshot(initial), step, frame)

    def attach(
        self,
        generation: int,
        atoms: Atoms,
        *,
        global_step: int,
        global_frame: int,
    ) -> SegmentGeneration:
        """Attach a durable exact snapshot to an existing active generation."""

        generation = _counter(generation, "generation")
        step = _counter(global_step, "global_step")
        frame = _counter(global_frame, "global_frame")
        directory = self.segments / f"{generation:04d}"
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        snapshot = _canonical_snapshot(atoms)
        assign_atom_ids(snapshot)
        return SegmentGeneration(
            generation=generation,
            directory=directory,
            trajectory_path=directory / "trajectory.traj",
            atoms=snapshot,
            global_step=step,
            global_frame=frame,
            _identity=_identity(snapshot),
        )

    def bind(self, segment: SegmentGeneration, atoms: Atoms) -> SegmentGeneration:
        """Bind an acquired runtime's live atoms object to an active generation."""

        expected = self.segments / f"{segment.generation:04d}"
        if segment.directory != expected or not expected.is_dir():
            raise ValueError("segment does not belong to this store")
        if _identity(atoms) != segment._identity:
            raise ValueError("runtime atom identities do not match the segment")
        return SegmentGeneration(
            generation=segment.generation,
            directory=segment.directory,
            trajectory_path=segment.trajectory_path,
            atoms=atoms,
            global_step=segment.global_step,
            global_frame=segment.global_frame,
            _identity=segment._identity,
        )

    def checkpoint(
        self,
        segment: SegmentGeneration,
        atoms: Atoms,
        *,
        global_step: int,
        global_frame: int,
        exact_restart: ExactRestartSnapshot | None = None,
    ) -> ResumeToken:
        """Atomically publish one structural or exact checkpoint for a generation."""

        expected = self.segments / f"{segment.generation:04d}"
        if segment.directory != expected or not expected.is_dir():
            raise ValueError("segment does not belong to this store")
        step = _counter(global_step, "global_step")
        frame = _counter(global_frame, "global_frame")
        if step < segment.global_step or frame < segment.global_frame:
            raise ValueError("checkpoint counters cannot move backwards")
        if _identity(atoms) != segment._identity:
            raise ValueError("checkpoint atom identities do not match the segment")
        if exact_restart is not None:
            if _identity(exact_restart.atoms) != segment._identity:
                raise ValueError("exact-restart atom identities do not match the segment")
            if not _same_atomic_state(atoms, exact_restart.atoms):
                raise ValueError("exact-restart atoms do not match the checkpoint boundary")

        final = expected / "checkpoint"
        if final.exists():
            raise FileExistsError(f"generation {segment.generation} is already checkpointed")
        temporary = expected / f".checkpoint-{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            checkpoint = temporary / "atoms.traj"
            write(checkpoint, _transport_snapshot(atoms), format="traj")
            loaded = read(checkpoint)
            atom_ids(loaded)
            assign_atom_ids(loaded)
            fidelity = "structural"
            if exact_restart is not None:
                exact_restart.write(temporary / "exact-restart")
                ExactRestartSnapshot.read(temporary / "exact-restart")
                fidelity = "exact"
            token = ResumeToken(
                temporary / "resume.json",
                segment.generation,
                step,
                frame,
                fidelity,
            )
            _write_token(token.path, token)
            os.replace(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return ResumeToken(
            final / "resume.json",
            segment.generation,
            step,
            frame,
            fidelity,
        )

    def read_exact(self, token: ResumeToken) -> ExactRestartSnapshot:
        """Load the exact runtime snapshot referenced by a token."""

        expected = self.segments / f"{token.source_generation:04d}" / "checkpoint" / "resume.json"
        if token.path != expected:
            raise ValueError("resume token does not belong to this store")
        persisted = ResumeToken.read(expected)
        if persisted != token or token.fidelity != "exact":
            raise ValueError("resume token does not reference an exact checkpoint")
        assert token.exact_restart_path is not None
        snapshot = ExactRestartSnapshot.read(token.exact_restart_path)
        if _identity(snapshot.atoms) != _identity(read(token.checkpoint_path)):
            raise ValueError("exact checkpoint atom identities do not match")
        return snapshot

    def resume(
        self,
        token: ResumeToken,
        *,
        recover_empty: bool = False,
    ) -> SegmentGeneration:
        """Restore a checkpoint into a new immutable trajectory generation."""

        expected = self.segments / f"{token.source_generation:04d}" / "checkpoint" / "resume.json"
        if token.path != expected:
            raise ValueError("resume token does not belong to this store")
        persisted = ResumeToken.read(expected)
        if persisted != token:
            raise ValueError("resume token conflicts with its persisted record")
        atoms = read(token.checkpoint_path)
        atom_ids(atoms)
        assign_atom_ids(atoms)
        atoms.info.pop("atom_ids", None)
        atoms.calc = None
        return self._generation(
            token.next_generation,
            atoms,
            token.global_step,
            token.global_frame,
            recover_empty=recover_empty,
        )


__all__ = ["ResumeToken", "SegmentGeneration", "SegmentStore"]
