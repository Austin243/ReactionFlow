"""Structural checkpoints and immutable trajectory generations."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from ase import Atoms
from ase.io import read, write

from .detection import assign_atom_ids, atom_ids


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
    """Pointer to one complete structural checkpoint."""

    path: Path
    source_generation: int
    global_step: int
    global_frame: int

    @property
    def checkpoint_path(self) -> Path:
        return self.path.parent / "atoms.traj"

    @property
    def next_generation(self) -> int:
        return self.source_generation + 1

    @classmethod
    def read(cls, path: str | Path) -> ResumeToken:
        path = Path(path).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or value.get("fidelity") != "structural":
            raise ValueError("unsupported resume token")
        source = _counter(value.get("source_generation"), "source_generation")
        step = _counter(value.get("global_step"), "global_step")
        frame = _counter(value.get("global_frame"), "global_frame")
        return cls(path, source, step, frame)


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


def _write_token(path: Path, token: ResumeToken) -> None:
    value = {
        "schema_version": 1,
        "fidelity": "structural",
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

    def checkpoint(
        self,
        segment: SegmentGeneration,
        atoms: Atoms,
        *,
        global_step: int,
        global_frame: int,
    ) -> ResumeToken:
        """Atomically publish one structural checkpoint for a generation."""

        expected = self.segments / f"{segment.generation:04d}"
        if segment.directory != expected or not expected.is_dir():
            raise ValueError("segment does not belong to this store")
        step = _counter(global_step, "global_step")
        frame = _counter(global_frame, "global_frame")
        if step < segment.global_step or frame < segment.global_frame:
            raise ValueError("checkpoint counters cannot move backwards")
        if _identity(atoms) != segment._identity:
            raise ValueError("checkpoint atom identities do not match the segment")

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
            token = ResumeToken(
                temporary / "resume.json",
                segment.generation,
                step,
                frame,
            )
            _write_token(token.path, token)
            os.replace(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return ResumeToken(final / "resume.json", segment.generation, step, frame)

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
