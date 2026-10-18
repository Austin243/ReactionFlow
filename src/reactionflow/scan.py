"""Replay saved frames through bond detection and reaction tracking, without the model."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.formula import Formula
from ase.io import iread
from ase.io.trajectory import Trajectory

from .candidates import ReactionCandidate, ReactionClasses, ReactionTracker
from .detection import Bond, BondChangeDetector, BondDetectorConfig, atom_ids, canonical_copy
from .status import _describe

type Frame = tuple[int, float | None, Atoms]
type OccurrenceKey = tuple[tuple[int, int, int], frozenset[Bond]]

_TABLES = {
    "bond_changes": [
        "first_seen_frame",
        "confirmed_frame",
        "time_ps",
        "event",
        "atom_a",
        "atom_b",
        "elements",
        "distance_A",
        "between_molecules",
    ],
    "candidates": [
        "reactant_frame",
        "product_frame",
        "observed_frame",
        "reactant_time_ps",
        "product_time_ps",
        "resolved",
        "at_end",
        "atoms",
        "changes",
        "class",
        "class_label",
        "occurrence_id",
    ],
    "species": ["frame", "time_ps", "formula", "molecules", "count"],
}


@dataclass(frozen=True, slots=True)
class ScanResult:
    """The rows of each scan table and the counts that summarize them."""

    bond_changes: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    species: list[dict[str, Any]]
    summary: dict[str, Any]

    def write(self, directory: str | Path) -> Path:
        """Write one CSV file per table and ``summary.json`` into ``directory``."""

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for name, fields in _TABLES.items():
            with (path / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(getattr(self, name))
        text = json.dumps(self.summary, indent=2, sort_keys=True) + "\n"
        (path / "summary.json").write_text(text, encoding="utf-8")
        return path


def run_frames(root: str | Path, *, timestep_fs: float | None = None) -> Iterator[Frame]:
    """Yield each saved observation of a ``ReactionRun`` once, in order.

    A new trajectory generation starts with the frame it resumed from, which is skipped. Reading
    stops at the first frame that cannot be read, such as the one a running job is writing.
    """

    segments = sorted(
        (path for path in (Path(root) / "segments").iterdir() if path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    last = -1
    for segment in segments:
        path = segment / "trajectory.traj"
        if not path.is_file() or path.stat().st_size == 0:
            return
        with Trajectory(path) as trajectory:
            for index in range(len(trajectory)):
                try:
                    atoms = trajectory[index]
                except Exception:
                    return
                frame = int(atoms.info["reactionflow_global_frame"])
                if frame <= last:
                    continue
                last = frame
                step = int(atoms.info["reactionflow_global_step"])
                time_ps = None if timestep_fs is None else step * timestep_fs / 1000
                yield frame, time_ps, canonical_copy(atoms)


def file_frames(path: str | Path, *, frame_ps: float | None = None) -> Iterator[Frame]:
    """Yield every frame of a trajectory file ASE can read, with IDs 0 to N-1 if it has none."""

    for frame, atoms in enumerate(iread(path, index=":")):
        yield frame, None if frame_ps is None else frame * frame_ps, canonical_copy(atoms)


def recorded_occurrences(root: str | Path) -> dict[OccurrenceKey, str]:
    """The candidates a run recorded, by frames and changed bonds, read without writing."""

    recorded = {}
    for path in sorted(Path(root).glob("candidates/*/candidate.json")):
        if path.parent.name.startswith("."):
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        data = metadata["candidate"]
        changes = {tuple(bond) for bond in data["reactant_bonds"]} ^ {
            tuple(bond) for bond in data["product_bonds"]
        }
        frames = (data["reactant_frame"], data["product_frame"], data["observed_frame"])
        recorded[(frames, frozenset(changes))] = metadata["occurrence_id"]
    return recorded


def _components(atoms: Iterable[int], bonds: Iterable[Bond]) -> dict[int, int]:
    """Map each atom ID to the smallest atom ID of its bonded component."""

    parent = {atom_id: atom_id for atom_id in atoms}

    def root(atom_id: int) -> int:
        while parent[atom_id] != atom_id:
            parent[atom_id] = parent[parent[atom_id]]
            atom_id = parent[atom_id]
        return atom_id

    for first, second in bonds:
        first, second = root(first), root(second)
        parent[max(first, second)] = min(first, second)
    return {atom_id: root(atom_id) for atom_id in parent}


def _changes(candidate: ReactionCandidate, symbols: Mapping[int, str]) -> str:
    return "; ".join(
        f"{kind} {symbols[first]}{first}-{symbols[second]}{second}"
        for kind, bonds in (
            ("formed", candidate.product_bonds - candidate.reactant_bonds),
            ("broken", candidate.reactant_bonds - candidate.product_bonds),
        )
        for first, second in sorted(bonds)
    )


def scan_frames(
    frames: Iterable[Frame],
    *,
    detector_config: BondDetectorConfig,
    stability_frames: int = 3,
    region_radius: int | None = 2,
    species_every: int = 10,
    occurrences: Mapping[OccurrenceKey, str] | None = None,
) -> ScanResult:
    """Replay ordered frames through a new detector and tracker and tabulate what they find.

    ``frames`` yields ``(frame, time_ps, atoms)`` with stable atom IDs and increasing frame
    numbers, as ``run_frames`` and ``file_frames`` do. Molecules are the bonded components of the
    first frame, and candidates fall into the local classes ``reactionflow status`` reports.
    ``occurrences``, from ``recorded_occurrences``, names the candidates a run recorded itself.
    """

    if species_every < 1:
        raise ValueError("species_every must be a positive integer")
    detector = BondChangeDetector(detector_config)
    tracker = ReactionTracker(stability_frames=stability_frames, region_radius=region_radius)
    classes = ReactionClasses(radius=2)
    labels: list[str] = []
    times: dict[int, float | None] = {}
    symbols: dict[int, str] = {}
    molecule: dict[int, int] = {}
    bond_changes: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    species: list[dict[str, Any]] = []
    waiting: frozenset[tuple[Bond, str]] = frozenset()
    reverted = net_changes = 0

    def add(candidate: ReactionCandidate, *, at_end: bool) -> None:
        nonlocal net_changes
        name = classes.find(candidate)
        if name is None:
            labels.append(_describe(candidate, 2))
            name = str(len(labels))
            classes.add(name, candidate)
        changed = candidate.reactant_bonds ^ candidate.product_bonds
        if candidate.resolved:
            net_changes += len(changed)
        window = (candidate.reactant_frame, candidate.product_frame, candidate.observed_frame)
        candidates.append(
            {
                "reactant_frame": candidate.reactant_frame,
                "product_frame": candidate.product_frame,
                "observed_frame": candidate.observed_frame,
                "reactant_time_ps": times[candidate.reactant_frame],
                "product_time_ps": times[candidate.product_frame],
                "resolved": candidate.resolved,
                "at_end": at_end,
                "atoms": len(candidate.atom_ids),
                "changes": _changes(candidate, symbols),
                "class": int(name),
                "class_label": labels[int(name) - 1],
                "occurrence_id": (occurrences or {}).get((window, frozenset(changed)), ""),
            }
        )

    for frame, time_ps, atoms in frames:
        times[frame] = time_ps
        events = detector.process(atoms, frame=frame)
        if not symbols:
            symbols = dict(zip(atom_ids(atoms), atoms.get_chemical_symbols(), strict=True))
            molecule = _components(symbols, detector.stable_bonds)
        # A crossing that leaves the pending list without being confirmed reverted in time.
        pending = detector.pending_changes
        confirmed = {(event.atom_ids, event.event_type) for event in events}
        reverted += len(waiting - pending - confirmed)
        waiting = pending
        for event in events:
            first, second = event.atom_ids
            bond_changes.append(
                {
                    "first_seen_frame": event.first_seen_frame,
                    "confirmed_frame": event.confirmed_frame,
                    "time_ps": times[event.first_seen_frame],
                    "event": event.event_type,
                    "atom_a": first,
                    "atom_b": second,
                    "elements": "-".join(event.symbols),
                    "distance_A": round(event.distance, 4),
                    "between_molecules": molecule[first] != molecule[second],
                }
            )
        for candidate in tracker.process(
            atoms,
            frame=frame,
            stable_bonds=detector.stable_bonds,
            pending_bonds=detector.pending_bonds,
        ):
            add(candidate, at_end=False)
        if frame % species_every == 0:
            members: dict[int, list[int]] = {}
            for atom_id, root in _components(symbols, detector.stable_bonds).items():
                members.setdefault(root, []).append(atom_id)
            counts = Counter(
                (
                    Formula.from_list([symbols[atom_id] for atom_id in group]).format("hill"),
                    len({molecule[atom_id] for atom_id in group}),
                )
                for group in members.values()
            )
            species.extend(
                {
                    "frame": frame,
                    "time_ps": time_ps,
                    "formula": formula,
                    "molecules": size,
                    "count": count,
                }
                for (formula, size), count in sorted(counts.items())
            )
    for candidate in tracker.finish():
        add(candidate, at_end=True)

    resolved = sum(row["resolved"] for row in candidates)
    summary: dict[str, Any] = {
        "frames": len(times),
        "first_frame": min(times, default=None),
        "last_frame": max(times, default=None),
        "reverted_crossings": reverted,
        "bonds_formed": sum(row["event"] == "formed" for row in bond_changes),
        "bonds_broken": sum(row["event"] == "broken" for row in bond_changes),
        "candidates_resolved": resolved,
        "candidates_unresolved": len(candidates) - resolved,
        "still_changing_at_end": sum(row["at_end"] for row in candidates),
        "net_changes_in_resolved": net_changes,
        "classes_in_resolved": len({row["class"] for row in candidates if row["resolved"]}),
        "settings": {
            "detector": detector_config.to_dict(),
            "candidate_stability_frames": stability_frames,
            "candidate_region_radius": region_radius,
            "species_every": species_every,
        },
    }
    if occurrences is not None:
        summary["recorded_by_run"] = len(occurrences)
        summary["matching_recorded"] = sum(bool(row["occurrence_id"]) for row in candidates)
    return ScanResult(bond_changes, candidates, species, summary)


__all__ = ["ScanResult", "file_frames", "recorded_occurrences", "run_frames", "scan_frames"]
