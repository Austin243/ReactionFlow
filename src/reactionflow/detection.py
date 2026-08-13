"""Persistent geometric bond-change detection for ASE trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from ase.neighborlist import neighbor_list

type Bond = tuple[int, int]
type ElementPair = tuple[str, str]
type EventType = Literal["formed", "broken"]


def _element_pair(value: object) -> ElementPair:
    parts = value.split("-") if isinstance(value, str) else value
    if not isinstance(parts, (tuple, list)) or len(parts) != 2:
        raise ValueError("pair keys must look like ('C', 'O') or 'C-O'")
    first, second = map(str, parts)
    if first not in atomic_numbers or second not in atomic_numbers:
        raise ValueError(f"invalid element pair {value!r}")
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _bond(first: int, second: int) -> Bond:
    if first == second:
        raise ValueError("a bond must connect two distinct atom IDs")
    return (first, second) if first < second else (second, first)


def atom_ids(atoms: Atoms) -> tuple[int, ...]:
    """Return the stable integer IDs carried by an ASE frame."""

    raw = atoms.arrays.get("atom_id")
    if raw is None:
        raw = atoms.info.get("atom_ids")
    if raw is None:
        raise ValueError("stable atom IDs are required")
    if getattr(raw, "dtype", None) is not None and not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("atom IDs must be integers")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in raw):
        raise ValueError("atom IDs must be integers")
    ids = tuple(int(value) for value in raw)
    if len(ids) != len(atoms) or len(set(ids)) != len(ids):
        raise ValueError("each atom needs one unique stable ID")
    return ids


def assign_atom_ids(atoms: Atoms) -> Atoms:
    """Assign missing IDs once and store them in the canonical ASE array."""

    try:
        ids = atom_ids(atoms)
    except ValueError:
        if "atom_id" in atoms.arrays or "atom_ids" in atoms.info:
            raise
        ids = tuple(range(len(atoms)))
    atoms.set_array("atom_id", np.asarray(ids, dtype=np.int64))
    return atoms


def _threshold(value: object) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise TypeError("a pair threshold must be (form_distance, break_distance)")
    form, breaking = map(float, value)
    if not 0 < form < breaking:
        raise ValueError("bond thresholds must satisfy 0 < form < break")
    return form, breaking


@dataclass(frozen=True)
class BondDetectorConfig:
    """Distance and persistence settings for bond-change detection."""

    form_scale: float = 1.15
    break_scale: float = 1.30
    persistence_frames: int = 3
    pair_thresholds: Mapping[object, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        form = float(self.form_scale)
        breaking = float(self.break_scale)
        if not 0 < form < breaking:
            raise ValueError("scales must satisfy 0 < form_scale < break_scale")
        if (
            isinstance(self.persistence_frames, bool)
            or not isinstance(self.persistence_frames, Integral)
            or self.persistence_frames < 1
        ):
            raise ValueError("persistence_frames must be positive")
        thresholds = {
            _element_pair(pair): _threshold(threshold)
            for pair, threshold in self.pair_thresholds.items()
        }
        object.__setattr__(self, "form_scale", form)
        object.__setattr__(self, "break_scale", breaking)
        object.__setattr__(self, "pair_thresholds", thresholds)

    def threshold_for(self, first: str, second: str) -> tuple[float, float]:
        pair = _element_pair((first, second))
        if pair in self.pair_thresholds:
            return self.pair_thresholds[pair]  # type: ignore[index,return-value]
        radius = covalent_radii[atomic_numbers[first]] + covalent_radii[atomic_numbers[second]]
        return self.form_scale * radius, self.break_scale * radius

    def to_dict(self) -> dict[str, Any]:
        return {
            "form_scale": self.form_scale,
            "break_scale": self.break_scale,
            "persistence_frames": self.persistence_frames,
            "pair_thresholds": {
                "-".join(pair): list(threshold)
                for pair, threshold in sorted(self.pair_thresholds.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BondDetectorConfig:
        return cls(
            form_scale=value.get("form_scale", 1.15),
            break_scale=value.get("break_scale", 1.30),
            persistence_frames=value.get("persistence_frames", 3),
            pair_thresholds=value.get("pair_thresholds", {}),
        )


@dataclass(frozen=True)
class BondEvent:
    """A persistent formation or break confirmed by the detector."""

    event_type: EventType
    atom_ids: Bond
    symbols: ElementPair
    distance: float
    form_distance: float
    break_distance: float
    first_seen_frame: int
    confirmed_frame: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "atom_ids": list(self.atom_ids),
            "symbols": list(self.symbols),
            "distance": self.distance,
            "form_distance": self.form_distance,
            "break_distance": self.break_distance,
            "first_seen_frame": self.first_seen_frame,
            "confirmed_frame": self.confirmed_frame,
        }


@dataclass
class _PendingChange:
    first_seen_frame: int
    count: int = 0


class BondChangeDetector:
    """Detect persistent geometric bond changes; the first frame is the baseline."""

    def __init__(self, config: BondDetectorConfig | None = None) -> None:
        self.config = config or BondDetectorConfig()
        self._numbers: dict[int, int] | None = None
        self._stable: set[Bond] = set()
        self._pending: dict[tuple[Bond, EventType], _PendingChange] = {}
        self._last_frame: int | None = None

    @property
    def stable_bonds(self) -> frozenset[Bond]:
        return frozenset(self._stable)

    @property
    def last_frame(self) -> int | None:
        return self._last_frame

    def _threshold(self, bond: Bond) -> tuple[float, float]:
        assert self._numbers is not None
        return self.config.threshold_for(
            chemical_symbols[self._numbers[bond[0]]],
            chemical_symbols[self._numbers[bond[1]]],
        )

    def _distances(self, atoms: Atoms, ids: tuple[int, ...]) -> dict[Bond, float]:
        if len(atoms) < 2:
            return {}
        symbols = set(atoms.get_chemical_symbols())
        cutoff = max(
            self.config.threshold_for(first, second)[1] for first in symbols for second in symbols
        )
        first, second, values = neighbor_list("ijd", atoms, cutoff)
        distances: dict[Bond, float] = {}
        for i, j, distance in zip(first, second, values, strict=True):
            if i == j:
                continue
            bond = _bond(ids[int(i)], ids[int(j)])
            distances[bond] = min(float(distance), distances.get(bond, float("inf")))

        indices = {atom_id: index for index, atom_id in enumerate(ids)}
        for bond in self._stable - distances.keys():
            distances[bond] = float(
                atoms.get_distance(
                    indices[bond[0]],
                    indices[bond[1]],
                    mic=bool(atoms.pbc.any()),
                )
            )
        return distances

    def process(self, atoms: Atoms, *, frame: int) -> list[BondEvent]:
        """Process one ordered frame and return newly confirmed events."""

        if self._last_frame is not None and frame <= self._last_frame:
            raise ValueError("frame numbers must increase")
        ids = atom_ids(atoms)
        numbers = dict(zip(ids, map(int, atoms.numbers), strict=True))
        if self._numbers is not None and numbers != self._numbers:
            raise ValueError("atom identities changed between frames")
        if self._numbers is None:
            self._numbers = numbers

        distances = self._distances(atoms, ids)
        if self._last_frame is None:
            self._stable = {
                bond for bond, distance in distances.items() if distance <= self._threshold(bond)[0]
            }
            self._last_frame = frame
            return []

        active: set[tuple[Bond, EventType]] = set()
        events: list[BondEvent] = []
        for bond, distance in sorted(distances.items()):
            form_distance, break_distance = self._threshold(bond)
            event_type: EventType | None = None
            if bond in self._stable and distance >= break_distance:
                event_type = "broken"
            elif bond not in self._stable and distance <= form_distance:
                event_type = "formed"
            if event_type is None:
                continue

            key = (bond, event_type)
            active.add(key)
            pending = self._pending.setdefault(key, _PendingChange(frame))
            pending.count += 1
            if pending.count < self.config.persistence_frames:
                continue

            if event_type == "formed":
                self._stable.add(bond)
            else:
                self._stable.remove(bond)
            events.append(
                BondEvent(
                    event_type=event_type,
                    atom_ids=bond,
                    symbols=(
                        chemical_symbols[numbers[bond[0]]],
                        chemical_symbols[numbers[bond[1]]],
                    ),
                    distance=distance,
                    form_distance=form_distance,
                    break_distance=break_distance,
                    first_seen_frame=pending.first_seen_frame,
                    confirmed_frame=frame,
                )
            )
            self._pending.pop(key)

        for key in set(self._pending) - active:
            self._pending.pop(key)
        self._last_frame = frame
        return events

    def export_state(self) -> dict[str, Any]:
        """Return a JSON-serializable checkpoint."""

        return {
            "schema_version": 1,
            "config": self.config.to_dict(),
            "numbers": (
                None if self._numbers is None else [list(item) for item in self._numbers.items()]
            ),
            "stable_bonds": [list(bond) for bond in sorted(self._stable)],
            "pending": [
                {
                    "atom_ids": list(bond),
                    "event_type": event_type,
                    "first_seen_frame": pending.first_seen_frame,
                    "count": pending.count,
                }
                for (bond, event_type), pending in self._pending.items()
            ],
            "last_frame": self._last_frame,
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        config: BondDetectorConfig | None = None,
    ) -> BondChangeDetector:
        """Restore a detector from a version-1 checkpoint."""

        if state.get("schema_version") != 1:
            raise ValueError("unsupported detector checkpoint version")
        stored_config = BondDetectorConfig.from_dict(state.get("config", {}))
        if config is not None and config.to_dict() != stored_config.to_dict():
            raise ValueError("checkpoint configuration does not match")
        detector = cls(config or stored_config)
        numbers = state.get("numbers")
        detector._numbers = (
            None if numbers is None else {int(key): int(value) for key, value in numbers}
        )
        detector._stable = {
            _bond(int(first), int(second)) for first, second in state.get("stable_bonds", [])
        }
        for item in state.get("pending", []):
            bond = _bond(*map(int, item["atom_ids"]))
            event_type = item["event_type"]
            detector._pending[(bond, event_type)] = _PendingChange(
                first_seen_frame=int(item["first_seen_frame"]),
                count=int(item["count"]),
            )
        detector._last_frame = state.get("last_frame")
        return detector


__all__ = [
    "BondChangeDetector",
    "BondDetectorConfig",
    "BondEvent",
    "assign_atom_ids",
    "atom_ids",
]
