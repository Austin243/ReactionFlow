"""Generic aggregation of stable topology changes into reaction candidates."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Collection
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import Any
from uuid import uuid4

import networkx as nx
from ase import Atoms
from ase.io import read, write

from .detection import Bond, assign_atom_ids, atom_ids


@dataclass(frozen=True, slots=True)
class ReactionCandidate:
    """One connected topology change with independent endpoint snapshots."""

    reactant: Atoms
    product: Atoms
    atom_ids: tuple[int, ...]
    reactant_bonds: frozenset[Bond]
    product_bonds: frozenset[Bond]
    reactant_frame: int
    product_frame: int
    observed_frame: int
    resolved: bool


@dataclass(slots=True)
class _PendingTopology:
    bonds: frozenset[Bond]
    product: Atoms
    product_frame: int
    count: int = 0


def _bonds(values: Collection[Bond], valid_ids: set[int]) -> frozenset[Bond]:
    result: set[Bond] = set()
    for first, second in values:
        if first == second or first not in valid_ids or second not in valid_ids:
            raise ValueError("bonds must connect two stable atom IDs in the frame")
        result.add((first, second) if first < second else (second, first))
    return frozenset(result)


def _changed_regions(
    reactant: frozenset[Bond], product: frozenset[Bond]
) -> tuple[tuple[int, ...], ...]:
    changed_atoms = {atom_id for bond in reactant ^ product for atom_id in bond}
    neighbors: dict[int, set[int]] = {}
    for first, second in reactant | product:
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)

    regions: list[tuple[int, ...]] = []
    unseen = set(changed_atoms)
    while unseen:
        start = min(unseen)
        region = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in neighbors.get(current, ()):
                if neighbor not in region:
                    region.add(neighbor)
                    stack.append(neighbor)
        unseen -= region
        regions.append(tuple(sorted(region)))
    return tuple(sorted(regions))


def _reaction_graph(candidate: ReactionCandidate, *, reverse: bool = False) -> nx.Graph:
    reactant_symbols = dict(
        zip(
            atom_ids(candidate.reactant),
            candidate.reactant.get_chemical_symbols(),
            strict=True,
        )
    )
    product_symbols = dict(
        zip(
            atom_ids(candidate.product),
            candidate.product.get_chemical_symbols(),
            strict=True,
        )
    )
    region = set(candidate.atom_ids)
    if (
        not region <= reactant_symbols.keys()
        or not region <= product_symbols.keys()
        or any(reactant_symbols[atom_id] != product_symbols[atom_id] for atom_id in region)
    ):
        raise ValueError("candidate endpoint identities do not match")

    reactant = set(_bonds(candidate.reactant_bonds, region))
    product = set(_bonds(candidate.product_bonds, region))
    graph = nx.Graph()
    graph.add_nodes_from(
        (atom_id, {"element": reactant_symbols[atom_id]}) for atom_id in candidate.atom_ids
    )
    for first, second in reactant | product:
        if (first, second) in reactant and (first, second) in product:
            change = "unchanged"
        elif (first, second) in product:
            change = "formed"
        else:
            change = "broken"
        if reverse:
            change = {"formed": "broken", "broken": "formed"}.get(change, change)
        graph.add_edge(first, second, change=change)
    return graph


def same_reaction(first: ReactionCandidate, second: ReactionCandidate) -> bool:
    """Return whether candidates are exact forward/reverse graph equivalents."""

    node_match = nx.algorithms.isomorphism.categorical_node_match("element", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("change", None)
    first_graph = _reaction_graph(first)
    return nx.is_isomorphic(
        first_graph,
        _reaction_graph(second),
        node_match=node_match,
        edge_match=edge_match,
    ) or nx.is_isomorphic(
        first_graph,
        _reaction_graph(second, reverse=True),
        node_match=node_match,
        edge_match=edge_match,
    )


class ReactionTracker:
    """Emit candidates after a complete product topology remains stable."""

    def __init__(self, *, stability_frames: int = 3) -> None:
        if (
            isinstance(stability_frames, bool)
            or not isinstance(stability_frames, Integral)
            or stability_frames < 1
        ):
            raise ValueError("stability_frames must be a positive integer")
        self.stability_frames = int(stability_frames)
        self._symbols: dict[int, str] | None = None
        self._accepted_bonds: frozenset[Bond] | None = None
        self._accepted: Atoms | None = None
        self._accepted_frame: int | None = None
        self._reactant: Atoms | None = None
        self._reactant_frame: int | None = None
        self._pending: _PendingTopology | None = None
        self._last_frame: int | None = None

    @property
    def last_frame(self) -> int | None:
        return self._last_frame

    def _clear_transition(self) -> None:
        self._reactant = None
        self._reactant_frame = None
        self._pending = None

    def _freeze_reactant(self) -> None:
        if self._reactant is None:
            assert self._accepted is not None and self._accepted_frame is not None
            self._reactant = self._accepted.copy()
            self._reactant_frame = self._accepted_frame

    def _candidates(self, *, observed_frame: int, resolved: bool) -> tuple[ReactionCandidate, ...]:
        assert self._accepted_bonds is not None
        assert self._reactant is not None and self._reactant_frame is not None
        assert self._pending is not None
        result: list[ReactionCandidate] = []
        for region in _changed_regions(self._accepted_bonds, self._pending.bonds):
            region_ids = set(region)
            result.append(
                ReactionCandidate(
                    reactant=self._reactant.copy(),
                    product=self._pending.product.copy(),
                    atom_ids=region,
                    reactant_bonds=frozenset(
                        bond for bond in self._accepted_bonds if set(bond) <= region_ids
                    ),
                    product_bonds=frozenset(
                        bond for bond in self._pending.bonds if set(bond) <= region_ids
                    ),
                    reactant_frame=self._reactant_frame,
                    product_frame=self._pending.product_frame,
                    observed_frame=observed_frame,
                    resolved=resolved,
                )
            )
        return tuple(result)

    def process(
        self,
        atoms: Atoms,
        *,
        frame: int,
        stable_bonds: Collection[Bond],
        pending_bonds: Collection[Bond] | None,
    ) -> tuple[ReactionCandidate, ...]:
        """Process one ordered stable or provisional topology observation."""

        if self._last_frame is not None and frame <= self._last_frame:
            raise ValueError("frame numbers must increase")
        ids = atom_ids(atoms)
        symbols = dict(zip(ids, atoms.get_chemical_symbols(), strict=True))
        if self._symbols is not None and symbols != self._symbols:
            raise ValueError("atom identities changed between frames")
        bonds = _bonds(stable_bonds, set(ids))
        proposal = None if pending_bonds is None else _bonds(pending_bonds, set(ids))
        snapshot = atoms.copy()
        if self._accepted_bonds is None and proposal is not None:
            raise ValueError("start the tracker before the detector has pending changes")
        self._symbols = symbols
        self._last_frame = frame

        if self._accepted_bonds is None:
            self._accepted_bonds = bonds
            self._accepted = snapshot
            self._accepted_frame = frame
            return ()

        proposed_bonds = bonds if proposal is None else proposal
        if proposed_bonds == self._accepted_bonds:
            self._pending = None
            if proposal is not None:
                self._freeze_reactant()
            else:
                self._accepted = snapshot
                self._accepted_frame = frame
                self._reactant = None
                self._reactant_frame = None
            return ()

        self._freeze_reactant()
        if self._pending is None or proposed_bonds != self._pending.bonds:
            self._pending = _PendingTopology(proposed_bonds, snapshot, frame)
        if proposal is not None:
            return ()
        if bonds == self._pending.bonds:
            self._pending.count += 1
        if self._pending.count < self.stability_frames:
            return ()

        result = self._candidates(observed_frame=frame, resolved=True)
        self._accepted_bonds = bonds
        self._accepted = snapshot
        self._accepted_frame = frame
        self._clear_transition()
        return result

    def finish(self) -> tuple[ReactionCandidate, ...]:
        """Drain an incomplete product topology as unresolved candidates."""

        if self._pending is None or self._last_frame is None:
            self._clear_transition()
            return ()
        result = self._candidates(observed_frame=self._last_frame, resolved=False)
        self._accepted_bonds = self._pending.bonds
        self._accepted = self._pending.product.copy()
        self._accepted_frame = self._pending.product_frame
        self._clear_transition()
        return result

    def write_checkpoint(self, path: str | Path) -> Path:
        """Atomically persist every state needed for exact monitor continuation."""

        final = Path(path).resolve()
        if final.exists():
            raise FileExistsError(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.parent / f".{final.name}-{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            snapshots = {
                "accepted": self._accepted,
                "reactant": self._reactant,
                "pending_product": (None if self._pending is None else self._pending.product),
            }
            files: dict[str, str] = {}
            for label, atoms in snapshots.items():
                if atoms is None:
                    continue
                snapshot = atoms.copy()
                snapshot.calc = None
                snapshot.info["atom_ids"] = list(atom_ids(snapshot))
                filename = f"{label}.traj"
                destination = temporary / filename
                write(destination, snapshot, format="traj")
                files[filename] = _file_digest(destination)

            value: dict[str, Any] = {
                "schema_version": 1,
                "stability_frames": self.stability_frames,
                "symbols": (
                    None
                    if self._symbols is None
                    else [[atom_id, symbol] for atom_id, symbol in self._symbols.items()]
                ),
                "accepted_bonds": _checkpoint_bonds(self._accepted_bonds),
                "accepted_frame": self._accepted_frame,
                "reactant_frame": self._reactant_frame,
                "pending": (
                    None
                    if self._pending is None
                    else {
                        "bonds": _checkpoint_bonds(self._pending.bonds),
                        "product_frame": self._pending.product_frame,
                        "count": self._pending.count,
                    }
                ),
                "last_frame": self._last_frame,
                "files": files,
            }
            (temporary / "tracker.json").write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return final

    @classmethod
    def read_checkpoint(
        cls,
        path: str | Path,
        *,
        stability_frames: int | None = None,
    ) -> ReactionTracker:
        """Restore and integrity-check a version-1 tracker checkpoint."""

        root = Path(path).resolve()
        value = json.loads((root / "tracker.json").read_text(encoding="utf-8"))
        if value.get("schema_version") != 1:
            raise ValueError("unsupported reaction-tracker checkpoint")
        stored_stability = int(value["stability_frames"])
        if stability_frames is not None and stored_stability != stability_frames:
            raise ValueError("reaction-tracker checkpoint configuration does not match")
        files = value.get("files")
        if not isinstance(files, dict):
            raise ValueError("reaction-tracker checkpoint has an invalid file manifest")
        expected_files = {
            name
            for name, present in (
                ("accepted.traj", value.get("accepted_bonds") is not None),
                ("reactant.traj", value.get("reactant_frame") is not None),
                ("pending_product.traj", value.get("pending") is not None),
            )
            if present
        }
        if set(files) != expected_files:
            raise ValueError("reaction-tracker checkpoint has an invalid snapshot set")
        for name, expected_digest in files.items():
            snapshot_path = root / name
            if not snapshot_path.is_file() or _file_digest(snapshot_path) != expected_digest:
                raise ValueError(f"reaction-tracker checkpoint failed integrity check: {name}")

        tracker = cls(stability_frames=stored_stability)
        symbols = value.get("symbols")
        tracker._symbols = (
            None if symbols is None else {int(atom_id): str(symbol) for atom_id, symbol in symbols}
        )
        accepted_bonds = value.get("accepted_bonds")
        tracker._accepted_bonds = _restore_bonds(accepted_bonds)
        tracker._accepted = (
            None if accepted_bonds is None else _read_tracker_atoms(root / "accepted.traj")
        )
        tracker._accepted_frame = _optional_frame(value.get("accepted_frame"))
        tracker._reactant_frame = _optional_frame(value.get("reactant_frame"))
        tracker._reactant = (
            None if tracker._reactant_frame is None else _read_tracker_atoms(root / "reactant.traj")
        )
        pending = value.get("pending")
        if pending is not None:
            pending_bonds = _restore_bonds(pending["bonds"])
            assert pending_bonds is not None
            tracker._pending = _PendingTopology(
                bonds=pending_bonds,
                product=_read_tracker_atoms(root / "pending_product.traj"),
                product_frame=int(pending["product_frame"]),
                count=int(pending["count"]),
            )
        tracker._last_frame = _optional_frame(value.get("last_frame"))
        tracker._validate_checkpoint()
        return tracker

    def _validate_checkpoint(self) -> None:
        if (self._accepted_bonds is None) != (self._accepted is None):
            raise ValueError("reaction-tracker checkpoint has an incomplete accepted state")
        if (self._accepted is None) != (self._accepted_frame is None):
            raise ValueError("reaction-tracker checkpoint has an invalid accepted frame")
        if (self._reactant is None) != (self._reactant_frame is None):
            raise ValueError("reaction-tracker checkpoint has an invalid reactant state")
        if self._pending is not None and self._reactant is None:
            raise ValueError("reaction-tracker checkpoint has an incomplete transition")
        for atoms in (self._accepted, self._reactant):
            if atoms is not None:
                atom_ids(atoms)
                atoms.calc = None
        if self._pending is not None:
            atom_ids(self._pending.product)
            self._pending.product.calc = None
            if self._pending.count < 0:
                raise ValueError("reaction-tracker checkpoint has a negative stability count")


def _file_digest(path: Path) -> str:
    checksum = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _read_tracker_atoms(path: Path) -> Atoms:
    atoms = read(path)
    atom_ids(atoms)
    assign_atom_ids(atoms)
    atoms.info.pop("atom_ids", None)
    atoms.calc = None
    return atoms


def _checkpoint_bonds(values: Collection[Bond] | None) -> list[list[int]] | None:
    if values is None:
        return None
    return [list(bond) for bond in sorted(values)]


def _restore_bonds(
    values: Collection[Collection[int]] | None,
) -> frozenset[Bond] | None:
    if values is None:
        return None
    return frozenset(
        (first, second) if first < second else (second, first)
        for first, second in (tuple(map(int, bond)) for bond in values)
    )


def _optional_frame(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("reaction-tracker checkpoint frame must be a non-negative integer")
    return value


__all__ = ["ReactionCandidate", "ReactionTracker", "same_reaction"]
