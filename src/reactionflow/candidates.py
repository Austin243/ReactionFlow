"""Generic aggregation of stable topology changes into reaction candidates."""

from __future__ import annotations

import json
import shutil
from collections.abc import Collection
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from uuid import uuid4

import networkx as nx
from ase import Atoms
from ase.io import read, write

from ._durable import ensure_directory, publish
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
    atom_ids: tuple[int, ...]
    reactant: Atoms
    reactant_bonds: frozenset[Bond]
    reactant_frame: int
    bonds: frozenset[Bond]
    product: Atoms
    product_frame: int
    count: int = 0


def _within(bonds: frozenset[Bond], region: Collection[int]) -> frozenset[Bond]:
    ids = set(region)
    return frozenset(bond for bond in bonds if set(bond) <= ids)


def _bonds(values: Collection[Bond], valid_ids: set[int]) -> frozenset[Bond]:
    result: set[Bond] = set()
    for first, second in values:
        if first == second or first not in valid_ids or second not in valid_ids:
            raise ValueError("bonds must connect two stable atom IDs in the frame")
        result.add((first, second) if first < second else (second, first))
    return frozenset(result)


def _changed_regions(
    reactant: frozenset[Bond],
    product: frozenset[Bond],
    stable: frozenset[Bond] | None = None,
) -> tuple[tuple[int, ...], ...]:
    stable = product if stable is None else stable
    changed_atoms = {
        atom_id for bond in (reactant ^ product) | (stable ^ product) for atom_id in bond
    }
    neighbors: dict[int, set[int]] = {}
    for first, second in reactant | product | stable:
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


def reaction_key(candidate: ReactionCandidate) -> str:
    """Hash shared by forward/reverse-equivalent candidates; confirm matches with same_reaction."""

    return min(
        nx.weisfeiler_lehman_graph_hash(
            _reaction_graph(candidate, reverse=reverse),
            node_attr="element",
            edge_attr="change",
        )
        for reverse in (False, True)
    )


class ReactionTracker:
    """Confirm connected topology changes independently of activity elsewhere."""

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
        # The latest frame seeds newly changing regions; pending regions keep their own endpoints.
        self._accepted: Atoms | None = None
        self._accepted_frame: int | None = None
        self._last_bonds: frozenset[Bond] = frozenset()
        self._pending: list[_PendingTopology] = []
        self._last_frame: int | None = None

    @property
    def last_frame(self) -> int | None:
        return self._last_frame

    def _candidate(
        self, pending: _PendingTopology, *, observed_frame: int, resolved: bool
    ) -> ReactionCandidate | None:
        reactant_bonds = _within(pending.reactant_bonds, pending.atom_ids)
        if reactant_bonds == pending.bonds:
            return None
        return ReactionCandidate(
            reactant=pending.reactant.copy(),
            product=pending.product.copy(),
            atom_ids=pending.atom_ids,
            reactant_bonds=reactant_bonds,
            product_bonds=pending.bonds,
            reactant_frame=pending.reactant_frame,
            product_frame=pending.product_frame,
            observed_frame=observed_frame,
            resolved=resolved,
        )

    def _accept(self, pending: _PendingTopology) -> None:
        assert self._accepted_bonds is not None
        self._accepted_bonds = (
            self._accepted_bonds - _within(self._accepted_bonds, pending.atom_ids)
        ) | pending.bonds

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
        proposal = bonds if pending_bonds is None else _bonds(pending_bonds, set(ids))
        if self._accepted_bonds is None and pending_bonds is not None:
            raise ValueError("start the tracker before the detector has pending changes")
        snapshot = atoms.copy()
        self._symbols = symbols
        result: list[ReactionCandidate] = []
        remaining: list[_PendingTopology] = []
        if self._accepted_bonds is None:
            self._accepted_bonds = bonds
        else:
            assert self._accepted is not None and self._accepted_frame is not None
            for region in _changed_regions(self._accepted_bonds, proposal, bonds):
                overlaps = [
                    item for item in self._pending if set(item.atom_ids).intersection(region)
                ]
                local_product = _within(proposal, region)
                if (
                    len(overlaps) == 1
                    and overlaps[0].atom_ids == region
                    and overlaps[0].bonds == local_product
                ):
                    pending = overlaps[0]
                else:
                    # A merge/split or another local change starts a new stability window. The
                    # oldest whole-frame reactant remains the pre-crossing endpoint.
                    oldest = (
                        min(overlaps, key=lambda item: item.reactant_frame) if overlaps else None
                    )
                    pending = _PendingTopology(
                        atom_ids=region,
                        reactant=self._accepted if oldest is None else oldest.reactant,
                        reactant_bonds=self._last_bonds
                        if oldest is None
                        else oldest.reactant_bonds,
                        reactant_frame=self._accepted_frame
                        if oldest is None
                        else oldest.reactant_frame,
                        bonds=local_product,
                        product=snapshot,
                        product_frame=frame,
                    )
                if _within(bonds ^ proposal, region):
                    pending.count = 0
                else:
                    pending.count += 1
                if pending.count < self.stability_frames:
                    remaining.append(pending)
                    continue
                # If a region joins an independently accepted event, the oldest snapshot can
                # precede that event. Keep this overlapping history as unresolved, rather than
                # claiming a reactant topology that does not match the retained geometry.
                resolved = _within(pending.reactant_bonds, region) == _within(
                    self._accepted_bonds, region
                )
                candidate = self._candidate(pending, observed_frame=frame, resolved=resolved)
                if candidate is not None:
                    result.append(candidate)
                self._accept(pending)
        self._pending = remaining
        self._accepted = snapshot
        self._accepted_frame = self._last_frame = frame
        self._last_bonds = bonds
        return tuple(result)

    def finish(self) -> tuple[ReactionCandidate, ...]:
        """Drain incomplete regions once, without labeling them resolved."""

        if self._last_frame is None:
            return ()
        result: list[ReactionCandidate] = []
        for pending in self._pending:
            candidate = self._candidate(pending, observed_frame=self._last_frame, resolved=False)
            if candidate is not None:
                result.append(candidate)
            self._accept(pending)
        self._pending = []
        return tuple(result)

    def write_checkpoint(self, path: str | Path) -> Path:
        """Atomically persist independent windows and their endpoint snapshots."""

        final = Path(path).resolve()
        if final.exists():
            raise FileExistsError(final)
        ensure_directory(final.parent)
        temporary = final.parent / f".{final.name}-{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            snapshots = {"accepted.traj": self._accepted}
            pending_data = []
            for index, item in enumerate(self._pending):
                reactant_name = f"reactant-{index}.traj"
                product_name = f"product-{index}.traj"
                snapshots[reactant_name] = item.reactant
                snapshots[product_name] = item.product
                pending_data.append(
                    {
                        "atom_ids": list(item.atom_ids),
                        "reactant": reactant_name,
                        "product": product_name,
                        "reactant_bonds": _checkpoint_bonds(item.reactant_bonds),
                        "reactant_frame": item.reactant_frame,
                        "bonds": _checkpoint_bonds(item.bonds),
                        "product_frame": item.product_frame,
                        "count": item.count,
                    }
                )
            files: dict[str, str] = {}
            for filename, atoms in snapshots.items():
                if atoms is None:
                    continue
                snapshot = atoms.copy()
                snapshot.calc = None
                snapshot.info["atom_ids"] = list(atom_ids(snapshot))
                destination = temporary / filename
                write(destination, snapshot, format="traj")
                files[filename] = _file_digest(destination)
            value = {
                "schema_version": 2,
                "stability_frames": self.stability_frames,
                "symbols": None if self._symbols is None else list(self._symbols.items()),
                "accepted_bonds": _checkpoint_bonds(self._accepted_bonds),
                "accepted_frame": self._accepted_frame,
                "last_bonds": _checkpoint_bonds(self._last_bonds),
                "pending": pending_data,
                "last_frame": self._last_frame,
                "files": files,
            }
            (temporary / "tracker.json").write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            publish(temporary, final)
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
        """Restore a tracker, including legacy whole-system stability windows."""

        root = Path(path).resolve()
        value = json.loads((root / "tracker.json").read_text(encoding="utf-8"))
        version = value.get("schema_version")
        if version not in (1, 2):
            raise ValueError("unsupported reaction-tracker checkpoint")
        stored_stability = int(value["stability_frames"])
        if stability_frames is not None and stored_stability != stability_frames:
            raise ValueError("reaction-tracker checkpoint configuration does not match")
        pending_data = value.get("pending")
        expected_files = {"accepted.traj"} if value.get("accepted_bonds") is not None else set()
        if version == 1:
            if value.get("reactant_frame") is not None:
                expected_files.add("reactant.traj")
            if pending_data is not None:
                expected_files.add("pending_product.traj")
        else:
            for index, item in enumerate(pending_data):
                if (
                    item["reactant"] != f"reactant-{index}.traj"
                    or item["product"] != f"product-{index}.traj"
                ):
                    raise ValueError("reaction-tracker checkpoint has invalid snapshot names")
                expected_files.update((item["reactant"], item["product"]))
        files = value.get("files")
        if not isinstance(files, dict) or set(files) != expected_files:
            raise ValueError("reaction-tracker checkpoint has an invalid snapshot set")
        for name, expected_digest in files.items():
            snapshot_path = root / name
            if not snapshot_path.is_file() or _file_digest(snapshot_path) != expected_digest:
                raise ValueError(f"reaction-tracker checkpoint failed integrity check: {name}")

        tracker = cls(stability_frames=stored_stability)
        symbols = value.get("symbols")
        tracker._symbols = (
            None if symbols is None else {int(key): str(symbol) for key, symbol in symbols}
        )
        tracker._accepted_bonds = _restore_bonds(value.get("accepted_bonds"))
        tracker._accepted = (
            None if tracker._accepted_bonds is None else _read_tracker_atoms(root / "accepted.traj")
        )
        tracker._accepted_frame = _optional_frame(value.get("accepted_frame"))
        tracker._last_frame = _optional_frame(value.get("last_frame"))
        if version == 1:
            tracker._last_bonds = tracker._accepted_bonds or frozenset()
            if pending_data is not None:
                product_bonds = _restore_bonds(pending_data["bonds"])
                assert tracker._accepted_bonds is not None and product_bonds is not None
                for region in _changed_regions(tracker._accepted_bonds, product_bonds):
                    tracker._pending.append(
                        _PendingTopology(
                            atom_ids=region,
                            reactant=_read_tracker_atoms(root / "reactant.traj"),
                            reactant_bonds=tracker._accepted_bonds,
                            reactant_frame=int(value["reactant_frame"]),
                            bonds=_within(product_bonds, region),
                            product=_read_tracker_atoms(root / "pending_product.traj"),
                            product_frame=int(pending_data["product_frame"]),
                            count=int(pending_data["count"]),
                        )
                    )
        else:
            tracker._last_bonds = _restore_bonds(value["last_bonds"])
            for item in pending_data:
                tracker._pending.append(
                    _PendingTopology(
                        atom_ids=tuple(map(int, item["atom_ids"])),
                        reactant=_read_tracker_atoms(root / item["reactant"]),
                        reactant_bonds=_restore_bonds(item["reactant_bonds"]),
                        reactant_frame=int(item["reactant_frame"]),
                        bonds=_restore_bonds(item["bonds"]),
                        product=_read_tracker_atoms(root / item["product"]),
                        product_frame=int(item["product_frame"]),
                        count=int(item["count"]),
                    )
                )
        tracker._validate_checkpoint()
        return tracker

    def _validate_checkpoint(self) -> None:
        if (self._accepted_bonds is None) != (self._accepted is None):
            raise ValueError("reaction-tracker checkpoint has an incomplete accepted state")
        if (self._accepted is None) != (self._accepted_frame is None):
            raise ValueError("reaction-tracker checkpoint has an invalid accepted frame")
        seen: set[int] = set()
        for pending in self._pending:
            region = set(pending.atom_ids)
            if seen.intersection(region) or not region or len(region) != len(pending.atom_ids):
                raise ValueError("reaction-tracker checkpoint has invalid pending regions")
            seen.update(region)
            if pending.count < 0:
                raise ValueError("reaction-tracker checkpoint has a negative stability count")
            for atoms in (pending.reactant, pending.product):
                symbols = dict(zip(atom_ids(atoms), atoms.get_chemical_symbols(), strict=True))
                if symbols != self._symbols or not region <= symbols.keys():
                    raise ValueError("reaction-tracker checkpoint has inconsistent atom identities")
            if _bonds(pending.bonds, region) != pending.bonds:
                raise ValueError("reaction-tracker checkpoint has invalid product bonds")


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


__all__ = ["ReactionCandidate", "ReactionTracker", "reaction_key", "same_reaction"]
