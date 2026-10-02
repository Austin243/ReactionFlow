"""Generic aggregation of stable topology changes into reaction candidates."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import networkx as nx
from ase import Atoms
from ase.io import read, write

from ._durable import file_digest
from .detection import Bond, atom_ids, canonical_copy, transport_copy


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
class _ReactantOrigin:
    atom_ids: tuple[int, ...]
    atoms: Atoms
    bonds: frozenset[Bond]
    frame: int


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
    origins: tuple[_ReactantOrigin, ...] = ()


def _within(bonds: frozenset[Bond], region: Collection[int]) -> frozenset[Bond]:
    ids = set(region)
    return frozenset(bond for bond in bonds if set(bond) <= ids)


def _select_origin(
    origins: Collection[_ReactantOrigin],
    region: Collection[int],
    accepted_bonds: frozenset[Bond],
    product_bonds: frozenset[Bond],
) -> _ReactantOrigin:
    """Prefer an actual snapshot of this region's accepted basin."""

    ordered = sorted(origins, key=lambda origin: (origin.frame, origin.atom_ids))
    if not ordered:
        raise ValueError("a pending reaction region requires a reactant origin")
    baseline = _within(accepted_bonds, region)
    for origin in ordered:
        if _within(origin.bonds, region) == baseline:
            return origin
    # Overlapping histories can lack a common baseline. Preserve a truthful unresolved
    # change, rather than dropping it because one origin happens to match the product.
    return next(
        (origin for origin in ordered if _within(origin.bonds, region) != product_bonds),
        ordered[0],
    )


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


def _isomorphic(first: nx.Graph, second: nx.Graph) -> bool:
    return nx.is_isomorphic(
        first,
        second,
        node_match=nx.algorithms.isomorphism.categorical_node_match("element", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("change", None),
    )


def _graph_hash(graph: nx.Graph) -> str:
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="element", edge_attr="change")


def same_reaction(first: ReactionCandidate, second: ReactionCandidate) -> bool:
    """Return whether candidates are exact forward/reverse graph equivalents."""

    graph = _reaction_graph(first)
    return _isomorphic(graph, _reaction_graph(second)) or _isomorphic(
        graph, _reaction_graph(second, reverse=True)
    )


def reaction_key(candidate: ReactionCandidate) -> str:
    """Hash shared by forward/reverse-equivalent candidates; confirm matches with same_reaction."""

    return min(
        _graph_hash(_reaction_graph(candidate, reverse=reverse)) for reverse in (False, True)
    )


class ReactionClasses:
    """Names of known reaction topologies, found by key and confirmed by exact graph matching."""

    def __init__(self) -> None:
        self._graphs: dict[str, list[tuple[str, nx.Graph]]] = {}

    def find(self, candidate: ReactionCandidate) -> str | None:
        """Return the name of the class equivalent to this candidate, if one is known."""

        directions = [_reaction_graph(candidate, reverse=reverse) for reverse in (False, True)]
        for name, graph in self._graphs.get(min(map(_graph_hash, directions)), ()):
            if any(_isomorphic(graph, direction) for direction in directions):
                return name
        return None

    def add(self, name: str, candidate: ReactionCandidate) -> None:
        self._graphs.setdefault(reaction_key(candidate), []).append(
            (name, _reaction_graph(candidate))
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
                    # Keep one origin assignment per atom, sharing full snapshots. A merge
                    # must retain each region's baseline so a later split can recover either
                    # reaction without inventing a mixed whole-system geometry.
                    origins = []
                    covered = set()
                    for item in overlaps:
                        for origin in item.origins:
                            support = tuple(sorted(set(origin.atom_ids).intersection(region)))
                            if support:
                                origins.append(
                                    _ReactantOrigin(
                                        support, origin.atoms, origin.bonds, origin.frame
                                    )
                                )
                                covered.update(support)
                    added = tuple(sorted(set(region) - covered))
                    if added:
                        origins.append(
                            _ReactantOrigin(
                                added, self._accepted, self._last_bonds, self._accepted_frame
                            )
                        )
                    reactant = _select_origin(origins, region, self._accepted_bonds, local_product)
                    pending = _PendingTopology(
                        atom_ids=region,
                        reactant=reactant.atoms,
                        reactant_bonds=reactant.bonds,
                        reactant_frame=reactant.frame,
                        bonds=local_product,
                        product=snapshot,
                        product_frame=frame,
                        origins=tuple(origins),
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
        # Match the latest geometry, including provisional changes outside pending regions.
        # A later merge must not describe those atoms using only their older stable topology.
        self._last_bonds = proposal
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
        """Write independent windows and their endpoint snapshots to a new directory.

        A caller that needs crash safety publishes the directory, as ReactionRun does.
        """

        final = Path(path).resolve()
        final.mkdir(parents=True)
        snapshots = {"accepted.traj": self._accepted}
        origin_names = {}
        pending_data = []
        for index, item in enumerate(self._pending):
            product_name = f"product-{index}.traj"
            snapshots[product_name] = item.product
            origins = []
            for origin in item.origins:
                if origin.frame not in origin_names:
                    name = f"reactant-origin-{len(origin_names)}.traj"
                    origin_names[origin.frame] = name
                    snapshots[name] = origin.atoms
                origins.append(
                    {
                        "atom_ids": list(origin.atom_ids),
                        "snapshot": origin_names[origin.frame],
                        "bonds": _checkpoint_bonds(origin.bonds),
                        "frame": origin.frame,
                    }
                )
            pending_data.append(
                {
                    "atom_ids": list(item.atom_ids),
                    "product": product_name,
                    "origins": origins,
                    "bonds": _checkpoint_bonds(item.bonds),
                    "product_frame": item.product_frame,
                    "count": item.count,
                }
            )
        files: dict[str, str] = {}
        for filename, atoms in snapshots.items():
            if atoms is None:
                continue
            destination = final / filename
            write(destination, transport_copy(atoms), format="traj")
            files[filename] = file_digest(destination)
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
        (final / "tracker.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return final

    @classmethod
    def read_checkpoint(
        cls,
        path: str | Path,
        *,
        stability_frames: int | None = None,
    ) -> ReactionTracker:
        """Restore a tracker written by ``write_checkpoint``."""

        root = Path(path).resolve()
        value = json.loads((root / "tracker.json").read_text(encoding="utf-8"))
        if value.get("schema_version") != 2:
            raise ValueError("unsupported reaction-tracker checkpoint")
        stored_stability = int(value["stability_frames"])
        if stability_frames is not None and stored_stability != stability_frames:
            raise ValueError("reaction-tracker checkpoint configuration does not match")
        snapshots: dict[str, Atoms] = {}
        for name, expected_digest in value["files"].items():
            snapshot_path = root / name
            if not snapshot_path.is_file() or file_digest(snapshot_path) != expected_digest:
                raise ValueError(f"reaction-tracker checkpoint failed integrity check: {name}")
            snapshots[name] = canonical_copy(read(snapshot_path))

        tracker = cls(stability_frames=stored_stability)
        symbols = value["symbols"]
        tracker._symbols = (
            None if symbols is None else {int(key): str(symbol) for key, symbol in symbols}
        )
        tracker._accepted_bonds = _restore_bonds(value["accepted_bonds"])
        tracker._accepted = snapshots.get("accepted.traj")
        tracker._accepted_frame = value["accepted_frame"]
        tracker._last_bonds = _restore_bonds(value["last_bonds"])
        tracker._last_frame = value["last_frame"]
        for item in value["pending"]:
            region = tuple(map(int, item["atom_ids"]))
            bonds = _restore_bonds(item["bonds"])
            origins = tuple(
                _ReactantOrigin(
                    tuple(map(int, origin["atom_ids"])),
                    snapshots[origin["snapshot"]],
                    _restore_bonds(origin["bonds"]),
                    int(origin["frame"]),
                )
                for origin in item["origins"]
            )
            reactant = _select_origin(origins, region, tracker._accepted_bonds, bonds)
            tracker._pending.append(
                _PendingTopology(
                    atom_ids=region,
                    reactant=reactant.atoms,
                    reactant_bonds=reactant.bonds,
                    reactant_frame=reactant.frame,
                    bonds=bonds,
                    product=snapshots[item["product"]],
                    product_frame=int(item["product_frame"]),
                    count=int(item["count"]),
                    origins=origins,
                )
            )
        return tracker


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


__all__ = [
    "ReactionCandidate",
    "ReactionClasses",
    "ReactionTracker",
    "reaction_key",
    "same_reaction",
]
