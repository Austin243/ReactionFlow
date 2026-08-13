"""Generic aggregation of stable topology changes into reaction candidates."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from numbers import Integral

from ase import Atoms

from .detection import Bond, atom_ids


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


__all__ = ["ReactionCandidate", "ReactionTracker"]
