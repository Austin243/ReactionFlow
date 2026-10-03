"""Bonds formed and broken between the two minima of a transition, for reporting only."""

from __future__ import annotations

from collections import Counter

from ase import Atoms

from ..detection import BondDetectorConfig, atom_ids, classify_bonds


def bond_changes(
    reactant: Atoms, product: Atoms, config: BondDetectorConfig | None = None
) -> dict[str, list[dict]]:
    """A bond forms when it is inside the formation distance in the product and was outside the
    breaking distance in the reactant; breaking is the reverse. Atoms keep their stable IDs."""

    config = config or BondDetectorConfig()
    before, before_gap = classify_bonds(reactant, config)
    after, after_gap = classify_bonds(product, config)
    symbols = dict(zip(atom_ids(reactant), reactant.get_chemical_symbols(), strict=True))

    def listed(bonds: set) -> list[dict]:
        return [
            {"atom_ids": list(bond), "elements": "-".join(sorted(symbols[i] for i in bond))}
            for bond in sorted(bonds)
        ]

    return {
        "formed": listed(after - before - before_gap),
        "broken": listed(before - after - after_gap),
    }


def reversed_changes(changes: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {"formed": changes["broken"], "broken": changes["formed"]}


def describe(changes: dict[str, list[dict]]) -> str:
    """For example 'formed C-N; broken C-H x2', or 'no bond change'."""

    parts = []
    for kind in ("formed", "broken"):
        counts = Counter(bond["elements"] for bond in changes[kind])
        if counts:
            pairs = ", ".join(
                pair if count == 1 else f"{pair} x{count}" for pair, count in sorted(counts.items())
            )
            parts.append(f"{kind} {pairs}")
    return "; ".join(parts) or "no bond change"
