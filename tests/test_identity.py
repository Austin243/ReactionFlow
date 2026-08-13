from __future__ import annotations

import numpy as np
from ase import Atoms

from reactionflow import ReactionCandidate, same_reaction


def candidate(
    symbols: list[str],
    ids: tuple[int, ...],
    reactant_bonds: set[tuple[int, int]],
    product_bonds: set[tuple[int, int]],
    *,
    order: tuple[int, ...] | None = None,
    product_order: tuple[int, ...] | None = None,
    region_ids: tuple[int, ...] | None = None,
) -> ReactionCandidate:
    order = order or tuple(range(len(ids)))
    product_order = product_order or order

    def endpoint(indices: tuple[int, ...]) -> Atoms:
        atoms = Atoms(
            [symbols[index] for index in indices],
            positions=[[index * 2, 0, 0] for index in indices],
        )
        atoms.set_array("atom_id", np.asarray([ids[index] for index in indices]))
        return atoms

    return ReactionCandidate(
        reactant=endpoint(order),
        product=endpoint(product_order),
        atom_ids=region_ids or tuple(sorted(ids)),
        reactant_bonds=frozenset(reactant_bonds),
        product_bonds=frozenset(product_bonds),
        reactant_frame=0,
        product_frame=1,
        observed_frame=2,
        resolved=True,
    )


def test_identity_ignores_atom_numbering_order_geometry_and_direction() -> None:
    forward = candidate(
        ["C", "O", "H", "He"],
        (1, 2, 3, 4),
        {(1, 2), (2, 3)},
        {(1, 2), (1, 3)},
        product_order=(3, 2, 1, 0),
        region_ids=(1, 2, 3),
    )
    renumbered = candidate(
        ["C", "O", "H", "Ne"],
        (30, 10, 20, 99),
        {(10, 30), (10, 20)},
        {(30, 10), (20, 30)},
        order=(2, 3, 0, 1),
        product_order=(1, 0, 3, 2),
        region_ids=(10, 20, 30),
    )
    renumbered.reactant.positions *= 7
    renumbered.product.positions += [13, -4, 2]
    reverse = candidate(
        ["C", "O", "H"],
        (300, 100, 200),
        {(100, 300), (200, 300)},
        {(100, 300), (100, 200)},
        order=(1, 2, 0),
    )

    assert same_reaction(forward, renumbered)
    assert same_reaction(forward, reverse)


def test_identity_requires_exact_elements_and_change_labels() -> None:
    reference = candidate(
        ["C", "O", "H"],
        (1, 2, 3),
        {(1, 2), (2, 3)},
        {(1, 2), (1, 3)},
    )
    changed_element = candidate(
        ["C", "O", "Cl"],
        (1, 2, 3),
        {(1, 2), (2, 3)},
        {(1, 2), (1, 3)},
    )
    changed_connectivity = candidate(
        ["C", "O", "H"],
        (1, 2, 3),
        {(1, 2), (2, 3)},
        {(2, 3), (1, 3)},
    )

    assert not same_reaction(reference, changed_element)
    assert not same_reaction(reference, changed_connectivity)
