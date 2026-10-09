"""The atom pairs a guided EON search pushes, and how; no EON needed."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from reactionflow.detection import assign_atom_ids
from reactionflow.search.eon import EONSettings, guided_push

# C bonded to both H, and N alone: 3.0 Å from C, 1.9 Å from the first H, 4.1 Å from the second.
ATOMS = assign_atom_ids(
    Atoms("CHHN", positions=[[0, 0, 0], [1.09, 0, 0], [-1.09, 0, 0], [3.0, 0, 0]])
)


def push(guide, fixed=(), seed=0):
    mask = np.isin(np.arange(len(ATOMS)), fixed)
    return guided_push(ATOMS, mask, EONSettings(guide=guide), np.random.default_rng(seed))


def picks(guide, fixed=()):
    return {
        (record["push"], *record["atom_ids"])
        for _, _, record in (push(guide, fixed, seed) for seed in range(40))
    }


def test_a_guide_lists_element_pairs_to_form_or_break():
    guide = EONSettings(guide={"form": ["N-C", ("C", "N")], "break": ["H-C"]}).guide
    assert guide == {"form": ["C-N"], "break": ["C-H"], "within_A": 3.5}
    for bad, error in (
        ({"form": "C-N"}, "list of element pairs"),
        ({"form": ["C-Q"]}, "invalid element pair"),
        ({"form": [], "break": []}, "needs an element pair"),
        ({"form": ["C-N"], "near": 3.0}, "takes 'form', 'break'"),
        ({"form": ["C-N"], "within_A": 0}, "within_A must be positive"),
    ):
        with pytest.raises(ValueError, match=error):
            EONSettings(guide=bad)
    with pytest.raises(ValueError, match="leave displace unset"):
        EONSettings(guide={"form": ["C-N"]}, displace="local")


def test_a_guided_push_moves_one_pair_of_the_guide_together_or_apart():
    # Not the two H, which share the C, nor the second H and N, beyond within_A.
    guide = {"form": ["C-N", "H-H", "H-N"], "break": ["C-H"]}
    assert picks(guide) == {("form", 0, 3), ("form", 1, 3), ("break", 0, 1), ("break", 0, 2)}
    assert picks({"form": ["C-N", "H-N"], "within_A": 2.0}) == {("form", 1, 3)}
    # Never a pair of fixed atoms.
    assert picks({"break": ["C-H"]}, fixed=[0, 1]) == {("break", 0, 2)}
    assert push({"form": ["N-N"]}) is None

    positions, mode, record = push({"form": ["C-N"]})
    assert record == {"push": "form", "atom_ids": [0, 3], "elements": "C-N"}
    # The pair closes by displace_size_A, each atom in proportion to the other's mass, and every
    # atom also moves a little at random.
    distance = np.linalg.norm(positions[3] - positions[0])
    assert distance == pytest.approx(3.0 - EONSettings.displace_size_A, abs=0.1)
    assert np.linalg.norm(mode) == pytest.approx(1)
    assert mode[0, 0] > 0 > mode[3, 0] and not mode[[1, 2]].any()
    np.testing.assert_allclose(ATOMS.get_masses() @ mode, 0, atol=1e-12)

    # A fixed atom stays put, and the other atom of its pair moves the whole push.
    positions, mode, _ = push({"break": ["C-H"]}, fixed=[0])
    moved = 1 if mode[1].any() else 2
    np.testing.assert_array_equal(positions[0], ATOMS.positions[0])
    assert not mode[0].any()
    distance = np.linalg.norm(positions[moved] - positions[0])
    assert distance == pytest.approx(1.09 + EONSettings.displace_size_A, abs=0.1)
