from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from reactionflow import ReactionTracker, atom_ids


def tracked_frame(symbols: str | list[str], frame: int, ids: tuple[int, ...]) -> Atoms:
    atoms = Atoms(symbols, positions=[[index, 0, 0] for index in range(len(ids))])
    atoms.set_array("atom_id", np.asarray(ids))
    atoms.info["frame_marker"] = frame
    return atoms


def test_tracker_waits_for_stable_product_and_selects_endpoints() -> None:
    ids = (10, 20, 30)
    accepted = {(10, 20)}
    tracker = ReactionTracker(stability_frames=2)
    baseline = tracked_frame("H3", 0, ids)
    assert tracker.process(baseline, frame=0, stable_bonds=accepted, pending_bonds=None) == ()

    crossing = tracked_frame("H3", 1, ids)
    assert tracker.process(crossing, frame=1, stable_bonds=accepted, pending_bonds=set()) == ()
    provisional_product = tracked_frame("H3", 2, ids)
    assert (
        tracker.process(
            provisional_product,
            frame=2,
            stable_bonds=set(),
            pending_bonds={(20, 30)},
        )
        == ()
    )

    first_product = tracked_frame("H3", 3, ids)
    product = {(20, 30)}
    assert tracker.process(first_product, frame=3, stable_bonds=product, pending_bonds=None) == ()
    (candidate,) = tracker.process(
        tracked_frame("H3", 4, ids), frame=4, stable_bonds=product, pending_bonds=None
    )

    assert (
        candidate.reactant_frame,
        candidate.product_frame,
        candidate.observed_frame,
    ) == (0, 2, 4)
    assert candidate.reactant_bonds == frozenset(accepted)
    assert candidate.product_bonds == frozenset(product)
    assert candidate.resolved is True
    baseline.info["frame_marker"] = provisional_product.info["frame_marker"] = -1
    assert candidate.reactant.info["frame_marker"] == 0
    assert candidate.product.info["frame_marker"] == 2


def test_tracker_returns_every_disconnected_region_after_reordering() -> None:
    ids = (10, 20, 30, 40)
    baseline = tracked_frame("H4", 0, ids)
    tracker = ReactionTracker(stability_frames=1)
    bonds = {(10, 20), (30, 40)}
    tracker.process(baseline, frame=0, stable_bonds=bonds, pending_bonds=None)

    reordered = tracked_frame("H4", 1, ids)[[2, 3, 0, 1]]
    candidates = tracker.process(reordered, frame=1, stable_bonds=set(), pending_bonds=None)

    assert [candidate.atom_ids for candidate in candidates] == [(10, 20), (30, 40)]
    assert [candidate.reactant_bonds for candidate in candidates] == [
        frozenset({(10, 20)}),
        frozenset({(30, 40)}),
    ]
    assert atom_ids(candidates[0].product) == (30, 40, 10, 20)


def test_candidate_boundaries_do_not_depend_on_elements() -> None:
    ids = (10, 20, 30)

    def outcome(symbols: str | list[str]) -> tuple[object, ...]:
        tracker = ReactionTracker(stability_frames=2)
        tracker.process(
            tracked_frame(symbols, 0, ids),
            frame=0,
            stable_bonds={(10, 20)},
            pending_bonds=None,
        )
        tracker.process(
            tracked_frame(symbols, 1, ids),
            frame=1,
            stable_bonds={(20, 30)},
            pending_bonds=None,
        )
        (candidate,) = tracker.process(
            tracked_frame(symbols, 2, ids),
            frame=2,
            stable_bonds={(20, 30)},
            pending_bonds=None,
        )
        return (
            candidate.atom_ids,
            candidate.reactant_bonds,
            candidate.product_bonds,
            candidate.reactant_frame,
            candidate.product_frame,
            candidate.observed_frame,
            candidate.resolved,
        )

    assert outcome("H3") == outcome(["C", "O", "Si"])


def test_finish_drains_pending_regions_as_unresolved() -> None:
    ids = (10, 20, 30, 40)
    tracker = ReactionTracker(stability_frames=3)
    tracker.process(
        tracked_frame(["C", "O", "H", "Cl"], 0, ids),
        frame=0,
        stable_bonds={(10, 20), (30, 40)},
        pending_bonds=None,
    )
    tracker.process(
        tracked_frame(["C", "O", "H", "Cl"], 1, ids),
        frame=1,
        stable_bonds={(10, 20), (30, 40)},
        pending_bonds=set(),
    )

    candidates = tracker.finish()
    assert [candidate.atom_ids for candidate in candidates] == [(10, 20), (30, 40)]
    assert all(not candidate.resolved for candidate in candidates)
    assert all(
        (candidate.reactant_frame, candidate.product_frame, candidate.observed_frame) == (0, 1, 1)
        for candidate in candidates
    )
    assert tracker.finish() == ()


def test_tracker_checkpoint_preserves_an_in_progress_topology_change(tmp_path) -> None:
    ids = (10, 20)
    tracker = ReactionTracker(stability_frames=2)
    tracker.process(
        tracked_frame("H2", 0, ids),
        frame=0,
        stable_bonds={(10, 20)},
        pending_bonds=None,
    )
    tracker.process(
        tracked_frame("H2", 1, ids),
        frame=1,
        stable_bonds={(10, 20)},
        pending_bonds=set(),
    )

    checkpoint = tracker.write_checkpoint(tmp_path / "tracker")
    restored = ReactionTracker.read_checkpoint(checkpoint, stability_frames=2)
    assert restored.last_frame == 1
    assert (
        restored.process(
            tracked_frame("H2", 2, ids),
            frame=2,
            stable_bonds=set(),
            pending_bonds=None,
        )
        == ()
    )
    (candidate,) = restored.process(
        tracked_frame("H2", 3, ids),
        frame=3,
        stable_bonds=set(),
        pending_bonds=None,
    )
    assert (candidate.reactant_frame, candidate.product_frame) == (0, 1)
    assert candidate.reactant_bonds == frozenset({(10, 20)})
    assert candidate.product_bonds == frozenset()

    with pytest.raises(FileExistsError):
        tracker.write_checkpoint(checkpoint)
    accepted = checkpoint / "accepted.traj"
    accepted.write_bytes(accepted.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="integrity check"):
        ReactionTracker.read_checkpoint(checkpoint)
