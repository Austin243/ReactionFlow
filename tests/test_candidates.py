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


def test_independent_noise_does_not_delay_a_stable_local_change() -> None:
    ids = (0, 1, 2, 3)
    tracker = ReactionTracker(stability_frames=2)
    tracker.process(tracked_frame("H4", 0, ids), frame=0, stable_bonds=[], pending_bonds=None)
    emitted = []
    for frame in range(1, 21):
        bonds = {(0, 1)} | ({(2, 3)} if frame % 2 == 0 else set())
        emitted.extend(
            tracker.process(
                tracked_frame("H4", frame, ids), frame=frame, stable_bonds=bonds, pending_bonds=None
            )
        )
    assert [(item.atom_ids, item.observed_frame, item.resolved) for item in emitted] == [
        ((0, 1), 2, True)
    ]
    (terminal,) = tracker.finish()
    assert terminal.atom_ids == (2, 3) and not terminal.resolved
    assert tracker.finish() == ()


def test_detector_pending_regions_survive_independent_confirmation_and_checkpoint(tmp_path) -> None:
    from reactionflow import BondChangeDetector, BondDetectorConfig, assign_atom_ids

    detector = BondChangeDetector(
        BondDetectorConfig(persistence_frames=2, pair_thresholds={"H-H": (0.8, 1.2)})
    )
    tracker = ReactionTracker(stability_frames=2)

    def observe(frame, distance_a, distance_b, monitor, reactions):
        atoms = assign_atom_ids(
            Atoms(
                "H4", positions=[[0, 0, 0], [distance_a, 0, 0], [10, 0, 0], [10 + distance_b, 0, 0]]
            )
        )
        monitor.process(atoms, frame=frame)
        return reactions.process(
            atoms,
            frame=frame,
            stable_bonds=monitor.stable_bonds,
            pending_bonds=monitor.pending_bonds,
        )

    assert observe(0, 0.6, 0.6, detector, tracker) == ()
    assert observe(1, 1.5, 0.6, detector, tracker) == ()
    assert observe(2, 1.5, 0.6, detector, tracker) == ()
    (first,) = observe(3, 1.5, 1.5, detector, tracker)
    assert first.atom_ids == (0, 1) and first.resolved
    assert (first.reactant_frame, first.product_frame, first.observed_frame) == (0, 1, 3)
    assert detector.pending_bonds is not None

    restored = ReactionTracker.read_checkpoint(tracker.write_checkpoint(tmp_path / "tracker"))
    resumed_detector = BondChangeDetector.from_state(detector.export_state())
    for monitor, reactions in ((detector, tracker), (resumed_detector, restored)):
        assert observe(4, 1.5, 1.5, monitor, reactions) == ()
        (second,) = observe(5, 1.5, 1.5, monitor, reactions)
        assert second.atom_ids == (2, 3) and second.resolved
        assert (second.reactant_frame, second.product_frame, second.observed_frame) == (2, 3, 5)
        assert second.reactant.positions[3, 0] == pytest.approx(10.6)
        assert second.product.positions[3, 0] == pytest.approx(11.5)
        assert observe(6, 1.5, 1.5, monitor, reactions) == ()
        assert reactions.finish() == ()


def test_interacting_regions_merge_and_restart_stability_with_the_oldest_reactant() -> None:
    ids = (0, 1, 2, 3)
    tracker = ReactionTracker(stability_frames=3)

    def observe(frame, bonds):
        return tracker.process(
            tracked_frame("H4", frame, ids), frame=frame, stable_bonds=bonds, pending_bonds=None
        )

    assert observe(0, []) == ()
    assert observe(1, [(0, 1)]) == ()
    assert observe(2, [(0, 1), (2, 3)]) == ()
    merged = [(0, 1), (1, 2), (2, 3)]
    assert observe(3, merged) == ()
    assert observe(4, merged) == ()
    (candidate,) = observe(5, merged)
    assert candidate.atom_ids == ids and candidate.resolved
    assert (candidate.reactant_frame, candidate.product_frame, candidate.observed_frame) == (
        0,
        3,
        5,
    )
    assert candidate.reactant.info["frame_marker"] == 0
    assert candidate.reactant_bonds == frozenset()
    assert candidate.product_bonds == frozenset(merged)
    assert observe(6, merged) == () and tracker.finish() == ()


def test_transient_bridge_splits_back_into_independent_windows() -> None:
    ids = (0, 1, 2, 3)
    tracker = ReactionTracker(stability_frames=2)

    def observe(frame, bonds):
        return tracker.process(
            tracked_frame("H4", frame, ids), frame=frame, stable_bonds=bonds, pending_bonds=None
        )

    observe(0, [])
    assert observe(1, [(0, 1), (1, 2), (2, 3)]) == ()
    assert observe(2, [(0, 1), (2, 3)]) == ()
    candidates = observe(3, [(0, 1), (2, 3)])
    assert [item.atom_ids for item in candidates] == [(0, 1), (2, 3)]
    assert all(item.resolved and item.reactant_frame == 0 for item in candidates)
    assert observe(4, [(0, 1), (2, 3)]) == ()
    assert tracker.finish() == ()


def test_merge_after_independent_acceptance_is_conservatively_unresolved() -> None:
    ids = (0, 1, 2, 3)
    tracker = ReactionTracker(stability_frames=2)
    tracker.process(tracked_frame("H4", 0, ids), frame=0, stable_bonds=[], pending_bonds=None)
    assert (
        tracker.process(
            tracked_frame("H4", 1, ids),
            frame=1,
            stable_bonds=[(2, 3)],
            pending_bonds=[(0, 1), (2, 3)],
        )
        == ()
    )
    (independent,) = tracker.process(
        tracked_frame("H4", 2, ids),
        frame=2,
        stable_bonds=[(2, 3)],
        pending_bonds=[(0, 1), (2, 3)],
    )
    assert independent.atom_ids == (2, 3) and independent.resolved
    merged = [(0, 1), (1, 2), (2, 3)]
    assert (
        tracker.process(
            tracked_frame("H4", 3, ids),
            frame=3,
            stable_bonds=merged,
            pending_bonds=None,
        )
        == ()
    )
    (overlapping,) = tracker.process(
        tracked_frame("H4", 4, ids),
        frame=4,
        stable_bonds=merged,
        pending_bonds=None,
    )
    assert overlapping.atom_ids == ids and not overlapping.resolved
    assert overlapping.reactant_frame == 0 and overlapping.reactant_bonds == frozenset()
    assert overlapping.product_bonds == frozenset(merged)
    assert (
        tracker.process(
            tracked_frame("H4", 5, ids),
            frame=5,
            stable_bonds=merged,
            pending_bonds=None,
        )
        == ()
    )


def test_pending_reversion_does_not_replace_the_pre_crossing_reactant() -> None:
    ids = (0, 1)
    tracker = ReactionTracker(stability_frames=2)
    tracker.process(tracked_frame("H2", 0, ids), frame=0, stable_bonds=[(0, 1)], pending_bonds=None)
    assert (
        tracker.process(tracked_frame("H2", 1, ids), frame=1, stable_bonds=[], pending_bonds=None)
        == ()
    )
    assert (
        tracker.process(
            tracked_frame("H2", 2, ids), frame=2, stable_bonds=[], pending_bonds=[(0, 1)]
        )
        == ()
    )
    assert (
        tracker.process(tracked_frame("H2", 3, ids), frame=3, stable_bonds=[], pending_bonds=None)
        == ()
    )
    (candidate,) = tracker.process(
        tracked_frame("H2", 4, ids), frame=4, stable_bonds=[], pending_bonds=None
    )
    assert candidate.reactant_frame == 0 and candidate.resolved
    assert candidate.reactant_bonds == frozenset({(0, 1)})
    assert candidate.product_bonds == frozenset()


@pytest.mark.parametrize("checkpoint_frame", [None, 2, 3])
@pytest.mark.parametrize("ending", ["merged", "first", "second"])
def test_merge_retains_independent_origins_after_provisional_cancellation(
    tmp_path, checkpoint_frame, ending
):
    from reactionflow import BondChangeDetector, BondDetectorConfig, assign_atom_ids

    detector = BondChangeDetector(
        BondDetectorConfig(persistence_frames=2, pair_thresholds={"H-H": (0.8, 1.2)})
    )
    tracker = ReactionTracker(stability_frames=2)
    final_positions = {
        "merged": [0, 0.6, 1.2, 1.8],
        "first": [0, 0.6, 5, 7],
        "second": [0, 2, 5, 5.6],
    }
    positions = [
        [0, 2, 5, 7],
        [0, 2, 5, 5.6],  # Provisional spectator bond 2-3.
        [0, 0.6, 5, 7],  # It reverts as the independent 0-1 change starts.
        [0, 0.6, 1.2, 1.8],  # The pending region now merges with atoms 2 and 3.
        final_positions[ending],
        final_positions[ending],
        final_positions[ending],
    ]
    emitted = []
    for frame, x in enumerate(positions):
        atoms = assign_atom_ids(Atoms("H4", positions=[[value, 0, 0] for value in x]))
        detector.process(atoms, frame=frame)
        emitted.extend(
            tracker.process(
                atoms,
                frame=frame,
                stable_bonds=detector.stable_bonds,
                pending_bonds=detector.pending_bonds,
            )
        )
        if frame == checkpoint_frame:
            tracker = ReactionTracker.read_checkpoint(
                tracker.write_checkpoint(tmp_path / "tracker")
            )
            detector = BondChangeDetector.from_state(detector.export_state())

    (candidate,) = emitted
    assert candidate.observed_frame == 5
    if ending == "merged":
        assert not candidate.resolved
        assert (candidate.reactant_frame, candidate.product_frame) == (1, 3)
        assert candidate.atom_ids == (0, 1, 2, 3)
        assert candidate.reactant_bonds == frozenset({(2, 3)})
        assert candidate.reactant.get_distance(2, 3) == pytest.approx(0.6)
        assert candidate.product_bonds == frozenset({(0, 1), (1, 2), (2, 3)})
    else:
        pair = (0, 1) if ending == "first" else (2, 3)
        assert candidate.resolved
        assert candidate.atom_ids == pair
        assert candidate.reactant_frame == (1 if ending == "first" else 2)
        assert candidate.product_frame == 4
        assert candidate.reactant_bonds == frozenset()
        assert candidate.product_bonds == frozenset({pair})
        assert candidate.reactant.get_distance(*pair) > 1.2
        assert candidate.product.get_distance(*pair) == pytest.approx(0.6)
    assert tracker.finish() == ()


def test_legacy_tracker_checkpoint_continues_each_pending_region(tmp_path) -> None:
    import hashlib
    import json

    from ase.io import write

    ids = (0, 1, 2, 3)
    root = tmp_path / "legacy"
    root.mkdir()
    files = {}
    for name, frame in (("accepted.traj", 0), ("reactant.traj", 0), ("pending_product.traj", 1)):
        atoms = tracked_frame("H4", frame, ids)
        atoms.info["atom_ids"] = list(ids)
        write(root / name, atoms)
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    value = {
        "schema_version": 1,
        "stability_frames": 2,
        "symbols": [[key, "H"] for key in ids],
        "accepted_bonds": [],
        "accepted_frame": 0,
        "reactant_frame": 0,
        "pending": {"bonds": [[0, 1], [2, 3]], "product_frame": 1, "count": 1},
        "last_frame": 1,
        "files": files,
    }
    (root / "tracker.json").write_text(json.dumps(value))
    tracker = ReactionTracker.read_checkpoint(root)
    candidates = tracker.process(
        tracked_frame("H4", 2, ids), frame=2, stable_bonds=[(0, 1), (2, 3)], pending_bonds=None
    )
    assert [item.atom_ids for item in candidates] == [(0, 1), (2, 3)]
    assert all(item.resolved and item.reactant_frame == 0 for item in candidates)
    assert tracker.finish() == ()
