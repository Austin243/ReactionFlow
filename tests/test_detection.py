from __future__ import annotations

import json

import numpy as np
from ase import Atoms

from reactionflow import BondChangeDetector, BondDetectorConfig, assign_atom_ids


def pair_frame(
    symbols: str,
    distance: float,
    *,
    ids: tuple[int, int] = (10, 20),
    cell: object = None,
    pbc: object = False,
) -> Atoms:
    atoms = Atoms(
        symbols,
        positions=[[0, 0, 0], [distance, 0, 0]],
        cell=cell,
        pbc=pbc,
    )
    atoms.set_array("atom_id", np.asarray(ids))
    return atoms


def config(*, persistence: int = 1, pair: str = "C-O") -> BondDetectorConfig:
    return BondDetectorConfig(
        persistence_frames=persistence,
        pair_thresholds={pair: (1.5, 1.8)},
    )


def test_persistence_hysteresis_formation_and_breakage() -> None:
    detector = BondChangeDetector(config(persistence=2))
    assert detector.process(pair_frame("CO", 2.0), frame=0) == []

    assert detector.process(pair_frame("CO", 1.5), frame=1) == []
    assert detector.pending_bonds == frozenset({(10, 20)})
    assert detector.process(pair_frame("CO", 1.6), frame=2) == []  # reset
    assert detector.pending_bonds is None
    assert detector.process(pair_frame("CO", 1.4), frame=3) == []
    formed = detector.process(pair_frame("CO", 1.3), frame=4)[0]
    assert (formed.event_type, formed.atom_ids, formed.symbols) == (
        "formed",
        (10, 20),
        ("C", "O"),
    )
    assert (formed.first_seen_frame, formed.confirmed_frame) == (3, 4)

    assert detector.process(pair_frame("CO", 1.8), frame=5) == []
    broken = detector.process(pair_frame("CO", 1.9), frame=6)[0]
    assert broken.event_type == "broken"
    assert detector.stable_bonds == frozenset()
    json.dumps(broken.to_dict())


def test_stable_ids_survive_atom_reordering() -> None:
    initial = assign_atom_ids(Atoms("CO", positions=[[0, 0, 0], [2.0, 0, 0]]))
    detector = BondChangeDetector(config())
    detector.process(initial, frame=0)

    reordered = Atoms("OC", positions=[[1.4, 0, 0], [0, 0, 0]])
    reordered.set_array("atom_id", np.asarray([1, 0]))
    event = detector.process(reordered, frame=1)[0]
    assert event.atom_ids == (0, 1)
    assert event.symbols == ("C", "O")


def test_periodic_minimum_image_bond_can_break_outside_the_cutoff() -> None:
    atoms = pair_frame(
        "CC",
        9.6,
        cell=[10, 10, 10],
        pbc=True,
    )
    detector = BondChangeDetector(config(pair="C-C"))
    detector.process(atoms, frame=0)
    assert detector.stable_bonds == frozenset({(10, 20)})

    broken = detector.process(pair_frame("CC", 10.0), frame=1)[0]
    assert broken.event_type == "broken"


def test_json_checkpoint_continues_a_pending_change() -> None:
    detector_config = config(persistence=2)
    detector = BondChangeDetector(detector_config)
    detector.process(pair_frame("CO", 2.0), frame=0)
    detector.process(pair_frame("CO", 1.4), frame=1)

    payload = json.loads(json.dumps(detector.export_state()))
    restored = BondChangeDetector.from_state(payload, config=detector_config)
    event = restored.process(pair_frame("CO", 1.3), frame=2)[0]
    assert event.event_type == "formed"


def test_equal_thresholds_are_element_neutral() -> None:
    outcomes = []
    for symbols in ("HH", "CO", "SiCl"):
        pair = "-".join(Atoms(symbols).get_chemical_symbols())
        detector = BondChangeDetector(config(pair=pair))
        detector.process(pair_frame(symbols, 2.0), frame=0)
        event = detector.process(pair_frame(symbols, 1.4), frame=1)[0]
        outcomes.append((event.event_type, event.atom_ids, event.confirmed_frame))

    assert outcomes == [("formed", (10, 20), 1)] * 3
