from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms

import reactionflow.store as store_module
from reactionflow import BondDetectorConfig, ReactionCandidate, atom_ids
from reactionflow.store import OccurrenceStore


def candidate(
    ids: tuple[int, int, int],
    *,
    symbols: tuple[str, str, str] = ("C", "O", "H"),
    resolved: bool = True,
    reverse: bool = False,
    offset: float = 0,
) -> ReactionCandidate:
    def endpoint(shift: float) -> Atoms:
        atoms = Atoms(
            symbols,
            positions=[[offset + shift + index, index / 10, 0] for index in range(3)],
            cell=[8, 9, 10],
            pbc=[True, False, True],
        )
        atoms.set_array("atom_id", np.asarray(ids))
        return atoms

    reactant_bonds = {(ids[0], ids[1]), (ids[1], ids[2])}
    product_bonds = {(ids[0], ids[1]), (ids[0], ids[2])}
    if reverse:
        reactant_bonds, product_bonds = product_bonds, reactant_bonds
    return ReactionCandidate(
        reactant=endpoint(0),
        product=endpoint(0.25),
        atom_ids=ids,
        reactant_bonds=frozenset(reactant_bonds),
        product_bonds=frozenset(product_bonds),
        reactant_frame=10,
        product_frame=11,
        observed_frame=12,
        resolved=resolved,
    )


def test_store_retains_classes_representatives_and_endpoint_bundles(tmp_path) -> None:
    config = BondDetectorConfig(persistence_frames=2, pair_thresholds={"C-O": (1.4, 1.8)})
    store = OccurrenceStore(tmp_path)
    unresolved, inserted = store.register(
        "segment-0-unresolved",
        candidate((1, 2, 3), resolved=False),
        detector_config=config,
    )
    assert inserted
    representative, inserted = store.register(
        "segment-1-forward",
        candidate((10, 20, 30), reverse=True, offset=2),
        detector_config=config,
    )
    assert inserted
    duplicate, inserted = store.register(
        "segment-2-duplicate",
        candidate((100, 200, 300), offset=4),
        detector_config=config,
    )
    assert inserted
    distinct, inserted = store.register(
        "segment-3-distinct",
        candidate((7, 8, 9), symbols=("C", "O", "Cl")),
        detector_config=config,
    )
    assert inserted

    assert unresolved.class_id == representative.class_id == duplicate.class_id
    assert [record.is_representative for record in store.records()] == [False, True, False, True]
    assert distinct.class_id != representative.class_id

    reopened = OccurrenceStore(tmp_path)
    assert [record.occurrence_id for record in reopened.records()] == [
        "segment-0-unresolved",
        "segment-1-forward",
        "segment-2-duplicate",
        "segment-3-distinct",
    ]
    loaded = reopened.load("segment-1-forward")
    assert atom_ids(loaded.reactant) == (10, 20, 30)
    expected = candidate((10, 20, 30), offset=2)
    np.testing.assert_allclose(loaded.product.positions, expected.product.positions)
    assert loaded.product_bonds == frozenset({(10, 20), (20, 30)})
    assert {path.name for path in representative.directory.iterdir()} == {
        "candidate.json",
        "reactant.traj",
        "product.traj",
    }
    metadata = json.loads((representative.directory / "candidate.json").read_text())
    assert metadata["detector_config"] == config.to_dict()
    assert reopened.load_detector_config("segment-1-forward") == config


def test_store_is_idempotent_and_recovers_atomic_publication(tmp_path, monkeypatch) -> None:
    config = BondDetectorConfig()
    store = OccurrenceStore(tmp_path)
    first = candidate((1, 2, 3))
    record, inserted = store.register("same-id", first, detector_config=config)
    assert inserted
    replayed, inserted = store.register("same-id", first, detector_config=config)
    assert replayed == record
    assert not inserted
    assert len(store.records()) == 1

    with pytest.raises(ValueError, match="conflicting data"):
        store.register("same-id", candidate((1, 2, 3), offset=9), detector_config=config)

    real_write = store_module.write

    def fail_on_product(path, atoms) -> None:
        if path.name == "product.traj":
            raise OSError("simulated write failure")
        real_write(path, atoms)

    monkeypatch.setattr(store_module, "write", fail_on_product)
    with pytest.raises(OSError, match="simulated write failure"):
        store.register("write-failed", first, detector_config=config)
    assert not (tmp_path / "candidates" / "write-failed").exists()
    assert not (tmp_path / "candidates" / ".write-failed.tmp").exists()
    monkeypatch.setattr(store_module, "write", real_write)

    def fail_before_database(*_args, **_kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(store, "_insert", fail_before_database)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        store.register("published-only", candidate((10, 20, 30)), detector_config=config)
    assert (tmp_path / "candidates" / "published-only").is_dir()

    recovered = OccurrenceStore(tmp_path)
    assert [item.occurrence_id for item in recovered.records()] == ["same-id", "published-only"]
    assert recovered.records()[1].class_id == record.class_id
    assert recovered.records()[1].is_representative is False

    record.directory.rename(tmp_path / "removed-bundle")
    with pytest.raises(FileNotFoundError):
        recovered.register("same-id", first, detector_config=config)
