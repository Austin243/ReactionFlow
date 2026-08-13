from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

import reactionflow.segments as segment_module
from reactionflow import atom_ids
from reactionflow.segments import ResumeToken, SegmentStore


def test_structural_checkpoint_resumes_into_a_new_generation(tmp_path) -> None:
    atoms = Atoms(
        "H2",
        positions=[[0, 0, 0], [0.75, 0, 0]],
        cell=[8, 9, 10],
        pbc=[True, False, True],
        info={"atom_ids": [10, 20]},
    )
    atoms.set_momenta([[0.1, 0, 0], [-0.1, 0, 0]])
    atoms.calc = EMT()

    store = SegmentStore(tmp_path)
    first = store.start(atoms, global_step=4, global_frame=2)
    first.atoms.positions[1, 0] = 0.8
    token = store.checkpoint(first, first.atoms, global_step=14, global_frame=4)

    restored = store.resume(ResumeToken.read(token.path))
    assert (first.generation, restored.generation) == (0, 1)
    assert first.trajectory_path == tmp_path / "segments/0000/trajectory.traj"
    assert restored.trajectory_path == tmp_path / "segments/0001/trajectory.traj"
    assert first.trajectory_path != restored.trajectory_path
    assert (restored.global_step, restored.global_frame) == (14, 4)
    assert atom_ids(restored.atoms) == (10, 20)
    assert "atom_ids" not in restored.atoms.info
    np.testing.assert_allclose(restored.atoms.positions, first.atoms.positions)
    np.testing.assert_allclose(restored.atoms.get_momenta(), first.atoms.get_momenta())
    np.testing.assert_allclose(restored.atoms.cell, first.atoms.cell)
    np.testing.assert_array_equal(restored.atoms.pbc, first.atoms.pbc)
    assert restored.atoms.calc is None


def test_checkpoint_publication_is_atomic_and_generations_are_immutable(
    tmp_path, monkeypatch
) -> None:
    atoms = Atoms("He", positions=[[0, 0, 0]])
    atoms.set_array("atom_id", np.asarray([7]))
    store = SegmentStore(tmp_path)
    first = store.start(atoms)

    write_token = segment_module._write_token

    def fail_before_publication(*_args, **_kwargs) -> None:
        raise OSError("simulated token failure")

    monkeypatch.setattr(segment_module, "_write_token", fail_before_publication)
    with pytest.raises(OSError, match="simulated token failure"):
        store.checkpoint(first, first.atoms, global_step=1, global_frame=1)
    assert not (first.directory / "checkpoint").exists()
    assert not list(first.directory.glob(".checkpoint-*.tmp"))

    changed_identity = first.atoms.copy()
    changed_identity.set_array("atom_id", None)
    with pytest.raises(ValueError, match="stable atom IDs"):
        store.checkpoint(first, changed_identity, global_step=1, global_frame=1)
    changed_identity = first.atoms.copy()
    changed_identity.numbers[0] = 1
    with pytest.raises(ValueError, match="identities do not match"):
        store.checkpoint(first, changed_identity, global_step=1, global_frame=1)

    monkeypatch.setattr(segment_module, "_write_token", write_token)
    token = store.checkpoint(first, first.atoms, global_step=1, global_frame=1)
    assert token.path.is_file() and token.checkpoint_path.is_file()
    with pytest.raises(FileExistsError, match="already checkpointed"):
        store.checkpoint(first, first.atoms, global_step=2, global_frame=2)

    store.resume(token)
    with pytest.raises(FileExistsError):
        store.resume(token)
