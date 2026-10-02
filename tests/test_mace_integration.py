"""Optional local-weight MACE check; never downloads a checkpoint.

Set REACTIONFLOW_TEST_MACE_CHECKPOINT to an absolute existing model path, then run
``python -m pytest -q tests/test_mace_integration.py`` in the MACE environment.
MH-1 and Polar checks use REACTIONFLOW_TEST_MACE_MH1_CHECKPOINT and
REACTIONFLOW_TEST_MACE_POLAR_CHECKPOINT respectively; all weights stay local.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk, molecule
from ase.calculators.fd import calculate_numerical_forces

from reactionflow.adapters.mace import MACEAdapter
from reactionflow.campaign import TrajectorySpec


def _checkpoint(variable: str) -> Path:
    configured = os.environ.get(variable)
    checkpoint = Path(configured) if configured else None
    if checkpoint is None or not checkpoint.is_absolute() or not checkpoint.is_file():
        pytest.skip(f"set {variable} to an existing absolute checkpoint path")
    return checkpoint


def test_real_mace_npt_restart_is_bitwise_identical() -> None:
    checkpoint = _checkpoint("REACTIONFLOW_TEST_MACE_CHECKPOINT")

    trajectory = TrajectorySpec.from_dict(
        {
            "id": "real-mace-restart",
            "total_steps": 2,
            "timestep_fs": 0.1,
            "temperature_K": 100.0,
            "pressure_GPa": 0.0,
            "seed": 17,
        }
    )
    options = {"checkpoint": str(checkpoint), "device": "cpu", "dtype": "float64"}
    atoms = bulk("Si", "diamond", a=5.43)
    assert len(atoms) == 2

    adapter = MACEAdapter(trajectory=trajectory, options=options)
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(2)
        assert np.isfinite(runtime.atoms.get_potential_energy())
        assert np.isfinite(runtime.atoms.get_forces()).all()
        assert np.isfinite(runtime.atoms.get_stress()).all()
        full = runtime.snapshot()

    with adapter.start(atoms.copy()) as runtime:
        runtime.run(1)
        split = runtime.snapshot()
    fresh = MACEAdapter(trajectory=trajectory, options=options)
    with fresh.restore(split) as runtime:
        runtime.run(1)
        resumed = runtime.snapshot()

    assert resumed.calculator == full.calculator
    assert resumed.dynamics.metadata == full.dynamics.metadata
    assert resumed.atoms.arrays.keys() == full.atoms.arrays.keys()
    for name in full.atoms.arrays:
        np.testing.assert_array_equal(resumed.atoms.arrays[name], full.atoms.arrays[name])
    np.testing.assert_array_equal(resumed.atoms.cell.array, full.atoms.cell.array)
    np.testing.assert_array_equal(resumed.atoms.pbc, full.atoms.pbc)
    assert resumed.dynamics.arrays.keys() == full.dynamics.arrays.keys()
    for name in full.dynamics.arrays:
        np.testing.assert_array_equal(resumed.dynamics.arrays[name], full.dynamics.arrays[name])


def _fixed_cell_trajectory() -> TrajectorySpec:
    return TrajectorySpec(
        id="real-mace-fixed-cell",
        total_steps=2,
        timestep_fs=0.1,
        temperature_K=100.0,
        pressure_GPa=None,
        seed=17,
    )


@pytest.mark.parametrize(
    "head",
    ["omat_pbe", "omol", "spice_wB97M", "oc20_usemppbe", "matpes_r2scan", "mp_pbe_refit_add"],
)
def test_real_mh1_released_heads(head: str) -> None:
    checkpoint = _checkpoint("REACTIONFLOW_TEST_MACE_MH1_CHECKPOINT")
    adapter = MACEAdapter(
        trajectory=_fixed_cell_trajectory(),
        options={"checkpoint": str(checkpoint), "head": head, "device": "cpu"},
    )
    with adapter.calculator("neb") as calculator:
        atoms = molecule("H2O")
        atoms.calc = calculator
        assert calculator.head == head
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()


def test_real_polar_fixed_cell_forces_field_cache_and_exact_restart() -> None:
    checkpoint = _checkpoint("REACTIONFLOW_TEST_MACE_POLAR_CHECKPOINT")
    options = {"checkpoint": str(checkpoint), "family": "polar", "device": "cpu"}
    trajectory = _fixed_cell_trajectory()
    adapter = MACEAdapter(trajectory=trajectory, options=options)
    atoms = molecule("H2O")
    atoms.set_cell([4.6, 4.8, 5.0])
    atoms.center()
    atoms.pbc = True
    atoms.info.update(charge=0, spin=1, external_field=np.zeros(3))
    with adapter.calculator("neb") as calculator:
        probe = atoms.copy()
        probe.calc = calculator
        energy = probe.get_potential_energy()
        force = probe.get_forces()
        numerical = calculate_numerical_forces(probe, eps=1e-5)
        np.testing.assert_allclose(force, numerical, atol=1e-4, rtol=0)
        assert np.isfinite(probe.get_stress()).all()
        probe.info["external_field"][2] = 0.01
        field_energy = probe.get_potential_energy()
        assert field_energy != energy
        with MACEAdapter(trajectory=trajectory, options=options).calculator("neb") as fresh:
            assert field_energy == fresh.get_potential_energy(probe)
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(2)
        full = runtime.snapshot()
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(1)
        split = runtime.snapshot()
    with MACEAdapter(trajectory=trajectory, options=options).restore(split) as runtime:
        runtime.run(1)
        resumed = runtime.snapshot()
    assert resumed.calculator == full.calculator
    assert resumed.dynamics.metadata == full.dynamics.metadata
    for name in full.atoms.arrays:
        np.testing.assert_array_equal(resumed.atoms.arrays[name], full.atoms.arrays[name])
    for name in full.dynamics.arrays:
        np.testing.assert_array_equal(resumed.dynamics.arrays[name], full.dynamics.arrays[name])
    np.testing.assert_array_equal(resumed.atoms.cell.array, full.atoms.cell.array)


def test_real_renamed_polar_checkpoint_cannot_bypass_family_guard(tmp_path: Path) -> None:
    checkpoint = _checkpoint("REACTIONFLOW_TEST_MACE_POLAR_CHECKPOINT")
    renamed = tmp_path / "ordinary.model"
    renamed.symlink_to(checkpoint)
    trajectory = TrajectorySpec(
        id="mislabeled-polar",
        total_steps=2,
        timestep_fs=0.1,
        temperature_K=100.0,
        pressure_GPa=0.0,
        seed=17,
    )
    adapter = MACEAdapter(
        trajectory=trajectory, options={"checkpoint": str(renamed), "device": "cpu"}
    )
    with (
        pytest.raises(ValueError, match="checkpoint architecture does not match family"),
        adapter.calculator("neb"),
    ):
        pytest.fail("a Polar checkpoint cannot be leased without family='polar'")


def test_real_polar_six_stress_components_and_npt_exact_restart() -> None:
    checkpoint = _checkpoint("REACTIONFLOW_TEST_MACE_POLAR_CHECKPOINT")
    options = {"checkpoint": str(checkpoint), "family": "polar", "device": "cpu"}
    trajectory = TrajectorySpec(
        id="polar-npt",
        total_steps=2,
        timestep_fs=0.1,
        temperature_K=100.0,
        pressure_GPa=0.2,
        seed=17,
    )
    atoms = molecule("H2O")
    atoms.rotate(31, [1, 2, 3])
    atoms.set_cell([[4.6, 0.2, 0.1], [0.3, 4.8, 0.4], [0.2, 0.1, 5.0]])
    atoms.center()
    atoms.pbc = True
    atoms.info.update(charge=0, spin=1, external_field=[0.02, -0.01, 0.015])
    adapter = MACEAdapter(trajectory=trajectory, options=options)
    with adapter.calculator("neb") as calculator:
        probe = atoms.copy()
        probe.calc = calculator
        analytic_forces = probe.get_forces()
        stress = probe.get_stress()
        numerical = []
        # An independent symmetric-strain stencil with a different step tests
        # all three normal and three shear components, including their factors.
        step = 1e-4
        for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
            direction = np.zeros((3, 3))
            direction[i, j] = direction[j, i] = 1.0 if i == j else 0.5
            energies = []
            for sign in (1, -1):
                displaced = atoms.copy()
                displaced.set_cell(
                    atoms.cell @ (np.eye(3) + sign * step * direction), scale_atoms=True
                )
                energies.append(calculator.inner.get_potential_energy(displaced))
            numerical.append((energies[0] - energies[1]) / (2 * step * atoms.get_volume()))
        np.testing.assert_allclose(stress, numerical, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(probe.get_forces(), analytic_forces)
        np.testing.assert_array_equal(probe.positions, atoms.positions)
        np.testing.assert_array_equal(probe.cell, atoms.cell)
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(2)
        full = runtime.snapshot()
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(1)
        split = runtime.snapshot()
    with MACEAdapter(trajectory=trajectory, options=options).restore(split) as runtime:
        runtime.run(1)
        resumed = runtime.snapshot()
    assert full.calculator.metadata["stress"] == {
        "method": "central_finite_difference_free_energy",
        "eps_strain": 1e-5,
        "required_dtype": "float64",
    }
    assert resumed.calculator == full.calculator
    assert resumed.dynamics.metadata == full.dynamics.metadata
    for name in full.atoms.arrays:
        np.testing.assert_array_equal(resumed.atoms.arrays[name], full.atoms.arrays[name])
    for name in full.dynamics.arrays:
        np.testing.assert_array_equal(resumed.dynamics.arrays[name], full.dynamics.arrays[name])
    np.testing.assert_array_equal(resumed.atoms.cell.array, full.atoms.cell.array)
    assert not np.array_equal(full.atoms.cell.array, atoms.cell.array)
