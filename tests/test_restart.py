from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevinbaoab import LangevinBAOAB

from reactionflow import ComponentState, ExactRestartSnapshot
from reactionflow.ase_npt import restore_langevin_baoab, snapshot_langevin_baoab


class HarmonicSolid(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.positions
        self.results = {
            "energy": float(0.5 * np.sum(positions**2)),
            "forces": -positions.copy(),
            "stress": np.zeros(6),
        }


def _atoms() -> Atoms:
    atoms = Atoms(
        "H2",
        positions=[[0.2, 0.1, 0.3], [1.1, 0.4, 0.2]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.set_momenta([[0.03, -0.01, 0.02], [-0.02, 0.04, -0.03]])
    atoms.set_array("atom_id", np.asarray([10, 20]))
    return atoms


def _dynamics(atoms: Atoms, *, seed: int, hydrostatic: bool = True) -> LangevinBAOAB:
    atoms.calc = HarmonicSolid()
    return LangevinBAOAB(
        atoms,
        timestep=0.25 * units.fs,
        temperature_K=300.0,
        externalstress=-0.1 * units.GPa,
        hydrostatic=hydrostatic,
        T_tau=25.0 * units.fs,
        P_tau=250.0 * units.fs,
        P_mass=5000.0,
        rng=np.random.default_rng(seed),
        logfile=None,
    )


def test_restart_artifact_round_trips_and_checks_integrity(tmp_path) -> None:
    atoms = _atoms()
    dynamics = _dynamics(atoms, seed=5)
    dynamics.run(2)
    snapshot = ExactRestartSnapshot(
        atoms=atoms,
        dynamics=snapshot_langevin_baoab(dynamics),
        calculator=ComponentState(
            kind="test.harmonic-solid",
            metadata={"parameters": (1, np.asarray([2, 3], dtype=np.int16))},
        ),
    )

    destination = snapshot.write(tmp_path / "restart-0001")
    restored = ExactRestartSnapshot.read(destination)

    np.testing.assert_array_equal(restored.atoms.positions, snapshot.atoms.positions)
    np.testing.assert_array_equal(restored.atoms.get_momenta(), snapshot.atoms.get_momenta())
    assert restored.calculator.kind == "test.harmonic-solid"
    assert restored.calculator.metadata["parameters"][0] == 1
    np.testing.assert_array_equal(
        restored.calculator.metadata["parameters"][1],
        np.asarray([2, 3], dtype=np.int16),
    )
    with pytest.raises(FileExistsError):
        snapshot.write(destination)

    arrays = destination / "arrays.npz"
    arrays.write_bytes(arrays.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="integrity check"):
        ExactRestartSnapshot.read(destination)


@pytest.mark.parametrize("hydrostatic", [True, False])
def test_langevin_baoab_restart_matches_uninterrupted_npt_exactly(tmp_path, hydrostatic) -> None:
    uninterrupted_atoms = _atoms()
    uninterrupted = _dynamics(uninterrupted_atoms, seed=77, hydrostatic=hydrostatic)
    uninterrupted.run(12)

    split_atoms = _atoms()
    split = _dynamics(split_atoms, seed=77, hydrostatic=hydrostatic)
    split.run(5)
    saved = ExactRestartSnapshot(
        atoms=split_atoms,
        dynamics=snapshot_langevin_baoab(split),
        calculator=ComponentState(kind="test.harmonic-solid", metadata={"spring": 1.0}),
    ).write(tmp_path / "restart")

    loaded = ExactRestartSnapshot.read(saved)
    resumed = restore_langevin_baoab(loaded.atoms, HarmonicSolid(), loaded.dynamics)
    resumed.run(7)

    assert resumed.nsteps == uninterrupted.nsteps == 12
    np.testing.assert_array_equal(loaded.atoms.positions, uninterrupted_atoms.positions)
    np.testing.assert_array_equal(loaded.atoms.get_momenta(), uninterrupted_atoms.get_momenta())
    np.testing.assert_array_equal(loaded.atoms.cell.array, uninterrupted_atoms.cell.array)
    np.testing.assert_array_equal(resumed.accel, uninterrupted.accel)
    np.testing.assert_array_equal(np.asarray(resumed.p_eps), np.asarray(uninterrupted.p_eps))
    np.testing.assert_array_equal(
        np.asarray(resumed.force_eps), np.asarray(uninterrupted.force_eps)
    )
    np.testing.assert_array_equal(
        np.asarray(resumed.gamma_mod), np.asarray(uninterrupted.gamma_mod)
    )
    assert resumed.rng.bit_generator.state == uninterrupted.rng.bit_generator.state


def test_exact_snapshot_rejects_inexact_components() -> None:
    with pytest.raises(ValueError, match="calculator component"):
        ExactRestartSnapshot(
            atoms=_atoms(),
            dynamics=ComponentState(kind="driver"),
            calculator=ComponentState(kind="calculator", exact=False),
        )


def test_langevin_restart_rejects_runtime_version_drift() -> None:
    atoms = _atoms()
    dynamics = _dynamics(atoms, seed=9)
    state = snapshot_langevin_baoab(dynamics)
    changed = ComponentState(
        kind=state.kind,
        metadata={**state.metadata, "ase_version": "different"},
        arrays=state.arrays,
    )

    with pytest.raises(ValueError, match="requires ASE different"):
        restore_langevin_baoab(_atoms(), HarmonicSolid(), changed)
