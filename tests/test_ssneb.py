from __future__ import annotations

from contextlib import contextmanager
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms

from reactionflow import BondDetectorConfig, PathwayConfig, ReactionCandidate, refine_pathway
from reactionflow.ssneb import cell_filter


class VolumeDoubleWell(Calculator):
    """E = (q² - 1/4)² + K/2 (V - V0 - alpha*q)², with fractional-coordinate q."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]
    stiffness = 0.02
    reference_volume = 216.0
    coupling = 12.0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        scaled = self.atoms.get_scaled_positions(wrap=False)
        q = 6 * (scaled[1, 0] - scaled[0, 0]) - 1.1
        volume_offset = self.atoms.get_volume() - self.reference_volume - self.coupling * q
        volume_derivative = self.stiffness * volume_offset
        derivative = 4 * q * (q * q - 0.25) - self.coupling * volume_derivative
        forces = np.zeros((len(self.atoms), 3))
        forces[0] = derivative * 6 * np.linalg.inv(self.atoms.cell)[:, 0]
        forces[1] = -forces[0]
        self.results = {
            "energy": (q * q - 0.25) ** 2 + 0.5 * self.stiffness * volume_offset**2,
            "forces": forces,
            "stress": np.diag([volume_derivative] * 3),
        }


@pytest.mark.parametrize("pressure_eV_A3", [0.0, 0.005])
def test_ssneb_finds_pressure_dependent_enthalpy_barrier(pressure_eV_A3):
    reactant = Atoms("H2He", cell=[6, 6, 6], pbc=True)
    reactant.set_array("atom_id", np.arange(3))
    reactant.set_scaled_positions([[0.2, 0.2, 0.2], [0.3, 0.2, 0.2], [0.7, 0.7, 0.7]])
    product = reactant.copy()
    product.positions[0, 0] -= 0.5
    product.positions[1, 0] += 0.5
    # Exercise equal initial cells and drift beyond the old fixed-cell limit.
    if pressure_eV_A3:
        product.set_cell([6.2, 6.2, 6.2], scale_atoms=True)
    candidate = ReactionCandidate(
        reactant=reactant,
        product=product,
        atom_ids=(0, 1),
        reactant_bonds=frozenset({(0, 1)}),
        product_bonds=frozenset(),
        reactant_frame=0,
        product_frame=1,
        observed_frame=2,
        resolved=True,
    )

    @contextmanager
    def volume_calculator(_stage):
        yield VolumeDoubleWell()

    outcome = refine_pathway(
        candidate,
        calculator_provider=volume_calculator,
        pressure_GPa=pressure_eV_A3 / units.GPa,
        detector_config=BondDetectorConfig(pair_thresholds={"H-H": (0.8, 1.2)}),
        config=PathwayConfig(
            active_radius=0.2,
            relax_fmax=0.001,
            images=5,
            neb_fmax=0.002,
        ),
    )
    assert outcome.converged, outcome.message
    assert outcome.method == "ssneb" and outcome.barrier_quantity == "enthalpy"
    # Stationary points satisfy dH/dq = 4q(q² - 1/4) + p*alpha = 0.
    roots = np.sort(np.roots([4, 0, -1, pressure_eV_A3 * VolumeDoubleWell.coupling]))
    expected_volumes = (
        VolumeDoubleWell.reference_volume
        + VolumeDoubleWell.coupling * roots
        - pressure_eV_A3 / VolumeDoubleWell.stiffness
    )
    expected_enthalpies = (
        (roots * roots - 0.25) ** 2
        + pressure_eV_A3 * (VolumeDoubleWell.reference_volume + VolumeDoubleWell.coupling * roots)
        - pressure_eV_A3**2 / (2 * VolumeDoubleWell.stiffness)
    )
    assert outcome.barrier == pytest.approx(
        expected_enthalpies[1] - expected_enthalpies[0], abs=2e-4
    )
    saddle = outcome.frequency_validation.transition_state_index
    assert saddle == 1 + int(np.argmax(outcome.enthalpies[1:-1]))
    np.testing.assert_allclose(
        np.asarray(outcome.volumes)[[0, saddle, -1]], expected_volumes, atol=0.03
    )
    np.testing.assert_allclose(
        outcome.enthalpies,
        np.asarray(outcome.energies) + pressure_eV_A3 * np.asarray(outcome.volumes),
    )
    assert np.ptp(outcome.volumes) > 10
    for image in outcome.images:
        np.testing.assert_allclose(image.get_scaled_positions(wrap=False)[2], [0.7, 0.7, 0.7])


def test_cell_and_atomic_forces_are_enthalpy_derivatives_in_a_sheared_cell():
    atoms = bulk("Ar", "fcc", a=5.2, cubic=True)
    reference = atoms.cell.copy()
    atoms.set_cell([[5.0, 0, 0], [0.5, 5.3, 0], [0.15, 0.2, 4.9]], scale_atoms=True)
    atoms.positions[0] += [0.1, -0.08, 0.03]
    atoms.set_constraint(FixAtoms(indices=[3]))
    atoms.calc = LennardJones(sigma=3.4, epsilon=0.01, smooth=True)
    filtered = cell_filter(atoms, 0.4, reference)
    positions = filtered.get_positions()
    forces = filtered.get_forces()
    delta = 1e-5
    coordinates = [(i, j) for i in range(3) for j in range(3)]
    coordinates += [(len(atoms) + i, j) for i in range(3) for j in range(i, 3)]
    for index in coordinates:
        plus, minus = positions.copy(), positions.copy()
        plus[index] += delta
        minus[index] -= delta
        filtered.set_positions(plus)
        eplus = filtered.get_potential_energy(force_consistent=False)
        filtered.set_positions(minus)
        eminus = filtered.get_potential_energy(force_consistent=False)
        assert forces[index] == pytest.approx(-(eplus - eminus) / (2 * delta), abs=1e-8)
    filtered.set_positions(positions)
    np.testing.assert_allclose(filtered.get_forces()[3], 0)
