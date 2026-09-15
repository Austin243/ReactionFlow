from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms

from reactionflow import BondDetectorConfig, PathwayConfig, ReactionCandidate, refine_pathway
from reactionflow.pathway import _prepare_endpoints
from reactionflow.run import ReactionRun
from reactionflow.ssneb import SSNEB, cell_filter


class VolumeDoubleWell(Calculator):
    """Analytic coupled atom/volume potential, with q measured in the reference cell.

    E = (q^2 - 1/4)^2 + K/2 (V - V0 - alpha*q)^2.
    At pressure p: V = V0 + alpha*q - p/K and
    dH/dq = 4*q*(q^2 - 1/4) + p*alpha.
    """

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


@contextmanager
def volume_calculator(_stage):
    yield VolumeDoubleWell()


def volume_candidate():
    reactant = Atoms("H2He", cell=[6, 6, 6], pbc=True)
    reactant.set_array("atom_id", np.arange(3))
    reactant.set_scaled_positions([[0.2, 0.2, 0.2], [0.3, 0.2, 0.2], [0.7, 0.7, 0.7]])
    product = reactant.copy()
    product.positions[0, 0] -= 0.5
    product.positions[1, 0] += 0.5
    # Larger than the old fixed-cell mapping limit; SSNEB must accept this.
    product.set_cell([6.2, 6.2, 6.2], scale_atoms=True)
    return ReactionCandidate(
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


@pytest.mark.parametrize("pressure_eV_A3", [0.0, 0.005])
def test_ssneb_finds_pressure_dependent_barrier_and_preserves_md_endpoints(
    pressure_eV_A3,
    tmp_path,
):
    candidate = volume_candidate()
    if pressure_eV_A3 == 0:
        candidate.product.set_cell(candidate.reactant.cell, scale_atoms=True)
    original = [
        (image.positions.copy(), image.cell.copy())
        for image in (candidate.reactant, candidate.product)
    ]
    outcome = refine_pathway(
        candidate,
        calculator_provider=volume_calculator,
        pressure_GPa=pressure_eV_A3 / units.GPa,
        detector_config=BondDetectorConfig(pair_thresholds={"H-H": (0.8, 1.2)}),
        config=PathwayConfig(
            active_radius=0.2,
            relax_fmax=0.001,
            relax_steps=500,
            images=5,
            neb_fmax=0.002,
            neb_steps=500,
            ci_neb_steps=500,
        ),
    )
    assert outcome.converged, outcome.message
    assert outcome.method == "ssneb" and outcome.barrier_quantity == "enthalpy"
    assert len(outcome.volumes) == len(outcome.enthalpies) == len(outcome.energies) == 5
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
    # The unstable direction couples q and V. At fixed V, the added
    # K*alpha^2 curvature makes the atomic Hessian positive: this existing
    # diagnostic must not be mistaken for full SSNEB saddle validation.
    assert outcome.frequency_validation.status == "zero_imaginary_modes"
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
        assert image.calc is None
        np.testing.assert_allclose(image.get_scaled_positions(wrap=False)[2], [0.7, 0.7, 0.7])
    for image, (positions, cell) in zip(
        (candidate.reactant, candidate.product), original, strict=True
    ):
        np.testing.assert_array_equal(image.positions, positions)
        np.testing.assert_array_equal(image.cell, cell)

    # Persist and reload the actual variable-cell band, including its provenance.
    run = ReactionRun.create(tmp_path)
    record, _ = run.occurrences.register(
        "ssneb-test",
        candidate,
        detector_config=BondDetectorConfig(),
    )
    run._publish_outcome(record.occurrence_id, outcome)
    loaded = ReactionRun.open(tmp_path)._load_outcome(record.occurrence_id)
    assert replace(loaded, images=()) == replace(outcome, images=())
    for first, second in zip(loaded.images, outcome.images, strict=True):
        np.testing.assert_array_equal(first.cell, second.cell)
        np.testing.assert_array_equal(first.positions, second.positions)


def test_cell_and_atomic_forces_are_enthalpy_derivatives_in_a_sheared_cell():
    atoms = bulk("Ar", "fcc", a=5.2, cubic=True)
    atoms.set_cell([[5.1, 0, 0], [0.3, 5.2, 0], [0.2, -0.1, 5.0]], scale_atoms=True)
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
    np.testing.assert_allclose(np.tril(filtered.get_forces()[-3:], -1), 0)


def test_variable_cell_preparation_removes_rotation_and_uses_periodic_mapping():
    candidate = volume_candidate()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    candidate.product.positions += candidate.product.cell[0]
    candidate.product.positions = candidate.product.positions @ rotation
    candidate.product.set_cell(candidate.product.cell @ rotation)
    reactant, product, _ = _prepare_endpoints(candidate, PathwayConfig(), variable_cell=True)
    np.testing.assert_allclose(product.cell, np.diag([6.2] * 3), atol=1e-14)
    np.testing.assert_allclose(
        product.get_scaled_positions(wrap=False)[0], [0.2 - 0.5 / 6, 0.2, 0.2]
    )
    assert reactant.get_volume() != product.get_volume()


def test_ssneb_interpolates_cell_and_atomic_coordinates_together():
    first, last, _ = _prepare_endpoints(volume_candidate(), PathwayConfig(), variable_cell=True)
    images = [first, first.copy(), last]
    band = SSNEB(images, pressure_GPa=0.0)
    band.interpolate()
    np.testing.assert_allclose(images[1].cell, (first.cell + last.cell) / 2)
    np.testing.assert_allclose(
        images[1].get_scaled_positions(wrap=False),
        (first.get_scaled_positions(wrap=False) + last.get_scaled_positions(wrap=False)) / 2,
    )


def test_npt_missing_stress_is_a_recorded_failure():
    from ase.calculators.harmonic import SpringCalculator

    @contextmanager
    def no_stress(_stage):
        yield SpringCalculator(volume_candidate().reactant.positions, k=1.0)

    outcome = refine_pathway(
        volume_candidate(),
        calculator_provider=no_stress,
        detector_config=BondDetectorConfig(),
        pressure_GPa=0.0,
    )
    assert outcome.status == "failed"
    assert "stress" in outcome.message
    assert outcome.method == "ssneb" and outcome.pressure_GPa == 0.0
    assert len(outcome.images) == 2
