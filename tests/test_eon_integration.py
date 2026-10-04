"""Real EON process searches: REACTIONFLOW_TEST_EON=1 pytest -q this_file.

Requires the optional reactionflow[eon] installation. An explicitly enabled run fails if the
backend is absent.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from reactionflow.detection import assign_atom_ids
from reactionflow.search.eon import EONBackend, EONSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("REACTIONFLOW_TEST_EON") != "1",
    reason="set REACTIONFLOW_TEST_EON=1 with reactionflow[eon] installed",
)


def well(u: float) -> tuple[float, float]:
    """(u² - 1)² + 0.8 (u³/3 - u) and its slope: minima at u = -1 and 1, saddle at -0.2."""

    return (u * u - 1) ** 2 + 0.8 * (u**3 / 3 - u), (u * u - 1) * (4 * u + 0.8)


class AnchoredWell(Calculator):
    """Atom 1 relative to the fixed atom 0: the double well along x, held in y and z."""

    implemented_properties = ("energy", "forces")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert atoms.info["charge"] == 0 and atoms.info["spin"] == 1
        x, y, z = atoms.positions[1] - atoms.positions[0]
        energy, slope = well(x)
        gradient = np.array([slope, 16 * y, 16 * z])
        self.results = {
            "energy": energy + 8 * (y * y + z * z),
            "forces": np.array([gradient, -gradient]),
        }


class FreeWell(Calculator):
    """Two free atoms whose energy depends only on their distance r, with u = r - 2 Å."""

    implemented_properties = ("energy", "forces")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        bond = atoms.positions[1] - atoms.positions[0]
        r = float(np.linalg.norm(bond))
        energy, slope = well(r - 2)
        force = slope * bond / r
        self.results = {"energy": energy, "forces": np.array([force, -force])}


def frequency(curvature: float, mass: float) -> float:
    """Frequency in 1/s of a `mass` u on a spring of `curvature` eV/Å²."""

    return math.sqrt(curvature / mass * 9.648533e27) / (2 * math.pi)


def first_good(backend, start):
    results = [backend.search(start, seed=seed) for seed in range(8)]
    good = [result for result in results if result.status == "good"]
    assert good, [result.message for result in results]
    return good[0]


def test_real_eon_finds_analytic_barriers_and_harmonic_prefactors():
    atoms = assign_atom_ids(Atoms("He2", positions=[[5, 5, 5], [4.1, 5.05, 5]], cell=[12] * 3))
    atoms.pbc = True
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.info.update(charge=0, spin=1)
    backend = EONBackend(AnchoredWell(), EONSettings(), temperature_K=300)
    start, energy = backend.relax(atoms)
    assert energy == pytest.approx(0.8 * 2 / 3, abs=1e-4)
    result = first_good(backend, start)
    assert result.saddle_energy - energy == pytest.approx(0.5461333, abs=1e-3)
    assert result.saddle_energy - result.product_energy == pytest.approx(1.6128, abs=1e-3)
    assert result.prefactor == pytest.approx(frequency(6.4, 4.002602), rel=0.02)
    assert result.reverse_prefactor == pytest.approx(frequency(9.6, 4.002602), rel=0.02)
    assert result.product.positions[1, 0] - 5 == pytest.approx(1, abs=0.01)
    assert result.saddle.info == atoms.info
    np.testing.assert_array_equal(result.product.arrays["atom_id"], atoms.arrays["atom_id"])


def test_real_eon_treats_rigid_rotation_of_a_free_molecule_as_no_change():
    atoms = assign_atom_ids(Atoms("He2", positions=[[5, 5, 5], [6.0, 5.3, 5]], cell=[12] * 3))
    backend = EONBackend(FreeWell(), EONSettings(prefactor=1e13), temperature_K=300)
    start, energy = backend.relax(atoms)
    result = first_good(backend, start)
    assert result.saddle_energy - energy == pytest.approx(0.5461333, abs=1e-3)
    product, product_energy = result.product, result.product_energy
    assert product.get_distance(0, 1) == pytest.approx(3, abs=0.01)
    turned = product.copy()
    turned.rotate(90, "z", center="COM")
    turned.translate([0.5, 0, 0])
    assert backend.same(product, product_energy, turned, product_energy)
    assert not backend.same(start, energy, product, product_energy)
