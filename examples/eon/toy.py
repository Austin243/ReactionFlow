"""Analytic CPU test potential; the helium labels do not represent helium chemistry."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class DoubleWell(Calculator):
    """Two minima at relative x = ±1 Å and one saddle at x = -0.2 Å.

    Atom 0 is the fixed anchor. In relative Cartesian coordinates, evaluated
    numerically in angstroms, the energy in eV is
    (x² - 1)² + 0.8 (x³/3 - x) + 8 (y² + z²).
    The transverse directions have positive curvature everywhere.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if len(self.atoms) != 2:
            raise ValueError("DoubleWell requires exactly two atoms")
        x, y, z = self.atoms.positions[1] - self.atoms.positions[0]
        energy = (x * x - 1) ** 2 + 0.8 * (x**3 / 3 - x) + 8 * (y * y + z * z)
        gradient = np.asarray([(x * x - 1) * (4 * x + 0.8), 16 * y, 16 * z])
        self.results = {
            "energy": float(energy),
            "forces": np.asarray([gradient, -gradient]),
        }
