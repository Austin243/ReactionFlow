"""Solid-state NEB in a common atomic/deformation coordinate space.

ASE's improved-tangent NEB supplies the springs and climbing-image projection;
UnitCellFilter supplies the energy-consistent gradient of E + P V with respect
to reference-cell atomic coordinates and the deformation gradient. Both parts
of that gradient participate in the same NEB projection.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units
from ase.cell import Cell
from ase.filters import UnitCellFilter
from ase.mep import NEB


class _TriangularCellFilter(UnitCellFilter):
    """Constrain the deformation gradient itself to six independent entries."""

    def set_positions(self, positions, **kwargs):
        positions = positions.copy()
        positions[-3:] = np.triu(positions[-3:])
        super().set_positions(positions, **kwargs)

    def get_forces(self, **kwargs):
        forces = super().get_forces(**kwargs)
        forces[-3:] = np.triu(forces[-3:])
        return forces


def cell_filter(atoms: Atoms, pressure_GPa: float, reference_cell: Cell) -> UnitCellFilter:
    """Use one reference and length scale for all images, including spectators."""
    # J = sqrt(N) * (V/N)^(1/3), the usual SSNEB atomic/cell length scale.
    scale = len(atoms) ** (1 / 6) * reference_cell.volume ** (1 / 3)
    # Lower-triangular cells have upper-triangular deformation gradients.
    # All six independent components permit volume and shape changes, while
    # removing rigid cell rotation from the path coordinates.
    return _TriangularCellFilter(
        atoms,
        orig_cell=reference_cell,
        cell_factor=scale,
        scalar_pressure=pressure_GPa * units.GPa,
    )


class _SolidStateImage:
    """Expose a cell filter as an NEB image in nonperiodic generalized space.

    Physical atoms retain their own periodic cells. Periodic wrapping is only
    done during endpoint preparation, never on the deformation coordinates.
    """

    pbc = np.zeros(3, dtype=bool)
    cell = Cell.new()

    def __init__(self, filtered: UnitCellFilter) -> None:
        self.filtered = filtered

    def __len__(self) -> int:
        return len(self.filtered)

    @property
    def calc(self):
        return self.filtered.atoms.calc

    def get_atomic_numbers(self):
        return np.concatenate([self.filtered.atoms.numbers, np.zeros(3, dtype=int)])

    def get_positions(self):
        return self.filtered.get_positions()

    def set_positions(self, positions, **kwargs):
        self.filtered.set_positions(positions, **kwargs)

    def get_forces(self, **kwargs):
        return self.filtered.get_forces(**kwargs)

    def get_potential_energy(self):
        return self.filtered.get_potential_energy(force_consistent=False)


class SSNEB(NEB):
    """Serial variable-cell NEB at a prescribed hydrostatic pressure."""

    def __init__(self, images: list[Atoms], *, pressure_GPa: float) -> None:
        self.physical_images = images
        reference = images[0].cell.copy()
        super().__init__(
            [_SolidStateImage(cell_filter(image, pressure_GPa, reference)) for image in images],
            climb=False,
            allow_shared_calculator=True,
            method="improvedtangent",
            remove_rotation_and_translation=False,
        )

    def interpolate(self):
        # Linear interpolation in reference atomic/deformation coordinates.
        # ASE IDPP assumes a common physical cell and is not applied here.
        super().interpolate(method="linear", mic=False, apply_constraint=True)

    def iterimages(self):
        yield from self.physical_images
