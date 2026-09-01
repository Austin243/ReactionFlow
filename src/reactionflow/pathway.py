"""Fixed-cell endpoint relaxation, serial climbing-image NEB, and frequencies."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.mep import NEB
from ase.optimize import FIRE
from ase.vibrations import Vibrations

from .candidates import ReactionCandidate
from .detection import BondDetectorConfig, assign_atom_ids, atom_ids

CalculatorProvider = Callable[[str], AbstractContextManager[Calculator]]


class _EndpointUnresolved(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PathwayConfig:
    """Small set of controls for endpoint relaxation, CI-NEB, and frequencies."""

    active_radius: float = 4.0
    relax_fmax: float = 0.05
    relax_steps: int = 500
    images: int = 7
    neb_fmax: float = 0.05
    neb_steps: int = 500
    ci_neb_steps: int = 500
    max_volume_change_fraction: float = 0.05
    max_cell_strain: float = 0.05
    frequency_delta: float = 0.01
    imaginary_frequency_cutoff_cm1: float = 50.0

    def __post_init__(self) -> None:
        if (
            self.active_radius <= 0
            or self.relax_fmax <= 0
            or self.neb_fmax <= 0
            or not np.isfinite(self.frequency_delta)
            or self.frequency_delta <= 0
        ):
            raise ValueError("radii, force tolerances, and frequency delta must be positive")
        if self.relax_steps < 1 or self.images < 3 or self.neb_steps < 1 or self.ci_neb_steps < 1:
            raise ValueError("step counts must be positive and NEB needs at least three images")
        if (
            self.max_volume_change_fraction < 0
            or self.max_cell_strain < 0
            or not np.isfinite(self.imaginary_frequency_cutoff_cm1)
            or self.imaginary_frequency_cutoff_cm1 < 0
        ):
            raise ValueError("cell-change limits and the imaginary cutoff must be non-negative")


@dataclass(frozen=True, slots=True)
class FrequencyValidation:
    """Constrained active-region frequency result for one climbing image."""

    status: str
    transition_state_index: int | None
    active_atom_ids: tuple[int, ...]
    displacement_A: float
    imaginary_cutoff_cm1: float
    scope: str = "active_atoms_fixed_environment"
    active_max_force_eV_A: float | None = None
    frequencies_cm1: tuple[float, ...] = ()
    imaginary_mode_indices: tuple[int, ...] = ()
    primary_mode_index: int | None = None
    primary_mode: tuple[tuple[float, float, float], ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class PathwayOutcome:
    """Bounded result with calculator-free snapshots of the attempted path."""

    status: str
    barrier: float | None = None
    energies: tuple[float, ...] = ()
    images: tuple[Atoms, ...] = ()
    message: str = ""
    frequency_validation: FrequencyValidation | None = None

    @property
    def converged(self) -> bool:
        return self.status in {"neb_converged", "ci_neb_converged"}


def _prepare_endpoints(
    candidate: ReactionCandidate,
    config: PathwayConfig,
) -> tuple[Atoms, Atoms, tuple[int, ...]]:
    reactant = assign_atom_ids(candidate.reactant.copy())
    product = assign_atom_ids(candidate.product.copy())
    reactant.calc = product.calc = None
    reactant.set_constraint()
    product.set_constraint()

    first_ids = atom_ids(reactant)
    second_ids = atom_ids(product)
    if set(first_ids) != set(second_ids):
        raise ValueError("reactant and product atom IDs differ")
    order = {atom_id: index for index, atom_id in enumerate(second_ids)}
    product = product[[order[atom_id] for atom_id in first_ids]]
    reactant.info.pop("atom_ids", None)
    product.info.pop("atom_ids", None)
    if reactant.get_chemical_symbols() != product.get_chemical_symbols():
        raise ValueError("reactant and product elements differ")
    if reactant.pbc.tolist() != product.pbc.tolist():
        raise _EndpointUnresolved("reactant and product periodicity differs")
    cells_differ = not np.array_equal(reactant.cell, product.cell)
    if cells_differ:
        if not reactant.pbc.all() or reactant.cell.rank < 3 or product.cell.rank < 3:
            raise _EndpointUnresolved("changed cells require fully periodic nonsingular endpoints")
        reactant_volume = abs(float(np.linalg.det(reactant.cell)))
        product_volume = abs(float(np.linalg.det(product.cell)))
        try:
            deformation = np.linalg.solve(reactant.cell, product.cell)
            strain = float(np.max(np.abs(np.linalg.svd(deformation, compute_uv=False) - 1)))
        except np.linalg.LinAlgError:
            raise _EndpointUnresolved("endpoint cell is singular") from None
        volume_change = abs(product_volume / reactant_volume - 1)
        if volume_change > config.max_volume_change_fraction or strain > config.max_cell_strain:
            raise _EndpointUnresolved("endpoint cell change exceeds pathway limits")
        product.set_cell(reactant.cell, scale_atoms=True)

    displacement = product.positions - reactant.positions
    if reactant.pbc.any():
        displacement = find_mic(displacement, cell=reactant.cell, pbc=reactant.pbc)[0]
    product.positions = reactant.positions + displacement

    indices = {atom_id: index for index, atom_id in enumerate(first_ids)}
    if not set(candidate.atom_ids) <= indices.keys():
        raise ValueError("candidate atom IDs are missing from its endpoints")
    changed_atoms = {
        atom_id for bond in candidate.reactant_bonds ^ candidate.product_bonds for atom_id in bond
    }
    centers = {indices[atom_id] for atom_id in changed_atoms}
    active = set(centers)
    for center in centers:
        for endpoint in (reactant, product):
            distances = endpoint.get_distances(center, range(len(endpoint)), mic=True)
            active.update(
                index
                for index, distance in enumerate(distances)
                if distance <= config.active_radius
            )

    frozen = sorted(set(range(len(reactant))) - active)
    if frozen:
        product.positions[frozen] = reactant.positions[frozen]
        constraint = FixAtoms(indices=frozen)
        reactant.set_constraint(constraint)
        product.set_constraint(constraint.copy())
    return reactant, product, tuple(sorted(active))


def _endpoint_status(
    candidate: ReactionCandidate,
    reactant: Atoms,
    product: Atoms,
    detector_config: BondDetectorConfig,
) -> tuple[str, str] | None:
    changed = candidate.reactant_bonds ^ candidate.product_bonds
    ids = atom_ids(reactant)
    indices = {atom_id: index for index, atom_id in enumerate(ids)}
    symbols = dict(zip(ids, reactant.get_chemical_symbols(), strict=True))
    region = tuple(candidate.atom_ids)
    changed_atoms = {atom_id for bond in changed for atom_id in bond}
    pairs = {
        tuple(sorted((first, second)))
        for index, first in enumerate(region)
        for second in region[index + 1 :]
    }
    pairs.update(
        tuple(sorted((changed_atom, atom_id)))
        for changed_atom in changed_atoms
        for atom_id in ids
        if changed_atom != atom_id
    )
    ordered_pairs = sorted(pairs)
    actual: list[tuple[bool | None, ...]] = []
    for endpoint in (reactant, product):
        states: list[bool | None] = []
        for first, second in ordered_pairs:
            form, breaking = detector_config.threshold_for(symbols[first], symbols[second])
            distance = endpoint.get_distance(indices[first], indices[second], mic=True)
            states.append(True if distance <= form else False if distance >= breaking else None)
        actual.append(tuple(states))

    if any(state is None for endpoint in actual for state in endpoint):
        return "unresolved", "a relaxed changed bond remains inside the hysteresis gap"
    if actual[0] == actual[1]:
        return "collapsed", "relaxed endpoints occupy the same changed-bond basin"
    expected = (
        tuple(bond in candidate.reactant_bonds for bond in ordered_pairs),
        tuple(bond in candidate.product_bonds for bond in ordered_pairs),
    )
    if tuple(actual) != expected:
        return "unresolved", "relaxed endpoints do not match the candidate topology"
    return None


def _snapshots(images: list[Atoms]) -> tuple[Atoms, ...]:
    return tuple(image.copy() for image in images)


def _classify_frequencies(
    frequencies: np.ndarray,
    cutoff_cm1: float,
) -> tuple[tuple[float, ...], tuple[int, ...], str]:
    values = np.asarray(frequencies, dtype=complex).reshape(-1)
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError("frequency calculation produced non-finite values")

    signed = tuple(
        -float(abs(value.imag)) if value.imag != 0 else float(value.real) for value in values
    )
    imaginary = tuple(
        index
        for index, value in enumerate(values)
        if abs(value.imag) > 0 and abs(value.imag) >= cutoff_cm1
    )
    status = (
        "zero_imaginary_modes"
        if not imaginary
        else "one_imaginary_mode"
        if len(imaginary) == 1
        else "multiple_imaginary_modes"
    )
    return signed, imaginary, status


def _validate_frequencies(
    images: list[Atoms],
    energies: tuple[float, ...],
    active_indices: tuple[int, ...],
    calculator: Calculator,
    config: PathwayConfig,
) -> FrequencyValidation:
    transition_state_index: int | None = None
    transition_state: Atoms | None = None
    active_atom_ids: tuple[int, ...] = ()
    active_max_force: float | None = None
    try:
        transition_state_index = 1 + int(np.argmax(energies[1:-1]))
        transition_state = images[transition_state_index].copy()
        transition_state.calc = calculator
        active_atom_ids = tuple(atom_ids(transition_state)[index] for index in active_indices)
        forces = transition_state.get_forces(apply_constraint=False)[list(active_indices)]
        active_max_force = float(np.linalg.norm(forces, axis=1).max())
        if not np.isfinite(active_max_force):
            raise ValueError("transition-state forces are non-finite")

        with TemporaryDirectory(prefix="reactionflow-frequencies-") as temporary:
            vibrations = Vibrations(
                transition_state,
                indices=active_indices,
                name=str(Path(temporary) / "vibrations"),
                delta=config.frequency_delta,
                nfree=2,
            )
            vibrations.run()
            vibration_data = vibrations.get_vibrations(
                method="standard",
                direction="central",
            )
            raw_frequencies = vibration_data.get_frequencies()
            frequencies, imaginary_indices, status = _classify_frequencies(
                raw_frequencies,
                config.imaginary_frequency_cutoff_cm1,
            )

            primary_index: int | None = None
            primary_mode: tuple[tuple[float, float, float], ...] = ()
            if imaginary_indices:
                primary_index = min(imaginary_indices, key=frequencies.__getitem__)
                vector = np.asarray(
                    vibration_data.get_modes(all_atoms=False)[primary_index],
                    dtype=float,
                )
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
                    raise ValueError("primary imaginary mode is not finite and non-zero")
                vector /= norm
                primary_mode = tuple(
                    tuple(float(component) for component in displacement) for displacement in vector
                )

        return FrequencyValidation(
            status=status,
            transition_state_index=transition_state_index,
            active_atom_ids=active_atom_ids,
            displacement_A=config.frequency_delta,
            imaginary_cutoff_cm1=config.imaginary_frequency_cutoff_cm1,
            active_max_force_eV_A=active_max_force,
            frequencies_cm1=frequencies,
            imaginary_mode_indices=imaginary_indices,
            primary_mode_index=primary_index,
            primary_mode=primary_mode,
        )
    except Exception as exc:
        return FrequencyValidation(
            status="failed",
            transition_state_index=transition_state_index,
            active_atom_ids=active_atom_ids,
            displacement_A=config.frequency_delta,
            imaginary_cutoff_cm1=config.imaginary_frequency_cutoff_cm1,
            active_max_force_eV_A=active_max_force,
            message=str(exc),
        )
    finally:
        if transition_state is not None:
            transition_state.calc = None


def _relax(
    atoms: Atoms,
    *,
    stage: str,
    calculator_provider: CalculatorProvider,
    config: PathwayConfig,
) -> bool:
    with calculator_provider(stage) as calculator:
        atoms.calc = calculator
        try:
            return bool(
                FIRE(atoms, logfile=None).run(
                    fmax=config.relax_fmax,
                    steps=config.relax_steps,
                )
            )
        finally:
            atoms.calc = None


def refine_pathway(
    candidate: ReactionCandidate,
    *,
    calculator_provider: CalculatorProvider,
    config: PathwayConfig | None = None,
    detector_config: BondDetectorConfig,
) -> PathwayOutcome:
    """Relax one candidate's endpoints and run a serial fixed-cell CI-NEB."""

    if not candidate.resolved:
        return PathwayOutcome("unresolved", message="candidate topology was not resolved")
    if candidate.reactant_bonds == candidate.product_bonds:
        return PathwayOutcome("unresolved", message="candidate has no changed bonds")
    options = config or PathwayConfig()
    images: list[Atoms] = []
    try:
        reactant, product, active_indices = _prepare_endpoints(candidate, options)
        images = [reactant, product]
        reactant_converged = _relax(
            reactant,
            stage="relax_reactant",
            calculator_provider=calculator_provider,
            config=options,
        )
        product_converged = _relax(
            product,
            stage="relax_product",
            calculator_provider=calculator_provider,
            config=options,
        )
        if not reactant_converged or not product_converged:
            return PathwayOutcome(
                "relaxation_failed",
                images=_snapshots(images),
                message="endpoint relaxation did not converge",
            )

        endpoint_status = _endpoint_status(candidate, reactant, product, detector_config)
        if endpoint_status is not None:
            status, message = endpoint_status
            return PathwayOutcome(status, images=_snapshots(images), message=message)

        images = [reactant]
        images.extend(reactant.copy() for _ in range(options.images - 2))
        images.append(product)
        band = NEB(
            images,
            climb=False,
            allow_shared_calculator=True,
            method="improvedtangent",
        )
        band.interpolate(method="idpp", mic=True)
        with calculator_provider("neb") as calculator:
            for image in images:
                image.calc = calculator
            try:
                neb_converged = bool(
                    FIRE(band, logfile=None).run(
                        fmax=options.neb_fmax,
                        steps=options.neb_steps,
                    )
                )
                if not neb_converged:
                    return PathwayOutcome(
                        "neb_failed",
                        images=_snapshots(images),
                        message="initial NEB did not converge",
                    )
                band.climb = True
                ci_neb_converged = bool(
                    FIRE(band, logfile=None).run(
                        fmax=options.neb_fmax,
                        steps=options.ci_neb_steps,
                    )
                )
                if not ci_neb_converged:
                    return PathwayOutcome(
                        "ci_neb_failed",
                        images=_snapshots(images),
                        message="climbing-image NEB did not converge",
                    )
                energies = tuple(float(image.get_potential_energy()) for image in images)
                frequency_validation = _validate_frequencies(
                    images,
                    energies,
                    active_indices,
                    calculator,
                    options,
                )
                return PathwayOutcome(
                    "ci_neb_converged",
                    barrier=max(energies) - energies[0],
                    energies=energies,
                    images=_snapshots(images),
                    frequency_validation=frequency_validation,
                )
            finally:
                for image in images:
                    image.calc = None
    except _EndpointUnresolved as exc:
        return PathwayOutcome("unresolved", images=_snapshots(images), message=str(exc))
    except Exception as exc:
        return PathwayOutcome("failed", images=_snapshots(images), message=str(exc))


__all__ = [
    "CalculatorProvider",
    "FrequencyValidation",
    "PathwayConfig",
    "PathwayOutcome",
    "refine_pathway",
]
