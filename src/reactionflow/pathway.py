"""Endpoint relaxation, NVT NEB or constant-pressure SSNEB, and frequencies."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.geometry import find_mic
from ase.mep import NEB
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE
from ase.vibrations import Vibrations

from .candidates import ReactionCandidate
from .detection import BondDetectorConfig, assign_atom_ids, atom_ids
from .ssneb import SSNEB, cell_filter

CalculatorProvider = Callable[[str], AbstractContextManager[Calculator]]

# Total displacement of the saddle along its unit imaginary mode before each side is relaxed.
_CONNECTIVITY_DISPLACEMENT_A = 0.1


class _EndpointUnresolved(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PathwayConfig:
    """Small set of controls for endpoint relaxation, CI-NEB, and frequencies."""

    active_radius: float = 4.0
    relax_fmax: float = 0.05
    relax_steps: int = 2000
    images: int = 7
    neb_fmax: float = 0.05
    neb_steps: int = 2000
    ci_neb_steps: int = 2000
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
class ConnectivityCheck:
    """Where the saddle relaxes when displaced both ways along its primary imaginary mode.

    Each side is "reactant", "product", "other", "ambiguous" (a changed bond left inside the
    hysteresis gap), "not_converged", or "no_step" (the displaced copy already met the force
    tolerance, so the mode is too soft to test at this displacement).
    """

    status: str
    displacement_A: float
    sides: tuple[str, ...] = ()
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
    connectivity: ConnectivityCheck | None = None
    method: str = "neb"
    pressure_GPa: float | None = None
    enthalpies: tuple[float, ...] = ()
    volumes: tuple[float, ...] = ()

    @property
    def barrier_quantity(self) -> str:
        return "potential_energy" if self.pressure_GPa is None else "enthalpy"

    @property
    def converged(self) -> bool:
        return self.status in {"neb_converged", "ci_neb_converged"}


def _prepare_endpoints(
    candidate: ReactionCandidate,
    config: PathwayConfig,
    *,
    variable_cell: bool = False,
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
    if variable_cell:
        for endpoint in (reactant, product):
            if (
                not endpoint.pbc.all()
                or endpoint.cell.rank < 3
                or not np.isfinite(endpoint.cell).all()
                or np.linalg.det(endpoint.cell) <= 0
            ):
                raise _EndpointUnresolved("SSNEB requires fully periodic right-handed cells")
            # Remove rigid cell rotation without changing fractional coordinates
            # or internal geometry. These are copies, never the MD checkpoint.
            cell, rotation = endpoint.cell.standard_form()
            endpoint.positions = endpoint.positions @ rotation.T
            endpoint.set_cell(cell)
        reference_positions = product.get_scaled_positions(wrap=False) @ reactant.cell
        displacement = find_mic(
            reference_positions - reactant.positions, cell=reactant.cell, pbc=True
        )[0]
        product.set_scaled_positions(
            np.linalg.solve(reactant.cell.T, (reactant.positions + displacement).T).T
        )
    elif cells_differ:
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

    if not variable_cell:
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

    # Atoms away from the reaction start from the reactant's positions in both endpoints, so both
    # relax from the same surroundings. Nothing is constrained: every atom and, for NPT, the cell
    # relax.
    environment = sorted(set(range(len(reactant))) - active)
    if environment:
        product.positions[environment] = (
            reactant.get_scaled_positions(wrap=False)[environment] @ product.cell
            if variable_cell
            else reactant.positions[environment]
        )
    return reactant, product, tuple(sorted(active))


def _bonded_pairs(atoms: Atoms, detector_config: BondDetectorConfig) -> set[tuple[int, int]]:
    """Atom-ID pairs within their bond-formation distance anywhere in the cell."""

    ids = atom_ids(atoms)
    symbols = atoms.get_chemical_symbols()
    elements = set(symbols)
    cutoff = max(detector_config.threshold_for(a, b)[0] for a in elements for b in elements)
    first, second, distances = neighbor_list("ijd", atoms, cutoff)
    return {
        (min(ids[i], ids[j]), max(ids[i], ids[j]))
        for i, j, distance in zip(first, second, distances, strict=True)
        if i != j and distance <= detector_config.threshold_for(symbols[i], symbols[j])[0]
    }


def _bond_topology(
    candidate: ReactionCandidate,
    template: Atoms,
    detector_config: BondDetectorConfig,
) -> tuple[Callable[[Atoms], tuple[bool | None, ...]], tuple[tuple[bool, ...], tuple[bool, ...]]]:
    """Classify the pairs that define the candidate's reaction with the detector thresholds.

    Returns a function giving one state per pair for a structure (bonded, not bonded, or None
    inside the hysteresis gap) and the states of the candidate's reactant and product.
    """

    ids = atom_ids(template)
    indices = {atom_id: index for index, atom_id in enumerate(ids)}
    symbols = dict(zip(ids, template.get_chemical_symbols(), strict=True))
    region = tuple(candidate.atom_ids)
    changed_atoms = {
        atom_id for bond in candidate.reactant_bonds ^ candidate.product_bonds for atom_id in bond
    }
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
    thresholds = [detector_config.threshold_for(symbols[a], symbols[b]) for a, b in ordered_pairs]

    def states(atoms: Atoms) -> tuple[bool | None, ...]:
        result: list[bool | None] = []
        for (first, second), (form, breaking) in zip(ordered_pairs, thresholds, strict=True):
            distance = atoms.get_distance(indices[first], indices[second], mic=True)
            result.append(True if distance <= form else False if distance >= breaking else None)
        return tuple(result)

    expected = (
        tuple(bond in candidate.reactant_bonds for bond in ordered_pairs),
        tuple(bond in candidate.product_bonds for bond in ordered_pairs),
    )
    return states, expected


def _endpoint_status(
    candidate: ReactionCandidate,
    reactant: Atoms,
    product: Atoms,
    detector_config: BondDetectorConfig,
) -> tuple[str, str] | None:
    states, expected = _bond_topology(candidate, reactant, detector_config)
    actual = (states(reactant), states(product))
    if any(state is None for endpoint in actual for state in endpoint):
        return "unresolved", "a relaxed changed bond remains inside the hysteresis gap"
    if actual[0] == actual[1]:
        return "collapsed", "relaxed endpoints occupy the same changed-bond basin"
    if actual != expected:
        return "unresolved", "relaxed endpoints do not match the candidate topology"
    # With every atom free, relaxation can also form or break bonds away from the reaction; a
    # difference between the endpoints there would enter the path and the barrier.
    outside = (
        _bonded_pairs(reactant, detector_config) ^ _bonded_pairs(product, detector_config)
    ) - (candidate.reactant_bonds ^ candidate.product_bonds)
    if outside:
        pairs = ", ".join(f"{first}-{second}" for first, second in sorted(outside)[:5])
        return "unresolved", f"relaxation changed bonds outside the reaction: {pairs}"
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


def _minimize(
    atoms: Atoms,
    calculator: Calculator,
    config: PathwayConfig,
    pressure_GPa: float | None,
) -> tuple[bool, int]:
    """Relax like an endpoint (cell filter at pressure for NPT); return (converged, steps)."""

    atoms.calc = calculator
    try:
        target = (
            atoms if pressure_GPa is None else cell_filter(atoms, pressure_GPa, atoms.cell.copy())
        )
        optimizer = FIRE(target, logfile=None)
        converged = bool(optimizer.run(fmax=config.relax_fmax, steps=config.relax_steps))
        return converged, int(optimizer.nsteps)
    finally:
        atoms.calc = None


def _relax(
    atoms: Atoms,
    *,
    stage: str,
    calculator_provider: CalculatorProvider,
    config: PathwayConfig,
    pressure_GPa: float | None = None,
) -> bool:
    with calculator_provider(stage) as calculator:
        return _minimize(atoms, calculator, config, pressure_GPa)[0]


def _check_connectivity(
    saddle: Atoms,
    mode: tuple[tuple[float, float, float], ...],
    active_indices: tuple[int, ...],
    calculator: Calculator,
    config: PathwayConfig,
    pressure_GPa: float | None,
    candidate: ReactionCandidate,
    detector_config: BondDetectorConfig,
) -> ConnectivityCheck:
    """Relax the saddle displaced both ways along its imaginary mode and classify each side."""

    sides: list[str] = []
    try:
        states, (reactant, product) = _bond_topology(candidate, saddle, detector_config)
        for sign in (-1.0, 1.0):
            side = saddle.copy()
            side.positions[list(active_indices)] += (
                sign * _CONNECTIVITY_DISPLACEMENT_A * np.asarray(mode, dtype=float)
            )
            converged, steps = _minimize(side, calculator, config, pressure_GPa)
            if not converged:
                sides.append("not_converged")
                continue
            if steps == 0:
                sides.append("no_step")
                continue
            reached = states(side)
            sides.append(
                "ambiguous"
                if None in reached
                else "reactant"
                if reached == reactant
                else "product"
                if reached == product
                else "other"
            )
    except Exception as exc:
        return ConnectivityCheck(
            "failed", _CONNECTIVITY_DISPLACEMENT_A, tuple(sides), message=str(exc)
        )
    if sorted(sides) == ["product", "reactant"]:
        status = "connects_endpoints"
    elif {"not_converged", "no_step"} & set(sides):
        status = "inconclusive"
    else:
        status = "does_not_connect"
    return ConnectivityCheck(status, _CONNECTIVITY_DISPLACEMENT_A, tuple(sides))


def refine_pathway(
    candidate: ReactionCandidate,
    *,
    calculator_provider: CalculatorProvider,
    config: PathwayConfig | None = None,
    detector_config: BondDetectorConfig,
    pressure_GPa: float | None = None,
) -> PathwayOutcome:
    """Run NVT NEB, or SSNEB when a target pressure (including zero) is supplied."""

    if pressure_GPa is not None and (
        isinstance(pressure_GPa, bool) or not np.isfinite(pressure_GPa)
    ):
        raise ValueError("pressure_GPa must be a finite number or None")
    outcome = _refine_pathway(
        candidate,
        calculator_provider=calculator_provider,
        config=config,
        detector_config=detector_config,
        pressure_GPa=pressure_GPa,
    )
    return replace(
        outcome,
        method="neb" if pressure_GPa is None else "ssneb",
        pressure_GPa=pressure_GPa,
    )


def _refine_pathway(
    candidate: ReactionCandidate,
    *,
    calculator_provider: CalculatorProvider,
    config: PathwayConfig | None,
    detector_config: BondDetectorConfig,
    pressure_GPa: float | None,
) -> PathwayOutcome:

    if not candidate.resolved:
        return PathwayOutcome("unresolved", message="candidate topology was not resolved")
    if candidate.reactant_bonds == candidate.product_bonds:
        return PathwayOutcome("unresolved", message="candidate has no changed bonds")
    options = config or PathwayConfig()
    images: list[Atoms] = []
    try:
        reactant, product, active_indices = _prepare_endpoints(
            candidate, options, variable_cell=pressure_GPa is not None
        )
        images = [reactant, product]
        reactant_converged = _relax(
            reactant,
            stage="relax_reactant",
            calculator_provider=calculator_provider,
            config=options,
            pressure_GPa=pressure_GPa,
        )
        product_converged = _relax(
            product,
            stage="relax_product",
            calculator_provider=calculator_provider,
            config=options,
            pressure_GPa=pressure_GPa,
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
        if pressure_GPa is None:
            band = NEB(
                images,
                climb=False,
                allow_shared_calculator=True,
                method="improvedtangent",
            )
            band.interpolate(method="idpp", mic=True)
        else:
            band = SSNEB(images, pressure_GPa=pressure_GPa)
            band.interpolate()
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
                volumes = (
                    () if pressure_GPa is None else tuple(image.get_volume() for image in images)
                )
                enthalpies = (
                    ()
                    if pressure_GPa is None
                    else tuple(
                        energy + pressure_GPa * units.GPa * volume
                        for energy, volume in zip(energies, volumes, strict=True)
                    )
                )
                profile = energies if pressure_GPa is None else enthalpies
                frequency_validation = _validate_frequencies(
                    images,
                    profile,
                    active_indices,
                    calculator,
                    options,
                )
                connectivity = None
                if frequency_validation.primary_mode:
                    assert frequency_validation.transition_state_index is not None
                    connectivity = _check_connectivity(
                        images[frequency_validation.transition_state_index],
                        frequency_validation.primary_mode,
                        active_indices,
                        calculator,
                        options,
                        pressure_GPa,
                        candidate,
                        detector_config,
                    )
                return PathwayOutcome(
                    "ci_neb_converged",
                    barrier=max(profile) - profile[0],
                    energies=energies,
                    enthalpies=enthalpies,
                    volumes=volumes,
                    images=_snapshots(images),
                    frequency_validation=frequency_validation,
                    connectivity=connectivity,
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
    "ConnectivityCheck",
    "FrequencyValidation",
    "PathwayConfig",
    "PathwayOutcome",
    "refine_pathway",
]
