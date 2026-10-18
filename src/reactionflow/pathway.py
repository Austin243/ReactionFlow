"""Endpoint relaxation, NVT NEB or constant-pressure SSNEB, and the saddle check."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from itertools import product
from numbers import Integral
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.mep import NEB
from ase.optimize import FIRE

from .candidates import ReactionCandidate
from .detection import Bond, BondDetectorConfig, atom_ids, canonical_copy, classify_bonds
from .ssneb import SSNEB, cell_filter

CalculatorProvider = Callable[[str], AbstractContextManager[Calculator]]

# The saddle's Hessian covers the atoms that move more than _ACTIVE_MOTION_A between the images
# either side of it, and at least _MIN_ACTIVE_ATOMS atoms. Each side of the connectivity check
# starts from the saddle moved along its lowest mode until the atom that moves most has moved
# _CONNECTIVITY_DISPLACEMENT_A.
_ACTIVE_MOTION_A = 0.05
_MIN_ACTIVE_ATOMS = 6
_CONNECTIVITY_DISPLACEMENT_A = 0.1

# A converged CI-NEB band ends as one of these, and only saddle_validated carries a barrier.
CONVERGED_STATUSES = frozenset(
    {"saddle_validated", "no_saddle", "saddle_rejected", "saddle_unchecked"}
)

# Settings of the frequency check that the saddle check replaced. Campaign and state files that
# still name them open, and the settings are ignored.
_RETIRED_KEYS = frozenset({"active_radius", "imaginary_frequency_cutoff_cm1"})


class _EndpointUnresolved(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PathwayConfig:
    """Small set of controls for endpoint relaxation, CI-NEB, and the saddle check.

    ``reactant_lookback`` is how many observations before the tracker's reactant frame refinement
    may go back for a frame with every changed bond on its reactant side, and ``split_steps``
    refines an event that changes several bonds one step per intermediate it passes through.
    """

    relax_fmax: float = 0.05
    relax_steps: int = 2000
    images: int = 7
    neb_fmax: float = 0.05
    neb_steps: int = 2000
    ci_neb_steps: int = 2000
    max_volume_change_fraction: float = 0.05
    max_cell_strain: float = 0.05
    frequency_delta: float = 0.01
    curvature_cutoff_eV_A2: float = 0.05
    reactant_lookback: int = 10
    split_steps: bool = True

    def __post_init__(self) -> None:
        if (
            self.relax_fmax <= 0
            or self.neb_fmax <= 0
            or not np.isfinite(self.frequency_delta)
            or self.frequency_delta <= 0
        ):
            raise ValueError("force tolerances and frequency delta must be positive")
        counts = (self.relax_steps, self.images, self.neb_steps, self.ci_neb_steps)
        if any(isinstance(count, bool) or not isinstance(count, Integral) for count in counts):
            raise ValueError("step counts and images must be whole numbers")
        if self.relax_steps < 1 or self.images < 3 or self.neb_steps < 1 or self.ci_neb_steps < 1:
            raise ValueError("step counts must be positive and NEB needs at least three images")
        if (
            self.max_volume_change_fraction < 0
            or self.max_cell_strain < 0
            or not np.isfinite(self.curvature_cutoff_eV_A2)
            or self.curvature_cutoff_eV_A2 < 0
        ):
            raise ValueError("cell-change limits and the curvature cutoff must be non-negative")
        lookback = self.reactant_lookback
        if isinstance(lookback, bool) or not isinstance(lookback, Integral) or lookback < 0:
            raise ValueError("reactant_lookback must be a non-negative whole number")
        if not isinstance(self.split_steps, bool):
            raise ValueError("split_steps must be true or false")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PathwayConfig:
        return cls(**{key: item for key, item in value.items() if key not in _RETIRED_KEYS})


@dataclass(frozen=True, slots=True)
class FrequencyValidation:
    """Fixed-cell Hessian of the saddle image over the atoms that move through it.

    Modes are the eigenvalues of the Cartesian Hessian below -curvature_cutoff_eV_A2.
    frequencies_cm1 comes from the same Hessian with mass weighting, with imaginary frequencies
    as negative values; a Hessian has the same number of negative eigenvalues either way.
    """

    status: str
    transition_state_index: int | None
    active_atom_ids: tuple[int, ...]
    displacement_A: float
    curvature_cutoff_eV_A2: float
    scope: str = "active_atoms_fixed_environment"
    active_max_force_eV_A: float | None = None
    eigenvalues_eV_A2: tuple[float, ...] = ()
    frequencies_cm1: tuple[float, ...] = ()
    tangent_overlap: float | None = None
    primary_mode: tuple[tuple[float, float, float], ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class ConnectivityCheck:
    """Where the saddle relaxes when moved both ways along its lowest mode.

    displacement_A is how far the atom that moves most is moved. Each side is "reactant",
    "product", "other", "ambiguous" (a pair left inside the hysteresis gap), "not_converged", or
    "no_step" (the moved copy already met the force tolerance, so the mode is too soft to test at
    this displacement). When both ends of the band have the same bonds, a side reaches the end
    nearer to it.
    """

    status: str
    displacement_A: float
    sides: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class PathwayOutcome:
    """Bounded result with calculator-free snapshots of the attempted path.

    ``frames`` are the saved frames the reactant was relaxed from and the product built from. An
    event refined in steps is ``stepwise``, with no band of its own and one result per step.
    """

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
    frames: tuple[int, int] | None = None
    steps: tuple[PathwayStep, ...] = ()

    @property
    def barrier_quantity(self) -> str:
        return "potential_energy" if self.pressure_GPa is None else "enthalpy"

    @property
    def converged(self) -> bool:
        return self.status in CONVERGED_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """The outcome without its images, as result.json and check.json store it."""

        return {
            "status": self.status,
            "barrier_eV": self.barrier,
            "energies_eV": list(self.energies),
            "method": self.method,
            "pressure_GPa": self.pressure_GPa,
            "barrier_quantity": self.barrier_quantity,
            "enthalpies_eV": list(self.enthalpies),
            "volumes_A3": list(self.volumes),
            "message": self.message,
            "frequency_validation": (
                None if self.frequency_validation is None else asdict(self.frequency_validation)
            ),
            "connectivity": None if self.connectivity is None else asdict(self.connectivity),
            "frames": None if self.frames is None else list(self.frames),
            "steps": [
                {"step": number, "status": step.outcome.status, "barrier_eV": step.outcome.barrier}
                for number, step in enumerate(self.steps, 1)
            ],
        }


@dataclass(frozen=True, slots=True)
class PathwayStep:
    """One step of an event refined through its intermediates, with the bonds of its two ends
    inside the event's region."""

    outcome: PathwayOutcome
    reactant_bonds: frozenset[Bond]
    product_bonds: frozenset[Bond]


def _standard_cell(atoms: Atoms) -> None:
    if (
        not atoms.pbc.all()
        or atoms.cell.rank < 3
        or not np.isfinite(atoms.cell).all()
        or np.linalg.det(atoms.cell) <= 0
    ):
        raise _EndpointUnresolved("SSNEB requires fully periodic right-handed cells")
    # Remove rigid cell rotation without changing fractional coordinates
    # or internal geometry. These are copies, never the MD checkpoint.
    cell, rotation = atoms.cell.standard_form()
    atoms.positions = atoms.positions @ rotation.T
    atoms.set_cell(cell)


def _prepare_frames(
    frames: Sequence[Atoms],
    config: PathwayConfig,
    *,
    variable_cell: bool = False,
) -> list[Atoms]:
    """Constraint-free copies, each in the first frame's atom order and placed next to it."""

    reactant = canonical_copy(frames[0])
    reactant.set_constraint()
    first_ids = atom_ids(reactant)
    if variable_cell:
        _standard_cell(reactant)
    prepared = [reactant]
    for frame in frames[1:]:
        product = canonical_copy(frame)
        product.set_constraint()
        second_ids = atom_ids(product)
        if set(first_ids) != set(second_ids):
            raise ValueError("reactant and product atom IDs differ")
        order = {atom_id: index for index, atom_id in enumerate(second_ids)}
        product = product[[order[atom_id] for atom_id in first_ids]]
        if reactant.get_chemical_symbols() != product.get_chemical_symbols():
            raise ValueError("reactant and product elements differ")
        if reactant.pbc.tolist() != product.pbc.tolist():
            raise _EndpointUnresolved("reactant and product periodicity differs")
        cells_differ = not np.array_equal(reactant.cell, product.cell)
        if variable_cell:
            _standard_cell(product)
            reference_positions = product.get_scaled_positions(wrap=False) @ reactant.cell
            displacement = find_mic(
                reference_positions - reactant.positions, cell=reactant.cell, pbc=True
            )[0]
            product.set_scaled_positions(
                np.linalg.solve(reactant.cell.T, (reactant.positions + displacement).T).T
            )
        elif cells_differ:
            if not reactant.pbc.all() or reactant.cell.rank < 3 or product.cell.rank < 3:
                raise _EndpointUnresolved(
                    "changed cells require fully periodic nonsingular endpoints"
                )
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
        prepared.append(product)
    return prepared


def _local(candidate: ReactionCandidate) -> Callable[[Collection[Bond]], set[Bond]]:
    """Pairs inside the reacting region, plus any pair involving an atom of a changed bond."""

    region = set(candidate.atom_ids)
    changed_atoms = {
        atom_id for bond in candidate.reactant_bonds ^ candidate.product_bonds for atom_id in bond
    }
    return lambda bonds: {
        bond for bond in bonds if region.issuperset(bond) or changed_atoms.intersection(bond)
    }


def _on_reactant_side(
    atoms: Atoms, candidate: ReactionCandidate, detector_config: BondDetectorConfig
) -> bool:
    """Whether each bond the candidate breaks is within its formation distance, and each bond it
    forms at or beyond its break distance."""

    bonded, gap = classify_bonds(atoms, detector_config)
    return all(
        bond in bonded if bond in candidate.reactant_bonds else bond not in bonded | gap
        for bond in candidate.reactant_bonds ^ candidate.product_bonds
    )


def _made(
    atoms: Atoms, candidate: ReactionCandidate, detector_config: BondDetectorConfig
) -> frozenset[Bond] | None:
    """The candidate's bond changes that a relaxed structure holds, or None when it also holds
    another change near them or a pair there inside the hysteresis gap."""

    local = _local(candidate)
    bonded, gap = classify_bonds(atoms, detector_config)
    made = frozenset(local(bonded) ^ candidate.reactant_bonds)
    if local(gap) or not made <= candidate.reactant_bonds ^ candidate.product_bonds:
        return None
    return made


def _matches_outside(
    relaxed: Atoms,
    frame: Atoms,
    changed: Collection[Bond],
    detector_config: BondDetectorConfig,
) -> bool:
    """Whether a relaxed structure has the frame's bonds everywhere but the changed ones.

    Only a pair on opposite sides of the hysteresis gap in the two counts as a difference.
    """

    bonded, gap = classify_bonds(relaxed, detector_config)
    expected, unknown = classify_bonds(frame, detector_config)
    return not ((bonded ^ expected) - gap - unknown).difference(changed)


def _endpoint_status(
    candidate: ReactionCandidate,
    reactant: Atoms,
    product: Atoms,
    detector_config: BondDetectorConfig,
) -> tuple[str, str] | None:
    """Check the relaxed endpoints against the candidate with the detector's thresholds."""

    changed_bonds = candidate.reactant_bonds ^ candidate.product_bonds
    local = _local(candidate)
    (reactant_bonds, reactant_gap), (product_bonds, product_gap) = (
        classify_bonds(endpoint, detector_config) for endpoint in (reactant, product)
    )
    if local(reactant_gap | product_gap):
        return "unresolved", "a relaxed changed bond remains inside the hysteresis gap"
    if local(reactant_bonds) == local(product_bonds):
        return "collapsed", "relaxed endpoints occupy the same changed-bond basin"
    if (local(reactant_bonds), local(product_bonds)) != (
        local(candidate.reactant_bonds),
        local(candidate.product_bonds),
    ):
        return "unresolved", "relaxed endpoints do not match the candidate topology"
    # With every atom free, relaxation can also form or break bonds away from the reaction; a
    # difference between the endpoints there would enter the path and the barrier.
    outside = (reactant_bonds ^ product_bonds) - changed_bonds
    if outside:
        pairs = ", ".join(f"{first}-{second}" for first, second in sorted(outside)[:5])
        return "unresolved", f"relaxation changed bonds outside the reaction: {pairs}"
    return None


def _snapshots(images: list[Atoms]) -> tuple[Atoms, ...]:
    return tuple(image.copy() for image in images)


def _hessian(
    atoms: Atoms, active: Sequence[int], calculator: Calculator, step: float
) -> np.ndarray:
    """Symmetrized central-difference Hessian in eV/Å² over the active atoms, cell fixed."""

    columns = []
    for atom, axis in product(active, range(3)):
        gradients = []
        for sign in (1.0, -1.0):
            trial = atoms.copy()
            trial.positions[atom, axis] += sign * step
            trial.calc = calculator
            gradients.append(-trial.get_forces(apply_constraint=False)[active].reshape(-1))
        columns.append((gradients[0] - gradients[1]) / (2 * step))
    hessian = np.column_stack(columns)
    return 0.5 * (hessian + hessian.T)


def _validate_frequencies(
    images: Sequence[Atoms],
    profile: Sequence[float],
    calculator: Calculator,
    config: PathwayConfig,
) -> FrequencyValidation:
    transition_state_index: int | None = None
    active_atom_ids: tuple[int, ...] = ()
    active_max_force: float | None = None
    try:
        transition_state_index = 1 + int(np.argmax(profile[1:-1]))
        saddle = images[transition_state_index]
        before, after = images[transition_state_index - 1], images[transition_state_index + 1]
        tangent = after.positions - before.positions
        if saddle.pbc.any():
            tangent = find_mic(tangent, cell=saddle.cell, pbc=saddle.pbc)[0]
        moves = np.linalg.norm(tangent, axis=1)
        active = np.flatnonzero(moves > _ACTIVE_MOTION_A)
        if len(active) < _MIN_ACTIVE_ATOMS:  # a very local mode: keep the atoms that move most
            active = np.sort(np.argsort(moves)[-_MIN_ACTIVE_ATOMS:])
        active_atom_ids = tuple(atom_ids(saddle)[index] for index in active)
        evaluated = saddle.copy()
        evaluated.calc = calculator
        forces = evaluated.get_forces(apply_constraint=False)[active]
        active_max_force = float(np.linalg.norm(forces, axis=1).max())
        hessian = _hessian(saddle, active, calculator, config.frequency_delta)
        if not np.isfinite(active_max_force) or not np.isfinite(hessian).all():
            raise ValueError("saddle forces or Hessian are non-finite")

        values, vectors = np.linalg.eigh(hessian)
        modes = int((values < -config.curvature_cutoff_eV_A2).sum())
        weights = np.repeat(saddle.get_masses()[active] ** -0.5, 3)
        weighted = np.linalg.eigvalsh(hessian * np.outer(weights, weights))
        # hbar sqrt(eV / Å² / amu) in cm⁻¹
        scale = units._hbar * 1e10 / np.sqrt(units._e * units._amu) / units.invcm
        along = tangent[active].reshape(-1)
        length = float(np.linalg.norm(along))
        return FrequencyValidation(
            status={0: "zero_imaginary_modes", 1: "one_imaginary_mode"}.get(
                modes, "multiple_imaginary_modes"
            ),
            transition_state_index=transition_state_index,
            active_atom_ids=active_atom_ids,
            displacement_A=config.frequency_delta,
            curvature_cutoff_eV_A2=config.curvature_cutoff_eV_A2,
            active_max_force_eV_A=active_max_force,
            eigenvalues_eV_A2=tuple(map(float, values)),
            frequencies_cm1=tuple(
                map(float, np.sign(weighted) * scale * np.sqrt(np.abs(weighted)))
            ),
            tangent_overlap=float(abs(vectors[:, 0] @ along) / length) if length else None,
            primary_mode=(
                tuple(tuple(map(float, row)) for row in vectors[:, 0].reshape(-1, 3))
                if modes
                else ()
            ),
        )
    except Exception as exc:
        return FrequencyValidation(
            status="failed",
            transition_state_index=transition_state_index,
            active_atom_ids=active_atom_ids,
            displacement_A=config.frequency_delta,
            curvature_cutoff_eV_A2=config.curvature_cutoff_eV_A2,
            active_max_force_eV_A=active_max_force,
            message=str(exc),
        )


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


def _rmsd(atoms: Atoms, reference: Atoms) -> float:
    difference = atoms.positions - reference.positions
    if reference.pbc.any():
        difference = find_mic(difference, cell=reference.cell, pbc=reference.pbc)[0]
    return float(np.sqrt((difference**2).sum(axis=1).mean()))


def _check_connectivity(
    saddle: Atoms,
    mode: tuple[tuple[float, float, float], ...],
    active_indices: tuple[int, ...],
    calculator: Calculator,
    config: PathwayConfig,
    pressure_GPa: float | None,
    detector_config: BondDetectorConfig,
    *,
    endpoints: tuple[Atoms, Atoms],
) -> ConnectivityCheck:
    """Relax the saddle moved both ways along its lowest mode and classify each side."""

    sides: list[str] = []
    try:
        (reactant, reactant_gap), (product, product_gap) = (
            classify_bonds(endpoint, detector_config) for endpoint in endpoints
        )
        if reactant_gap or product_gap:
            return ConnectivityCheck(
                "inconclusive",
                _CONNECTIVITY_DISPLACEMENT_A,
                message="a relaxed endpoint contains atom pairs inside the hysteresis gap",
            )
        vectors = np.asarray(mode, dtype=float)
        step = _CONNECTIVITY_DISPLACEMENT_A * vectors / np.linalg.norm(vectors, axis=1).max()
        for sign in (-1.0, 1.0):
            side = saddle.copy()
            side.positions[list(active_indices)] += sign * step
            converged, steps = _minimize(side, calculator, config, pressure_GPa)
            if not converged:
                sides.append("not_converged")
                continue
            if steps == 0:
                sides.append("no_step")
                continue
            # Every atom is free during descent; spectator changes and ambiguous pairs
            # must not count as reaching either of the actual relaxed NEB endpoints.
            bonds, gap = classify_bonds(side, detector_config)
            if gap:
                sides.append("ambiguous")
            elif bonds == reactant == product:
                # A step that changes no bond, such as a conformer change: the nearer end.
                nearer = _rmsd(side, endpoints[0]) <= _rmsd(side, endpoints[1])
                sides.append("reactant" if nearer else "product")
            else:
                sides.append(
                    "reactant" if bonds == reactant else "product" if bonds == product else "other"
                )
    except Exception as exc:
        return ConnectivityCheck(
            "failed", _CONNECTIVITY_DISPLACEMENT_A, tuple(sides), message=str(exc)
        )
    if sorted(sides) == ["product", "reactant"]:
        status = "connects_endpoints"
    elif {"not_converged", "no_step", "ambiguous"} & set(sides):
        status = "inconclusive"
    else:
        status = "does_not_connect"
    return ConnectivityCheck(status, _CONNECTIVITY_DISPLACEMENT_A, tuple(sides))


def _verdict(
    validation: FrequencyValidation, connectivity: ConnectivityCheck | None
) -> tuple[str, str]:
    if validation.status == "failed":
        return "saddle_unchecked", f"the Hessian failed: {validation.message}"
    if validation.status == "zero_imaginary_modes":
        return "no_saddle", "the highest image has no negative curvature beyond the cutoff"
    if validation.status == "multiple_imaginary_modes":
        return "saddle_rejected", "the highest image has more than one negative curvature"
    assert connectivity is not None
    if connectivity.status == "connects_endpoints":
        return "saddle_validated", ""
    if connectivity.status == "does_not_connect":
        return "saddle_rejected", "the saddle does not relax to both ends of the band"
    if connectivity.status == "failed":
        return "saddle_unchecked", f"the connectivity check failed: {connectivity.message}"
    detail = f": {connectivity.message}" if connectivity.message else ""
    return "saddle_unchecked", f"the connectivity check was inconclusive{detail}"


def check_saddle(
    images: Sequence[Atoms],
    calculator: Calculator,
    *,
    detector_config: BondDetectorConfig,
    config: PathwayConfig | None = None,
    pressure_GPa: float | None = None,
) -> PathwayOutcome:
    """Test whether a converged band's highest interior image is a saddle between its two ends.

    The image must have exactly one negative curvature in its fixed-cell Hessian, and the saddle
    moved both ways along that mode must relax to the two ends; only then does the outcome carry a
    barrier. The profile is the energy, or the enthalpy at ``pressure_GPa``, where the sides also
    relax the cell. The images are not changed.
    """

    options = config or PathwayConfig()
    band = [canonical_copy(image) for image in images]
    if len(band) < 3:
        raise ValueError("a band needs its two ends and at least one image between them")
    energies = []
    for image in band:
        image.calc = calculator
        try:
            energies.append(float(image.get_potential_energy()))
        finally:
            image.calc = None
    volumes = () if pressure_GPa is None else tuple(float(image.get_volume()) for image in band)
    enthalpies = (
        ()
        if pressure_GPa is None
        else tuple(
            energy + pressure_GPa * units.GPa * volume
            for energy, volume in zip(energies, volumes, strict=True)
        )
    )
    profile = energies if pressure_GPa is None else enthalpies
    common: dict[str, Any] = {
        "energies": tuple(energies),
        "enthalpies": enthalpies,
        "volumes": volumes,
        "images": tuple(band),
        "pressure_GPa": pressure_GPa,
    }
    # A saddle between two minima lies above both. When an endpoint is the highest point, the band
    # has not found one, so there is no barrier or saddle to check.
    if max(profile[1:-1]) <= max(profile[0], profile[-1]):
        return PathwayOutcome(
            "no_saddle",
            message="an endpoint is the highest point, so the band holds no saddle",
            **common,
        )
    validation = _validate_frequencies(band, profile, calculator, options)
    connectivity = None
    if validation.primary_mode:
        assert validation.transition_state_index is not None
        index = {atom_id: position for position, atom_id in enumerate(atom_ids(band[0]))}
        connectivity = _check_connectivity(
            band[validation.transition_state_index],
            validation.primary_mode,
            tuple(index[atom_id] for atom_id in validation.active_atom_ids),
            calculator,
            options,
            pressure_GPa,
            detector_config,
            endpoints=(band[0], band[-1]),
        )
    status, message = _verdict(validation, connectivity)
    return PathwayOutcome(
        status,
        barrier=max(profile) - profile[0] if status == "saddle_validated" else None,
        message=message,
        frequency_validation=validation,
        connectivity=connectivity,
        **common,
    )


def refine_pathway(
    candidate: ReactionCandidate,
    *,
    calculator_provider: CalculatorProvider,
    config: PathwayConfig | None = None,
    detector_config: BondDetectorConfig,
    pressure_GPa: float | None = None,
    frames: Mapping[int, Atoms] | None = None,
) -> PathwayOutcome:
    """Run NVT NEB, or SSNEB when a target pressure (including zero) is supplied, then gate the
    converged band with ``check_saddle``.

    ``frames`` are saved observations by frame number. The reactant is relaxed from the latest one
    no later than the candidate's reactant frame, and at most ``reactant_lookback`` observations
    before it, in which every changed bond is on its reactant side of the hysteresis gap. With
    ``split_steps``, an event that changes several bonds is refined one step at a time through the
    intermediates that the frames between its reactant and product relax into.
    """

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
        frames=frames or {},
    )
    method = "neb" if pressure_GPa is None else "ssneb"
    steps = tuple(
        replace(step, outcome=replace(step.outcome, method=method, pressure_GPa=pressure_GPa))
        for step in outcome.steps
    )
    return replace(outcome, method=method, pressure_GPa=pressure_GPa, steps=steps)


def _refine_pathway(
    candidate: ReactionCandidate,
    *,
    calculator_provider: CalculatorProvider,
    config: PathwayConfig | None,
    detector_config: BondDetectorConfig,
    pressure_GPa: float | None,
    frames: Mapping[int, Atoms],
) -> PathwayOutcome:

    if not candidate.resolved:
        return PathwayOutcome("unresolved", message="candidate topology was not resolved")
    if candidate.reactant_bonds == candidate.product_bonds:
        return PathwayOutcome("unresolved", message="candidate has no changed bonds")
    options = config or PathwayConfig()
    changed = candidate.reactant_bonds ^ candidate.product_bonds
    first, last = candidate.reactant_frame, candidate.product_frame
    saved = {**frames, first: candidate.reactant, last: candidate.product}
    # By the tracker's reactant frame a changed bond can already be inside its hysteresis gap, and
    # relaxing that frame can carry the change on, so the reactant comes from the latest frame
    # before the bonds began to change.
    start = next(
        (
            frame
            for frame in range(first, first - options.reactant_lookback - 1, -1)
            if frame in saved and _on_reactant_side(saved[frame], candidate, detector_config)
        ),
        first,
    )
    numbers = [frame for frame in sorted(saved) if start <= frame <= last]
    images: list[Atoms] = []
    try:
        variable_cell = pressure_GPa is not None
        prepared = dict(
            zip(
                numbers,
                _prepare_frames(
                    [saved[frame] for frame in numbers], options, variable_cell=variable_cell
                ),
                strict=True,
            )
        )
        if not set(candidate.atom_ids) <= set(atom_ids(prepared[first])):
            raise ValueError("candidate atom IDs are missing from its endpoints")
        images = [prepared[start], prepared[last]]

        def relaxed(frame: int, stage: str) -> tuple[Atoms, bool]:
            atoms = prepared[frame].copy()
            converged = _relax(
                atoms,
                stage=stage,
                calculator_provider=calculator_provider,
                config=options,
                pressure_GPa=pressure_GPa,
            )
            return atoms, converged

        reactant, converged = relaxed(start, "relax_reactant")
        if start != first and not (
            converged and _matches_outside(reactant, candidate.reactant, changed, detector_config)
        ):
            # Bonds away from the event differ between the two frames, so keep the tracker's.
            start = first
            reactant, converged = relaxed(start, "relax_reactant")
        state = _made(reactant, candidate, detector_config) if converged else None
        # The intermediates in the order the trajectory reached them, each with the first frame
        # that relaxes into it.
        ends: list[tuple[frozenset[Bond], int]] = []
        if options.split_steps and len(changed) > 1 and state is not None and state != changed:
            for frame in numbers:
                if not start < frame < last:
                    continue
                atoms, done = relaxed(frame, "relax_frame")
                made = _made(atoms, candidate, detector_config) if done else None
                if made == changed:
                    break
                if made is not None and made > (ends[-1][0] if ends else state):
                    ends.append((made, frame))
        ends.append((changed, last))
    except _EndpointUnresolved as exc:
        return PathwayOutcome("unresolved", images=_snapshots(images), message=str(exc))
    except Exception as exc:
        return PathwayOutcome("failed", images=_snapshots(images), message=str(exc))

    steps: list[PathwayStep] = []
    before: frozenset[Bond] = frozenset()
    if options.split_steps and state is not None and 0 < len(state) < len(changed):
        # The reactant frame relaxes into an intermediate, so the steps from there still count.
        before = state
        steps.append(
            PathwayStep(
                PathwayOutcome(
                    "unresolved",
                    images=_snapshots([reactant]),
                    message="the reactant frame relaxes partway along the event",
                ),
                candidate.reactant_bonds,
                candidate.reactant_bonds ^ state,
            )
        )
    source: int = start
    end: Atoms | None = reactant
    for made, frame in ends:
        step = replace(
            candidate,
            reactant_bonds=candidate.reactant_bonds ^ before,
            product_bonds=candidate.reactant_bonds ^ made,
        )
        if end is None:
            outcome = PathwayOutcome("unresolved", message="the step before ended elsewhere")
        else:
            outcome, end = _refine_step(
                step,
                end,
                converged,
                prepared[frame],
                calculator_provider=calculator_provider,
                options=options,
                detector_config=detector_config,
                pressure_GPa=pressure_GPa,
            )
        steps.append(
            PathwayStep(
                replace(outcome, frames=(source, frame)), step.reactant_bonds, step.product_bonds
            )
        )
        before, source, converged = made, frame, True
    if len(steps) == 1:
        return steps[0].outcome
    return PathwayOutcome(
        "stepwise",
        message=f"refined in {len(steps)} steps through {len(steps) - 1} intermediates",
        frames=(start, last),
        steps=tuple(steps),
    )


def _refine_step(
    candidate: ReactionCandidate,
    reactant: Atoms,
    reactant_converged: bool,
    frame: Atoms,
    *,
    calculator_provider: CalculatorProvider,
    options: PathwayConfig,
    detector_config: BondDetectorConfig,
    pressure_GPa: float | None,
) -> tuple[PathwayOutcome, Atoms | None]:
    """Refine the candidate from its relaxed reactant to a product built from the frame.

    Returns the outcome and, once both ends match the candidate, the relaxed product, where the
    next step starts.
    """

    images = [reactant]
    end: Atoms | None = None
    try:
        reacting = {
            atom_id
            for bond in candidate.reactant_bonds ^ candidate.product_bonds
            for atom_id in bond
        }
        index = {atom_id: position for position, atom_id in enumerate(atom_ids(reactant))}
        moved = sorted(
            {
                index[atom_id]
                for bond in candidate.reactant_bonds | candidate.product_bonds
                if reacting.intersection(bond)
                for atom_id in bond
            }
        )
        # The product starts from the relaxed reactant, cell included, with only the atoms in
        # changed bonds and the atoms bonded to them where they sat in the product frame. Every
        # other atom starts at the same minimum in both endpoints, so the band spends its images
        # on the reaction rather than on hot-frame motion around it, and a hot frame cannot relax
        # the surroundings differently in each. Those atoms are placed, not moved by how far they
        # went between the frames: relaxing the reactant can swing a stretched atom round its
        # partner, and that move added on can land it on a neighbor. Nothing is constrained:
        # every atom and, for NPT, the cell relax.
        product = frame.copy()
        start = reactant.get_scaled_positions(wrap=False)
        start[moved] = product.get_scaled_positions(wrap=False)[moved]
        product.set_cell(reactant.cell)
        product.set_scaled_positions(start)
        images = [reactant, product]
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
            ), None

        endpoint_status = _endpoint_status(candidate, reactant, product, detector_config)
        if endpoint_status is not None:
            status, message = endpoint_status
            return PathwayOutcome(status, images=_snapshots(images), message=message), None

        end = product
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
                # The endpoints never move in the band, but ASE asks for their energies on every
                # step; evaluate each once with this lease instead.
                for endpoint in (images[0], images[-1]):
                    endpoint.calc = SinglePointCalculator(
                        endpoint, energy=endpoint.get_potential_energy()
                    )
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
                    ), end
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
                    ), end
                return check_saddle(
                    images,
                    calculator,
                    detector_config=detector_config,
                    config=options,
                    pressure_GPa=pressure_GPa,
                ), end
            finally:
                for image in images:
                    image.calc = None
    except _EndpointUnresolved as exc:
        return PathwayOutcome("unresolved", images=_snapshots(images), message=str(exc)), end
    except Exception as exc:
        return PathwayOutcome("failed", images=_snapshots(images), message=str(exc)), end


__all__ = [
    "CONVERGED_STATUSES",
    "CalculatorProvider",
    "ConnectivityCheck",
    "FrequencyValidation",
    "PathwayConfig",
    "PathwayOutcome",
    "PathwayStep",
    "check_saddle",
    "refine_pathway",
]
