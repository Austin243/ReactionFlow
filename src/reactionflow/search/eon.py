"""EON process searches and state comparison through ReactionFlow's live ASE calculator.

The supported interface is pyeonclient 0.4.2, the EON 3.5.0 client. One search is EON's own
ProcessSearchJob: a random push, a min-mode saddle search, relaxation of both sides along the
saddle mode, the check that exactly one side is the starting minimum, the barriers, and harmonic
transition-state prefactors. States are compared with EON's structure comparison.
"""

from __future__ import annotations

import contextlib
import copy
import math
import tempfile
from dataclasses import asdict, dataclass, field
from importlib import import_module
from importlib.metadata import version
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.geometry import get_distances

from .._durable import file_digest
from ..detection import atom_ids, canonical_copy

EON_VERSION = "0.4.2"
# EON ends a search with these after its saddle converged: 6 not connected to the start,
# 7 prefactors out of range, 9 minimizations from the saddle not converged, 10 Hessian failed.
NOT_CONNECTED = 6
_AFTER_SADDLE = {NOT_CONNECTED, 7, 9, 10}


def _number(value: object, name: str, *, integer: bool = False) -> float:
    kind = Integral if integer else Real
    if isinstance(value, bool) or not isinstance(value, kind) or not math.isfinite(value):
        raise ValueError(f"eon.{name} must be a finite {'integer' if integer else 'number'}")
    if value <= 0:
        raise ValueError(f"eon.{name} must be positive")
    return int(value) if integer else float(value)


@dataclass(frozen=True)
class EONSettings:
    """Search settings. The push covers every free atom unless `displace` is "local"."""

    displace: str = "all"
    displace_size_A: float = 0.5
    displace_radius_A: float = 4.0
    displace_centers: tuple = ()
    force_tolerance: float = 0.01
    force_metric: str = "norm"
    max_iterations: int = 1000
    max_energy_eV: float = 20.0
    prefactor: float | None = None
    match_distance_A: float = 0.1
    match_energy_eV: float = 0.01
    equivalent_atoms: bool = True
    rigid_rotation: bool | str = "auto"
    log: str = "file"

    def __post_init__(self) -> None:
        if self.displace not in {"all", "local"}:
            raise ValueError("eon.displace must be 'all' or 'local'")
        if self.force_metric not in {"norm", "max_atom"}:
            raise ValueError("eon.force_metric must be 'norm' or 'max_atom'")
        if self.rigid_rotation != "auto" and type(self.rigid_rotation) is not bool:
            raise ValueError("eon.rigid_rotation must be 'auto', true, or false")
        if self.log not in {"file", "terminal"}:
            raise ValueError("eon.log must be 'file' or 'terminal'")
        for name in (
            "displace_size_A",
            "displace_radius_A",
            "force_tolerance",
            "max_energy_eV",
            "match_distance_A",
            "match_energy_eV",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        _number(self.max_iterations, "max_iterations", integer=True)
        if self.prefactor is not None:
            object.__setattr__(self, "prefactor", _number(self.prefactor, "prefactor"))
        centers = self.displace_centers
        if isinstance(centers, str) or not isinstance(centers, (list, tuple)):
            raise ValueError("eon.displace_centers must be a list of elements or atom indices")
        for center in centers:
            if isinstance(center, bool) or not isinstance(center, (str, Integral)):
                raise ValueError("eon.displace_centers holds element symbols or atom indices")
            if isinstance(center, Integral) and center < 0:
                raise ValueError("eon.displace_centers atom indices must be non-negative")
        object.__setattr__(self, "displace_centers", tuple(centers))
        if type(self.equivalent_atoms) is not bool:
            raise ValueError("eon.equivalent_atoms must be true or false")

    def rotation_for(self, atoms: Atoms) -> bool:
        """EON can treat rigid rotation as a symmetry only when no atom is fixed."""

        if self.rigid_rotation != "auto":
            return self.rigid_rotation
        return free_rotation(atoms) == "free"


@dataclass(frozen=True)
class ProcessResult:
    """One EON process search. Prefactors are in 1/s, energies in eV. A search that EON rejects
    after its saddle converged keeps the saddle and the minima reached from it, with energies."""

    status: str
    message: str
    saddle: Atoms | None = None
    product: Atoms | None = None
    reactant_energy: float | None = None
    saddle_energy: float | None = None
    product_energy: float | None = None
    prefactor: float | None = None
    reverse_prefactor: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    minima: tuple[tuple[Atoms, float], ...] = ()


def _load_eon():
    try:
        module = import_module("pyeonclient")
    except ImportError as error:
        raise RuntimeError(
            "EON search requires the optional backend; install "
            f"'reactionflow[eon]' (pyeonclient[ase]=={EON_VERSION})"
        ) from error
    installed = version("pyeonclient")
    if installed != EON_VERSION:
        raise RuntimeError(f"EON requires pyeonclient=={EON_VERSION}; found {installed}")
    return module


def validate_start(atoms: Atoms) -> np.ndarray:
    """Check a structure EON can search from and return its per-atom fixed mask."""

    if "atom_id" not in atoms.arrays and "atom_ids" not in atoms.info:
        atoms = canonical_copy(atoms)
    atom_ids(atoms)
    if len(atoms) == 0 or not np.isfinite(atoms.positions).all():
        raise ValueError("EON requires a non-empty structure with finite positions")
    if not np.isfinite(atoms.cell).all() or abs(np.linalg.det(atoms.cell)) < 1e-12:
        raise ValueError("EON requires a nonsingular fixed cell, including molecules")
    if atoms.pbc.any() and not atoms.pbc.all():
        raise ValueError("EON requires fully periodic or nonperiodic structures")
    fixed = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            raise ValueError("EON supports only FixAtoms constraints")
        fixed[constraint.get_indices()] = True
    if fixed.all():
        raise ValueError("EON requires at least one unconstrained atom")
    return fixed


def free_rotation(atoms: Atoms) -> str | None:
    """'free' when nothing stops the structure turning as a whole, 'pinned' when only fixed atoms
    fewer than three or on one line hold it, otherwise None (a periodic cell or a firm anchor)."""

    fixed = validate_start(atoms)
    if atoms.pbc.any():
        return None
    if not fixed.any():
        return "free"
    anchors = atoms.positions[fixed] - atoms.positions[fixed].mean(axis=0)
    if len(anchors) >= 3 and np.linalg.matrix_rank(anchors, tol=1e-3) >= 2:
        return None
    return "pinned"


class _PreservingCalculator(Calculator):
    """Restore info/arrays omitted by EON's geometry-only ASE conversion."""

    implemented_properties = ("energy", "forces")

    def __init__(self, calculator: Calculator, template: Atoms):
        super().__init__()
        self.calculator = calculator
        self.template = template.copy()
        self.template.info = copy.deepcopy(template.info)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if not np.array_equal(atoms.numbers, self.template.numbers):
            raise ValueError("EON changed atom order or species")
        # EON passes a zero force-evaluation cell for isolated molecules while
        # retaining Matter's original cell for geometry/I/O. Restore the ASE
        # cell from the source; a periodic calculation must keep its real cell.
        isolated_force_cell = not self.template.pbc.any() and not atoms.cell.array.any()
        if not isolated_force_cell and not np.allclose(
            atoms.cell, self.template.cell, rtol=0, atol=1e-12
        ):
            raise ValueError("EON changed the fixed cell")
        physical = self.template.copy()
        physical.info = copy.deepcopy(self.template.info)
        physical.set_positions(atoms.positions, apply_constraint=False)
        physical.calc = self.calculator
        energy = physical.get_potential_energy()
        # EON applies the fixed-coordinate mask itself.
        forces = physical.get_forces(apply_constraint=False)
        if not np.isfinite(energy) or forces.shape != physical.positions.shape:
            raise ValueError("calculator returned invalid energy or force shape")
        if not np.isfinite(forces).all():
            raise ValueError("calculator returned nonfinite forces")
        self.results = {"energy": energy, "forces": forces}


def _ini(sections: dict[str, dict[str, object]]) -> str:
    def text(value: object) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        return repr(value) if isinstance(value, float) else str(value)

    lines = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {text(value)}" for key, value in values.items())
    return "\n".join(lines) + "\n"


def _results(path: Path) -> dict[str, str]:
    """EON's results.dat: one '<value> <key>' pair per line."""

    pairs = {}
    for line in path.read_text().splitlines():
        value, _, key = line.strip().rpartition(" ")
        if key:
            pairs[key] = value
    return pairs


def _moved(atoms: Atoms, positions: np.ndarray) -> Atoms:
    moved = atoms.copy()
    moved.info = copy.deepcopy(atoms.info)
    moved.set_positions(np.asarray(positions, dtype=float), apply_constraint=False)
    return moved


class EONBackend:
    """EON on one caller-owned calculator, in eV, Å and atomic mass units."""

    def __init__(self, calculator: Calculator, settings: EONSettings, temperature_K: float):
        self.calculator = calculator
        self.settings = settings
        self.temperature_K = float(temperature_K)
        self._eon = _load_eon()
        # Comparisons never evaluate the potential; they need only the comparison settings.
        self._comparison = {
            rotation: self._parameters(seed=1, rotation=rotation) for rotation in (False, True)
        }

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "backend": "pyeonclient",
            "version": EON_VERSION,
            # Where EON's log goes does not change any result.
            "settings": {k: v for k, v in asdict(self.settings).items() if k != "log"},
            "source_sha256": file_digest(Path(__file__)),
        }

    def _parameters(
        self, *, seed: int, rotation: bool, center: int = 0, radius: float = 0.0, size: float = 0.0
    ):
        options = self.settings
        sections = {
            # EON reseeds its own generator from random_seed whenever settings are loaded.
            "Main": {"random_seed": seed % 2147483646 + 1, "temperature": self.temperature_K},
            "Saddle Search": {
                "method": "min_mode",
                "min_mode_method": "dimer",
                "client_displace_type": "listed_atoms",
                "displace_atom_list": center,
                "displace_radius": radius,
                "displace_magnitude": size,
                "max_energy": options.max_energy_eV,
                "max_iterations": options.max_iterations,
                "remove_rotation": rotation,
            },
            "Dimer": {"remove_rotation": rotation},
            "Process Search": {"minimize_first": False},
            # EON's saddle search takes its force tolerance and metric from here as well.
            "Optimizer": {
                "converged_force": options.force_tolerance,
                "convergence_metric": options.force_metric,
                "max_iterations": options.max_iterations,
            },
            "Prefactor": {"default_value": options.prefactor or 0.0},
            "Structure Comparison": {
                "distance_difference": options.match_distance_A,
                "energy_difference": options.match_energy_eV,
                "indistinguishable_atoms": options.equivalent_atoms,
                # Also drops the rigid-body zero modes from the prefactor Hessians.
                "check_rotation": rotation,
            },
        }
        parameters = self._eon.Parameters()
        parameters.load_ini_text(_ini(sections))
        parameters.quiet = True
        parameters.write_log = False
        parameters.write_movies = False
        return parameters

    def _matter(self, atoms: Atoms, fixed: np.ndarray, parameters, *, calculator=None):
        bridge = _PreservingCalculator(calculator or self.calculator, atoms)
        potential = self._eon.potential_from_ase(bridge)
        geometry = atoms.copy()
        geometry.set_constraint()
        matter = self._eon.from_ase(geometry, potential, parameters)
        matter.fixed = np.repeat(fixed[:, None], 3, axis=1).astype(np.int32)
        return matter, potential

    def _push(self, atoms: Atoms, fixed: np.ndarray, seed: int) -> tuple[int, float, float]:
        """Choose the push center and give EON the Gaussian width for the requested size."""

        free = np.flatnonzero(~fixed)
        candidates = free
        options = self.settings
        if options.displace == "local" and options.displace_centers:
            symbols = atoms.get_chemical_symbols()
            wanted = set(options.displace_centers)
            candidates = np.array([i for i in free if i in wanted or symbols[i] in wanted], int)
            if not len(candidates):
                raise ValueError("no free atom matches eon.displace_centers")
        center = int(np.random.default_rng(seed).choice(candidates))
        if options.displace == "all":
            radius, moved = 1e6, len(free)
        else:
            radius = options.displace_radius_A
            distances = get_distances(
                atoms.positions[center], atoms.positions[free], cell=atoms.cell, pbc=atoms.pbc
            )[1][0]
            moved = int(np.count_nonzero(distances <= radius))
        return center, radius, options.displace_size_A / math.sqrt(3 * moved)

    def relax(self, atoms: Atoms) -> tuple[Atoms, float]:
        """Relax a starting structure with the optimizer the process searches use."""

        source = canonical_copy(atoms)
        fixed = validate_start(source)
        parameters = self._parameters(seed=1, rotation=self.settings.rotation_for(source))
        matter, _ = self._matter(source, fixed, parameters)
        relaxed, converged = matter.relax()
        if not converged:
            raise ValueError("starting structure did not relax to a minimum")
        return _moved(source, relaxed.positions), float(relaxed.potential_energy)

    def search(self, atoms: Atoms, *, seed: int) -> ProcessResult:
        """Run one process search from a relaxed state; failures are results, not errors."""

        metadata: dict[str, Any] = {"seed": int(seed)}
        try:
            source = canonical_copy(atoms)
            fixed = validate_start(source)
            center, radius, size = self._push(source, fixed, seed)
            metadata.update(center=center, displace_magnitude_A=size)
            parameters = self._parameters(
                seed=seed,
                rotation=self.settings.rotation_for(source),
                center=center,
                radius=radius,
                size=size,
            )
            matter, potential = self._matter(source, fixed, parameters)
            # pyeonclient 0.4.2 binds the full job in _core without re-exporting it.
            job = self._eon._core.ProcessSearchJob(potential, parameters)
            # The job writes EON's results.dat and .con files to the working directory.
            with tempfile.TemporaryDirectory() as work, contextlib.chdir(work):
                job.run_from_matter(matter)
                results = _results(Path("results.dat"))
            message = results.get("termination_reason_text", "")
            reason = int(results["termination_reason"])
            metadata.update(
                force_calls=int(results.get("total_force_calls", 0)), termination_reason=reason
            )
            if reason in _AFTER_SADDLE:
                return ProcessResult(
                    "failed",
                    message,
                    saddle=_moved(source, job.saddle.positions),
                    saddle_energy=float(results["potential_energy_saddle"]),
                    metadata=metadata,
                    minima=tuple(
                        (_moved(source, side.positions), float(side.potential_energy))
                        for side in (job.min1, job.min2)
                    ),
                )
            if reason != 0:
                return ProcessResult("failed", message, metadata=metadata)
            return ProcessResult(
                "good",
                message,
                saddle=_moved(source, job.saddle.positions),
                product=_moved(source, job.min2.positions),
                reactant_energy=float(results["potential_energy_reactant"]),
                saddle_energy=float(results["potential_energy_saddle"]),
                product_energy=float(results["potential_energy_product"]),
                prefactor=float(results["prefactor_reactant_to_product"]),
                reverse_prefactor=float(results["prefactor_product_to_reactant"]),
                metadata=metadata,
            )
        except Exception as error:
            return ProcessResult("failed", f"{type(error).__name__}: {error}", metadata=metadata)

    def prefactors(self, first: Atoms, saddle: Atoms, second: Atoms) -> tuple[float, float] | None:
        """EON's harmonic prefactors from `first` over `saddle` to `second` and back, or None when
        EON cannot compute them or they fall outside its accepted range. A fixed `prefactor` is
        returned as is."""

        if self.settings.prefactor is not None:
            return self.settings.prefactor, self.settings.prefactor
        parameters = self._parameters(seed=1, rotation=self.settings.rotation_for(first))
        fixed = validate_start(first)
        matters = [self._matter(atoms, fixed, parameters)[0] for atoms in (first, saddle, second)]
        try:
            # EON writes its Hessian and frequencies to the working directory.
            with tempfile.TemporaryDirectory() as work, contextlib.chdir(work):
                values = tuple(float(v) for v in self._eon.get_prefactors(parameters, *matters))
        except Exception:
            return None
        low, high = parameters.prefactor_min_value, parameters.prefactor_max_value
        return values if all(low <= value <= high for value in values) else None

    def same(self, first: Atoms, first_energy: float, second: Atoms, second_energy: float) -> bool:
        """EON's structure comparison, after an energy check that needs no calculation."""

        if abs(first_energy - second_energy) > self.settings.match_energy_eV:
            return False
        rotation = self.settings.rotation_for(first)
        matters = [
            self._matter(atoms, validate_start(atoms), self._comparison[rotation])[0]
            for atoms in (first, second)
        ]
        # EON's comparison up to both rotation and exchange matches no structure of two or more
        # elements, not even an exact copy, so such a structure is compared up to rotation only.
        exchange = self.settings.equivalent_atoms and not (rotation and len(set(first.numbers)) > 1)
        return bool(self._eon.structures_equal(*matters, exchange))
