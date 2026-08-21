"""Generic exact MD adapter for deterministic, stateless ASE calculators."""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from importlib import import_module
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path
from typing import Any

import ase
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md.langevinbaoab import LangevinBAOAB
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from ..ase_npt import restore_langevin_baoab, snapshot_langevin_baoab
from ..campaign import TrajectorySpec
from ..restart import ComponentState, ExactRestartSnapshot

_CALCULATOR_KIND = "reactionflow.ase-calculator"
_ALLOWED_OPTIONS = {
    "calculator_factory",
    "calculator_kwargs",
    "model_files",
    "packages",
}
_ALLOWED_CONDITIONS = {
    "barostat_mass",
    "barostat_tau_fs",
    "disable_cell_langevin",
    "hydrostatic",
    "thermostat_tau_fs",
    "zero_total_momentum",
}


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _optional_positive_float(value: object, name: str) -> float | None:
    return None if value is None else _positive_float(value, name)


def _string_sequence(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an array of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise TypeError(f"{name} must be an array of non-empty strings")
    return result


class _ASERuntime:
    def __init__(self, dynamics: LangevinBAOAB, calculator_state: ComponentState) -> None:
        self._dynamics = dynamics
        self._calculator_state = calculator_state

    @property
    def atoms(self) -> Atoms:
        return self._dynamics.atoms

    @property
    def nsteps(self) -> int:
        return int(self._dynamics.nsteps)

    def run(self, steps: int) -> None:
        self._dynamics.run(steps)

    def snapshot(self) -> ExactRestartSnapshot:
        return ExactRestartSnapshot(
            atoms=self.atoms,
            dynamics=snapshot_langevin_baoab(self._dynamics),
            calculator=self._calculator_state,
        )


class ASELangevinBAOABAdapter:
    """Shared exact NVT/NPT runtime for adapters that lease ASE calculators."""

    def __init__(self, *, trajectory: TrajectorySpec) -> None:
        unknown_conditions = set(trajectory.conditions) - _ALLOWED_CONDITIONS
        if unknown_conditions:
            raise ValueError(f"unknown ASE trajectory conditions: {sorted(unknown_conditions)}")
        self.trajectory = trajectory

    def _lease(self) -> AbstractContextManager[tuple[Calculator, ComponentState]]:
        raise NotImplementedError

    def _dynamics(self, atoms: Atoms, calculator: Calculator) -> LangevinBAOAB:
        conditions = self.trajectory.conditions
        rng = np.random.default_rng(self.trajectory.seed)
        atoms.calc = calculator
        if "momenta" not in atoms.arrays:
            MaxwellBoltzmannDistribution(
                atoms,
                temperature_K=self.trajectory.temperature_K,
                rng=rng,
            )
            if bool(conditions.get("zero_total_momentum", True)):
                Stationary(atoms, preserve_temperature=True)
        pressure = self.trajectory.pressure_GPa
        return LangevinBAOAB(
            atoms,
            timestep=self.trajectory.timestep_fs * units.fs,
            temperature_K=self.trajectory.temperature_K,
            externalstress=None if pressure is None else -pressure * units.GPa,
            hydrostatic=bool(conditions.get("hydrostatic", True)),
            T_tau=_positive_float(
                conditions.get("thermostat_tau_fs", 100.0),
                "thermostat_tau_fs",
            )
            * units.fs,
            P_tau=(
                None
                if pressure is None
                else _positive_float(
                    conditions.get("barostat_tau_fs", 1000.0),
                    "barostat_tau_fs",
                )
                * units.fs
            ),
            P_mass=_optional_positive_float(
                conditions.get("barostat_mass"),
                "barostat_mass",
            ),
            disable_cell_langevin=bool(conditions.get("disable_cell_langevin", False)),
            rng=rng,
            logfile=None,
        )

    @contextmanager
    def start(self, atoms: Atoms) -> Iterator[_ASERuntime]:
        with self._lease() as (calculator, state):
            dynamics = self._dynamics(atoms, calculator)
            try:
                yield _ASERuntime(dynamics, state)
            finally:
                atoms.calc = None

    @contextmanager
    def restore(self, snapshot: ExactRestartSnapshot) -> Iterator[_ASERuntime]:
        with self._lease() as (calculator, state):
            if snapshot.calculator != state:
                raise ValueError("exact calculator environment differs from the checkpoint")
            dynamics = restore_langevin_baoab(snapshot.atoms, calculator, snapshot.dynamics)
            try:
                yield _ASERuntime(dynamics, state)
            finally:
                snapshot.atoms.calc = None

    @contextmanager
    def calculator(self, _stage: str) -> Iterator[Calculator]:
        with self._lease() as (calculator, _state):
            yield calculator


class ASECalculatorAdapter(ASELangevinBAOABAdapter):
    """Load one deterministic ASE calculator from a campaign configuration."""

    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        unknown_options = set(options) - _ALLOWED_OPTIONS
        if unknown_options:
            raise ValueError(f"unknown ASE calculator adapter options: {sorted(unknown_options)}")
        factory = options.get("calculator_factory")
        if (
            not isinstance(factory, str)
            or factory.count(":") != 1
            or any(not part for part in factory.split(":"))
        ):
            raise ValueError("calculator_factory must be 'module:callable'")
        kwargs = options.get("calculator_kwargs", {})
        if not isinstance(kwargs, Mapping) or any(not isinstance(key, str) for key in kwargs):
            raise TypeError("calculator_kwargs must be an object with string keys")
        super().__init__(trajectory=trajectory)
        self.calculator_factory = factory
        self.calculator_kwargs = dict(kwargs)
        self.model_files = tuple(
            Path(item).expanduser().resolve()
            for item in _string_sequence(options.get("model_files", []), "model_files")
        )
        self.packages = _string_sequence(options.get("packages", []), "packages")
        self._model_hashes: dict[str, str] | None = None
        self._contract: ComponentState | None = None

    def _factory(self) -> tuple[Any, Any]:
        module_name, factory_name = self.calculator_factory.split(":", 1)
        module = import_module(module_name)
        factory = getattr(module, factory_name, None)
        if not callable(factory):
            raise TypeError(f"ASE calculator factory {self.calculator_factory!r} is not callable")
        return module, factory

    def _model_file_hashes(self) -> dict[str, str]:
        if self._model_hashes is not None:
            return self._model_hashes
        hashes: dict[str, str] = {}
        for path in self.model_files:
            if not path.is_file():
                raise FileNotFoundError(f"MLIP model file is missing: {path}")
            hashes[str(path)] = _sha256(path)
        self._model_hashes = hashes
        return self._model_hashes

    def _package_versions(self, module_name: str) -> dict[str, str]:
        names = set(self.packages)
        names.update(packages_distributions().get(module_name.partition(".")[0], ()))
        versions: dict[str, str] = {}
        for name in sorted(names):
            try:
                versions[name] = version(name)
            except PackageNotFoundError as error:
                raise RuntimeError(
                    f"required provenance package is not installed: {name}"
                ) from error
        return versions

    def _calculator_state(self, module: Any, calculator: Calculator) -> ComponentState:
        calculator_type = f"{type(calculator).__module__}:{type(calculator).__qualname__}"
        if self._contract is not None:
            if self._contract.metadata["calculator_type"] != calculator_type:
                raise TypeError("ASE calculator factory returned a different calculator type")
            return self._contract
        module_file = getattr(module, "__file__", None)
        source_path = None if module_file is None else Path(module_file).resolve()
        source_sha256 = (
            None if source_path is None or not source_path.is_file() else _sha256(source_path)
        )
        self._contract = ComponentState(
            kind=_CALCULATOR_KIND,
            metadata={
                "factory": self.calculator_factory,
                "calculator_type": calculator_type,
                "calculator_kwargs": self.calculator_kwargs,
                "model_files": self._model_file_hashes(),
                "package_versions": self._package_versions(module.__name__),
                "factory_source": None if source_path is None else str(source_path),
                "factory_source_sha256": source_sha256,
                "python_version": platform.python_version(),
                "ase_version": ase.__version__,
                "numpy_version": np.__version__,
            },
        )
        return self._contract

    @contextmanager
    def _lease(self) -> Iterator[tuple[Calculator, ComponentState]]:
        self._model_file_hashes()
        module, factory = self._factory()
        calculator = factory(**self.calculator_kwargs)
        if not isinstance(calculator, Calculator):
            raise TypeError(
                f"ASE calculator factory {self.calculator_factory!r} returned "
                f"{type(calculator).__name__}, not an ASE Calculator"
            )
        state = self._calculator_state(module, calculator)
        try:
            yield calculator, state
        finally:
            close = getattr(calculator, "close", None)
            if callable(close):
                close()


def create_adapter(
    *, trajectory: TrajectorySpec, options: Mapping[str, Any]
) -> ASECalculatorAdapter:
    """Campaign factory for a configured deterministic ASE calculator."""

    return ASECalculatorAdapter(trajectory=trajectory, options=options)


__all__ = ["ASECalculatorAdapter", "ASELangevinBAOABAdapter", "create_adapter"]
