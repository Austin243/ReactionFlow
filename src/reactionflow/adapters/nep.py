"""NEP89 with Calorine's in-memory, double-precision CPU calculator."""

from __future__ import annotations

import inspect
import platform
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import ase
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ..campaign import TrajectorySpec
from ..restart import ComponentState
from ._model_files import cached_download, model_cache, require_package
from .ase import ASELangevinBAOABAdapter, _sha256

VERSION = "4.0"
_RELEASE = "05982941c85b257fdf5a5c69fb7922ce7d7a5c80"
_URL = (
    f"https://raw.githubusercontent.com/brucefan1983/GPUMD/{_RELEASE}/"
    "potentials/nep/nep89_20250409/nep89_20250409.txt"
)


def catalog() -> dict[str, Any]:
    return {
        "backend": "nep",
        "factory": "reactionflow.adapters.nep:create_adapter",
        "package": f"calorine=={VERSION}",
        "models": [{"name": "nep89", "description": "NEP89, 2025-04-09 release; 89 elements"}],
        "notes": [
            "CPU float64 implementation; set campaign require_gpu=false. No Torch dependency.",
            "Requires fully periodic structures; the backend always applies periodic boundaries.",
            "Local energy-model nep.txt files are also supported; no explicit charge/spin inputs.",
            "Installing Calorine may require a C++17 compiler.",
        ],
        "sources": [
            "https://github.com/brucefan1983/GPUMD/tree/v5.0/potentials/nep/nep89_20250409",
            "https://calorine.materialsmodeling.org/get_started/ase_calculators.html",
        ],
    }


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    unknown = set(options) - {"model", "checkpoint", "device", "cache_dir"}
    if unknown:
        raise ValueError(f"unknown NEP options: {sorted(unknown)}")
    if ("model" in options) == ("checkpoint" in options):
        raise ValueError("NEP requires exactly one of model or checkpoint")
    if "model" in options and options["model"] != "nep89":
        raise ValueError("unknown NEP model; choose 'nep89' or an explicit checkpoint")
    if options.get("device", "cpu") != "cpu":
        raise ValueError("the built-in NEP adapter uses the CPU backend; set device='cpu'")
    for key in ("checkpoint", "cache_dir"):
        if key in options and (
            not isinstance(options[key], str) or not Path(options[key]).is_absolute()
        ):
            raise ValueError(f"NEP {key} must be an absolute path")
    require_package("calorine", VERSION, "nep")
    path = (
        Path(options["checkpoint"]).resolve()
        if "checkpoint" in options
        else cached_download(_URL, model_cache(options, "nep"), download=download)
    )
    if not path.is_file():
        raise FileNotFoundError(f"NEP checkpoint is missing: {path}")
    return {"checkpoint": path}


def _validate_atoms(atoms: Atoms) -> None:
    if not np.isfinite(atoms.cell).all() or not np.isfinite(atoms.positions).all():
        raise ValueError("NEP requires finite cell and atomic coordinates")
    if not atoms.pbc.all() or atoms.cell.rank != 3 or atoms.cell.volume <= 0:
        raise ValueError("Calorine CPUNEP requires a fully periodic structure with a 3D cell")
    if atoms.info.get("charge", 0) != 0 or atoms.info.get("spin", 1) != 1:
        raise ValueError("NEP89 does not accept explicit charge or spin states")
    if np.any(atoms.get_initial_charges()) or np.any(atoms.get_initial_magnetic_moments()):
        raise ValueError("NEP does not accept nonzero initial charge or magnetic-moment arrays")


class NEPAdapter(ASELangevinBAOABAdapter):
    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        super().__init__(trajectory=trajectory)
        self.options = dict(options)
        self.path = prepare(options, download=False)["checkpoint"]
        self._contract: ComponentState | None = None

    def preflight(self, atoms: Atoms) -> None:
        _validate_atoms(atoms)

    @contextmanager
    def _lease(self) -> Iterator[tuple[Calculator, ComponentState]]:
        require_package("calorine", VERSION, "nep")
        import _nepy
        from calorine.calculators import CPUNEP

        before = _sha256(self.path)
        checkpoint = self.path

        class PeriodicCPUNEP(CPUNEP):
            def _setup_nepy(self):
                # The native backend reads weights lazily, and rereads them when atom
                # counts change. Bind every such load to this lease's checkpoint bytes.
                try:
                    if _sha256(checkpoint) != before:
                        raise ValueError("NEP checkpoint changed before native model loading")
                    super()._setup_nepy()
                    if _sha256(checkpoint) != before:
                        raise ValueError("NEP checkpoint changed during native model loading")
                except Exception:
                    # A failed post-load check must not leave a usable native model.
                    self.nepy = None
                    self._nepy_atoms = None
                    raise

            def check_state(self, atoms, tol=1e-15):
                _validate_atoms(atoms)
                return super().check_state(atoms, tol=tol)

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                _validate_atoms(atoms if atoms is not None else self.atoms)
                super().calculate(atoms, properties, system_changes)
                self.results["free_energy"] = self.results["energy"]

        calculator = PeriodicCPUNEP(str(self.path))
        if calculator.model_type != "potential":
            raise ValueError("ReactionFlow requires a NEP energy model, not a tensor/charge model")
        calculator.implemented_properties = [*calculator.implemented_properties, "free_energy"]
        if _sha256(self.path) != before:
            raise ValueError("NEP checkpoint changed during calculator construction")
        state = ComponentState(
            kind="reactionflow.nep",
            metadata={
                "options": self.options,
                "model_files": {str(self.path): before},
                "calorine_version": VERSION,
                "native_library_sha256": _sha256(Path(_nepy.__file__)),
                "calculator_source_sha256": _sha256(Path(inspect.getfile(CPUNEP))),
                "adapter_source_sha256": _sha256(Path(__file__)),
                "runtime_source_sha256": _sha256(Path(__file__).with_name("ase.py")),
                "integrator_source_sha256": _sha256(Path(__file__).parents[1] / "ase_npt.py"),
                "python_version": platform.python_version(),
                "ase_version": ase.__version__,
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "device": "cpu",
                "dtype": "float64",
            },
        )
        if self._contract is not None and state != self._contract:
            raise ValueError("NEP model or execution environment changed during the run")
        self._contract = state
        yield calculator, state


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> NEPAdapter:
    return NEPAdapter(trajectory=trajectory, options=options)
