"""Prepared AIMNet2 family checkpoints with explicit electronic-state validation."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import distribution
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from .._durable import file_digest
from ..campaign import TrajectorySpec
from ._model_files import cached_download, model_cache, require_package
from ._torch import TorchModelAdapter

AIMNET_VERSION = "0.2.0"
_FAMILIES = {
    "aimnet2": "General organic and main-group chemistry, wB97M-D3",
    "aimnet2-2025": "B97-3c with improved intermolecular interactions",
    "aimnet2-b973c": "Legacy B97-3c reference for reproducibility",
    "aimnet2-nse": "Open-shell molecules and radicals",
    "aimnet2-pd": "Palladium chemistry, B97-3c/CPCM(THF) reference",
    "aimnet2-rxn": "Reactive paths and transition states; neutral H/C/N/O systems",
}
_OPTIONS = {"model", "model_index", "checkpoint", "device", "cache_dir", "coulomb_method"}


def catalog() -> dict[str, Any]:
    """Describe supported choices without importing or installing AIMNet."""

    return {
        "backend": "aimnet2",
        "factory": "reactionflow.adapters.aimnet2:create_adapter",
        "package": f"aimnet[ase]=={AIMNET_VERSION}",
        "models": [
            {"name": name, "description": domain, "model_index": [0, 1, 2, 3]}
            for name, domain in _FAMILIES.items()
        ],
        "options": {
            "model": "One listed family, or use an absolute local checkpoint path instead",
            "checkpoint": "Absolute path to a current AIMNet .pt checkpoint",
            "model_index": "Ensemble member 0-3; default 0, only with model",
            "device": "cpu or cuda; default cuda",
            "cache_dir": "Optional absolute model cache root",
            "coulomb_method": "auto (default), dsf, ewald, or pme",
        },
        "limitations": [
            "Explicit integer atoms.info charge and positive spin multiplicity are required",
            "Only aimnet2-nse supports spin multiplicities above one; RXN requires net charge zero",
            "AIMNet ASE uses float32 inputs; this adapter does not expose a dtype option",
            "auto uses simple molecular Coulomb and DSF for periodic structures",
            "NPT requires a fully periodic nonzero cell and working finite stress predictions",
            "Energy reference scales differ across model families",
        ],
        "sources": [
            "https://github.com/isayevlab/aimnetcentral",
            "https://huggingface.co/isayevlab/aimnet2-rxn",
        ],
    }


def _validate_options(options: Mapping[str, Any]) -> None:
    unknown = set(options) - _OPTIONS
    if unknown:
        raise ValueError(f"unknown AIMNet2 adapter options: {sorted(unknown)}")
    if ("model" in options) == ("checkpoint" in options):
        raise ValueError("AIMNet2 requires exactly one of model or checkpoint")
    if "model" in options:
        name = options["model"]
        if not isinstance(name, str) or name not in _FAMILIES:
            raise ValueError(f"unknown AIMNet2 model {name!r}; choose from {sorted(_FAMILIES)}")
        index = options.get("model_index", 0)
        if isinstance(index, bool) or not isinstance(index, int) or index not in range(4):
            raise ValueError("AIMNet2 model_index must be an integer from 0 to 3")
    elif "model_index" in options:
        raise ValueError("AIMNet2 model_index is only valid with a named model")
    for name in ("checkpoint", "cache_dir"):
        if name in options:
            value = options[name]
            if not isinstance(value, str) or not value or not Path(value).is_absolute():
                raise ValueError(f"AIMNet2 {name} must be an absolute path string")
    if options.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("AIMNet2 device must be cpu or cuda")
    if options.get("coulomb_method", "auto") not in ("auto", "dsf", "ewald", "pme"):
        raise ValueError("AIMNet2 coulomb_method must be auto, dsf, ewald, or pme")


def _package_file(relative: str) -> Path:
    return Path(distribution("aimnet").locate_file(f"aimnet/{relative}")).resolve()


def _registry() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(_package_file("calculators/model_registry.yaml").read_text())


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    """Resolve one model and its packaged D3 reference data without loading a predictor."""

    _validate_options(options)
    require_package("aimnet", AIMNET_VERSION, "aimnet2")
    if "checkpoint" in options:
        checkpoint = Path(options["checkpoint"]).resolve()
    else:
        registry = _registry()
        family = registry["aliases"][options["model"]].rsplit("_", 1)[0]
        key = f"{family}_{options.get('model_index', 0)}"
        spec = registry["models"][key]
        checkpoint = cached_download(
            spec["url"], model_cache(options, "aimnet2"), download=download
        )
        if spec.get("sha256") and file_digest(checkpoint) != spec["sha256"]:
            raise ValueError(f"AIMNet2 checkpoint differs from the published SHA-256 for {key}")
    files = {"checkpoint": checkpoint, "dftd3": _package_file("dftd3_data.pt")}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"AIMNet2 {name} file is missing: {path}")
    return files


def _electronic_state(atoms: Atoms, predictor: Any) -> None:
    for name in ("charge", "spin"):
        value = atoms.info.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"AIMNet2 requires explicit integer atoms.info[{name!r}]")
    spin = atoms.info["spin"]
    if spin < 1:
        raise ValueError("AIMNet2 spin must be a positive multiplicity")
    if "mult" in atoms.info and (
        isinstance(atoms.info["mult"], bool)
        or not isinstance(atoms.info["mult"], Integral)
        or atoms.info["mult"] != spin
    ):
        raise ValueError("AIMNet2 atoms.info['mult'] must agree with 'spin'")
    if not predictor.is_nse and spin != 1:
        raise ValueError(
            "this AIMNet2 model supports singlets only; use aimnet2-nse for open shells"
        )
    metadata = predictor.metadata or {}
    if metadata.get("supports_charged_systems") is False and atoms.info["charge"] != 0:
        raise ValueError("this AIMNet2 model requires net charge zero")


class AIMNet2Adapter(TorchModelAdapter):
    """One offline AIMNet2 model for MD and every refinement stage."""

    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        super().__init__(
            trajectory=trajectory,
            options=options,
            files=prepare(options, download=False),
            packages=("aimnet", "warp-lang", "nvalchemi-toolkit-ops"),
            kind="reactionflow.aimnet2",
        )

    def _new_calculator(self) -> Calculator:
        require_package("aimnet", AIMNET_VERSION, "aimnet2")
        import torch

        # AIMNet's ASE interface and model input tensors use float32.
        torch.set_default_dtype(torch.float32)
        from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

        predictor = AIMNet2Calculator(
            str(self.files["checkpoint"]), device=self.device, compile_model=False, train=False
        )
        if (predictor.metadata or {}).get("format_version") != 2:
            raise ValueError("AIMNet2 requires a current .pt checkpoint with model metadata")
        if self.options.get("model") == "aimnet2-rxn" and predictor.metadata.get("family") != "rxn":
            raise ValueError("AIMNet2-RXN checkpoint is missing its reactive-family metadata")
        method = self.options.get("coulomb_method", "auto")

        class ExplicitAIMNet2ASE(AIMNet2ASE):
            def check_state(self, atoms, tol=1e-15):
                _electronic_state(atoms, self.base_calc)
                return super().check_state(atoms, tol=tol)

            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                atoms = atoms if atoms is not None else self.atoms
                _electronic_state(atoms, self.base_calc)
                chosen = ("dsf" if atoms.pbc.any() else "simple") if method == "auto" else method
                if chosen in ("ewald", "pme") and not atoms.pbc.all():
                    raise ValueError("AIMNet2 Ewald/PME requires a fully periodic structure")
                if (
                    chosen != "simple"
                    and (self.base_calc.metadata or {}).get("coulomb_mode") == "full_embedded"
                ):
                    raise ValueError(
                        "this AIMNet2 checkpoint cannot change its embedded Coulomb method"
                    )
                if self.base_calc.has_external_coulomb:
                    self.base_calc.set_lrcoulomb_method(chosen)
                return super().calculate(atoms, properties, system_changes)

        return ExplicitAIMNet2ASE(predictor, validate_species=True)

    def _extra_metadata(self, calculator: Calculator) -> dict[str, Any]:
        return {
            "model_metadata": dict(calculator.base_calc.metadata),
            "input_dtype": "float32",
            "compile_model": False,
            "coulomb_method": self.options.get("coulomb_method", "auto"),
        }

    def preflight(self, atoms: Atoms) -> None:
        """Check actual inference capabilities before a new trajectory is published."""

        if self.trajectory.pressure_GPa is not None and (
            not atoms.pbc.all() or atoms.cell.rank != 3 or atoms.get_volume() <= 0
        ):
            raise ValueError("AIMNet2 NPT requires a fully periodic nonzero cell")
        with self.calculator("preflight") as calculator:
            probe = atoms.copy()
            probe.calc = calculator
            values = [probe.get_potential_energy(), probe.get_forces()]
            if self.trajectory.pressure_GPa is not None:
                values.append(probe.get_stress())
            if any(not np.isfinite(value).all() for value in values):
                raise ValueError("AIMNet2 preflight returned non-finite energy, forces, or stress")


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> AIMNet2Adapter:
    return AIMNet2Adapter(trajectory=trajectory, options=options)
