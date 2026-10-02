"""Conservative ORB models with explicit preparation and local-only inference."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ..campaign import TrajectorySpec
from ._model_files import cached_download, model_cache, require_package
from ._torch import TorchModelAdapter

ORB_VERSION = "0.7.0"
_BASE = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/"
_MODELS = {
    "orbmol-v2": ("orbmol_v2", "orbmol-v2-teqabfhg-20260523.ckpt"),
    "orb-v3-conservative-inf-omat": (
        "orb_v3_conservative_inf_omat",
        "orb-v3/orb-v3-conservative-inf-omat-20250404.ckpt",
    ),
    "orb-v3-conservative-inf-mpa": (
        "orb_v3_conservative_inf_mpa",
        "orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt",
    ),
}
_OPTIONS = {"model", "checkpoint", "device", "precision", "cache_dir"}


def catalog() -> dict[str, Any]:
    """Describe the pinned supported models without importing ORB or Torch."""

    return {
        "backend": "orb",
        "factory": "reactionflow.adapters.orb:create_adapter",
        "package": f"orb-models=={ORB_VERSION}",
        "models": [
            {"name": "orbmol-v2", "description": "Molecules/polymers (OMol25 + OPoly26)"},
            {"name": "orb-v3-conservative-inf-omat", "description": "Inorganic materials (OMat24)"},
            {
                "name": "orb-v3-conservative-inf-mpa",
                "description": "Materials (MPtraj + Alexandria)",
            },
        ],
        "options": sorted(_OPTIONS),
        "precision": ["float32-highest", "float64"],
        "notes": "Conservative forces only; model required even with a local checkpoint. "
        "OrbMol requires atoms.info integer charge and positive spin multiplicity.",
        "sources": ["https://github.com/orbital-materials/orb-models/blob/v0.7.0/MODELS.md"],
    }


def _validate(options: Mapping[str, Any]) -> None:
    unknown = set(options) - _OPTIONS
    if unknown:
        raise ValueError(f"unknown ORB adapter options: {sorted(unknown)}")
    model = options.get("model")
    if not isinstance(model, str) or model not in _MODELS:
        raise ValueError(f"ORB model must name a supported conservative model: {sorted(_MODELS)}")
    for name in ("checkpoint", "cache_dir"):
        if name in options:
            value = options[name]
            if not isinstance(value, str) or not value or not Path(value).is_absolute():
                raise ValueError(f"ORB {name} must be an absolute path")
    if options.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("ORB device must be 'cpu' or 'cuda'")
    if options.get("precision", "float32-highest") not in ("float32-highest", "float64"):
        raise ValueError("ORB precision must be 'float32-highest' or 'float64'")


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    """Resolve/download the selected weight file without constructing a model."""

    _validate(options)
    require_package("orb-models", ORB_VERSION, "orb")
    if "checkpoint" in options:
        path = Path(options["checkpoint"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ORB checkpoint does not exist: {path}")
    else:
        path = cached_download(
            _BASE + _MODELS[options["model"]][1], model_cache(options, "orb"), download=download
        )
    return {"checkpoint": path}


def _validate_molecule(atoms: Atoms) -> None:
    for name in ("charge", "spin"):
        value = atoms.info.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"OrbMol requires explicit integer atoms.info[{name!r}]")
        if name == "spin" and value < 1:
            raise ValueError("OrbMol spin must be a positive multiplicity")
    if atoms.pbc.any() and not atoms.pbc.all():
        raise ValueError("OrbMol-v2 supports only fully periodic or fully nonperiodic systems")


class _PrecisionAtomsAdapter:
    def __init__(self, adapter: Any, dtype: Any) -> None:
        self.adapter = adapter
        self.dtype = dtype

    def from_ase_atoms(self, **kwargs: Any) -> Any:
        # ORB otherwise reads the process default dtype on every evaluation. Model
        # construction restores that global setting, so bind graph precision explicitly.
        return self.adapter.from_ase_atoms(
            **kwargs, output_dtype=self.dtype, graph_construction_dtype=self.dtype
        )


class ORBAdapter(TorchModelAdapter):
    """One conservative ORB model for MD and every refinement stage."""

    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        super().__init__(
            trajectory=trajectory,
            options=options,
            files=prepare(options, download=False),
            packages=("orb-models", "nvalchemi-toolkit-ops", "warp-lang", "scipy"),
            kind="reactionflow.orb",
        )

    def preflight(self, atoms: Atoms) -> None:
        if self.options["model"] == "orbmol-v2":
            _validate_molecule(atoms)

    def _new_calculator(self) -> Calculator:
        require_package("orb-models", ORB_VERSION, "orb")
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

        constructor = getattr(pretrained, _MODELS[self.options["model"]][0])
        model, atoms_adapter = constructor(
            weights_path=str(self.files["checkpoint"]),
            device=self.device,
            precision=self.options.get("precision", "float32-highest"),
            compile=False,
        )
        model.enable_stress()
        atoms_adapter = _PrecisionAtomsAdapter(atoms_adapter, next(model.parameters()).dtype)
        molecular = self.options["model"] == "orbmol-v2"

        class CheckedORBCalculator(ORBCalculator):
            def check_state(self, atoms, tol=1e-15):
                if molecular:
                    _validate_molecule(atoms)
                changes = super().check_state(atoms, tol=tol)
                if molecular and self.atoms is not None:
                    changes.extend(
                        name
                        for name in ("charge", "spin")
                        if self.atoms.info.get(name) != atoms.info.get(name)
                    )
                return changes

            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                if molecular:
                    _validate_molecule(atoms if atoms is not None else self.atoms)
                return super().calculate(atoms, properties, system_changes)

        calculator = CheckedORBCalculator(
            model,
            atoms_adapter,
            device=self.device,
            edge_method="knn_alchemi",
            half_supercell=False,
        )
        if not calculator.conservative:
            raise ValueError("ReactionFlow requires conservative ORB energy-gradient forces")
        return calculator

    def _extra_metadata(self, calculator: Calculator) -> dict[str, Any]:
        return {"compile": False, "edge_method": "knn_alchemi", "half_supercell": False}


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> ORBAdapter:
    return ORBAdapter(trajectory=trajectory, options=options)


__all__ = ["ORBAdapter", "catalog", "create_adapter", "prepare"]
