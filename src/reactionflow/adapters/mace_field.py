"""Pinned MACE-Field fork with an explicit, constant electric field."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from importlib.metadata import distribution
from numbers import Real
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ..campaign import TrajectorySpec
from ._model_files import cached_download, model_cache, require_package
from ._torch import TorchModelAdapter
from .ase import _sha256

SOURCE_URL = "https://github.com/mdi-group/mace-field.git"
SOURCE_COMMIT = "136e4ef040d7c51a5b051a7a609ffb1f29307478"
INSTALL_REQUIREMENT = f"mace-torch @ git+{SOURCE_URL}@{SOURCE_COMMIT}"
_MODEL = "MACEField-MH-0-omat-dielectric"
_URL = f"https://github.com/mdi-group/mace-field/releases/download/1.0.2/{_MODEL}.model"
_SHA256 = "f92e043aaf2cd8879919db8452503553fe7b608cb749d8d169dd96d4aa094aa2"
_HEADS = ("pt_head", "mp-dielectric", "mp-ferroelectric")
_OPTIONS = {"model", "checkpoint", "head", "electric_field", "device", "dtype", "cache_dir"}


def catalog() -> dict[str, Any]:
    return {
        "backend": "mace_field",
        "factory": "reactionflow.adapters.mace_field:create_adapter",
        "package": INSTALL_REQUIREMENT,
        "models": [
            {
                "name": _MODEL,
                "description": "Inorganic dielectric/ferroelectric materials under a fixed field",
                "heads": list(_HEADS),
            }
        ],
        "options": sorted(_OPTIONS),
        "notes": [
            "Requires a separate environment: this fork replaces the mace-torch distribution.",
            "Explicit head and electric_field [Ex, Ey, Ez] in V/angstrom are required.",
            "The configured field overrides structure fields and stays fixed during MD/refinement.",
            "Energies are electric enthalpies at that field; NPT additionally includes PV.",
        ],
        "sources": ["https://github.com/mdi-group/mace-field/releases/tag/1.0.2"],
    }


def _require_fork() -> None:
    try:
        require_package("mace-torch", "0.3.15", "mace-field")
        require_package("e3nn", "0.4.4", "mace-field")
    except RuntimeError as error:
        raise RuntimeError(
            "MACE-Field needs its pinned fork and e3nn==0.4.4 in a separate environment; "
            "use 'reactionflow prepare campaign.json --install' or install "
            f"{INSTALL_REQUIREMENT!r}"
        ) from error
    try:
        origin = json.loads(distribution("mace-torch").read_text("direct_url.json") or "null")
        matches = (
            origin["url"].removesuffix(".git") == SOURCE_URL.removesuffix(".git")
            and origin["vcs_info"]["vcs"] == "git"
            and origin["vcs_info"]["commit_id"] == SOURCE_COMMIT
        )
    except (TypeError, KeyError, ValueError, AttributeError):
        matches = False
    if not matches:
        raise RuntimeError(
            "MACE-Field requires its pinned Git source, not the standard mace-torch package; "
            f"install {INSTALL_REQUIREMENT!r} in a separate environment"
        )


def _validate(options: Mapping[str, Any]) -> None:
    unknown = set(options) - _OPTIONS
    if unknown:
        raise ValueError(f"unknown MACE-Field options: {sorted(unknown)}")
    if ("model" in options) == ("checkpoint" in options):
        raise ValueError("MACE-Field requires exactly one of model or checkpoint")
    if "model" in options and options["model"] != _MODEL:
        raise ValueError(f"unknown MACE-Field model; choose {_MODEL!r}")
    head = options.get("head")
    if not isinstance(head, str) or not head:
        raise ValueError("MACE-Field requires an explicit head")
    if "model" in options and head not in _HEADS:
        raise ValueError(f"unknown released MACE-Field head; choose from {list(_HEADS)}")
    field = options.get("electric_field")
    if (
        not isinstance(field, Sequence)
        or isinstance(field, (str, bytes))
        or len(field) != 3
        or any(
            isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in field
        )
    ):
        raise ValueError("MACE-Field requires electric_field as three finite numbers in V/angstrom")
    for name in ("checkpoint", "cache_dir"):
        if name in options:
            value = options[name]
            if not isinstance(value, str) or not value or not Path(value).is_absolute():
                raise ValueError(f"MACE-Field {name} must be an absolute path")
    if options.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("MACE-Field device must be cpu or cuda")
    if options.get("dtype", "float64") not in ("float32", "float64"):
        raise ValueError("MACE-Field dtype must be float32 or float64")


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    _validate(options)
    _require_fork()
    if "checkpoint" in options:
        path = Path(options["checkpoint"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MACE-Field checkpoint does not exist: {path}")
    else:
        path = cached_download(_URL, model_cache(options, "mace-field"), download=download)
        if _sha256(path) != _SHA256:
            raise ValueError("released MACE-Field checkpoint checksum differs")
    return {"checkpoint": path}


def _validate_atoms(atoms: Atoms) -> None:
    if not math.isfinite(atoms.cell.volume) or atoms.cell.volume <= 0:
        raise ValueError("MACE-Field requires a nonzero cell volume for dielectric observables")


class MACEFieldAdapter(TorchModelAdapter):
    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        super().__init__(
            trajectory=trajectory,
            options=options,
            files=prepare(options, download=False),
            packages=("mace-torch", "e3nn"),
            kind="reactionflow.mace-field",
        )

    def preflight(self, atoms: Atoms) -> None:
        _validate_atoms(atoms)

    def _new_calculator(self) -> Calculator:
        _require_fork()
        import torch

        # This fork's __init__ changes Torch's process-wide pickle-loading policy.
        # Restore the caller's policy and load the selected local model explicitly.
        previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        try:
            from mace.calculators import MACECalculator
        finally:
            if previous is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous
        model = torch.load(self.files["checkpoint"], map_location=self.device, weights_only=False)
        dtype = getattr(torch, self.options.get("dtype", "float64"))

        class FixedPrecisionFieldCalculator(MACECalculator):
            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                _validate_atoms(atoms if atoms is not None else self.atoms)
                # Unlike current MACE, the fork reads the global dtype while forming
                # atomic graphs and the field tensor. Scope it to this evaluation.
                previous_dtype = torch.get_default_dtype()
                try:
                    torch.set_default_dtype(dtype)
                    return super().calculate(atoms, properties, system_changes)
                finally:
                    torch.set_default_dtype(previous_dtype)

        calculator = FixedPrecisionFieldCalculator(
            models=[model],
            model_type="MACEField",
            device=self.device,
            default_dtype=self.options.get("dtype", "float64"),
            head=self.options["head"],
            electric_field=[float(x) for x in self.options["electric_field"]],
            compile_mode=None,
            enable_cueq=False,
            enable_oeq=False,
        )
        if calculator.head != self.options["head"]:
            raise ValueError(f"MACE-Field did not select requested head {self.options['head']!r}")
        return calculator

    def _extra_metadata(self, calculator: Calculator) -> dict[str, Any]:
        return {
            "source_commit": SOURCE_COMMIT,
            "resolved_head": calculator.head,
            "electric_field_units": "V/angstrom",
        }


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> MACEFieldAdapter:
    return MACEFieldAdapter(trajectory=trajectory, options=options)


__all__ = ["MACEFieldAdapter", "catalog", "create_adapter", "prepare"]
