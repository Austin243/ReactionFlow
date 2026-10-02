"""Explicit MACE checkpoints with offline calculator construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

from ..campaign import TrajectorySpec
from ._model_files import cached_download, model_cache, require_package
from ._torch import TorchModelAdapter

MACE_VERSION = "0.3.16"

_ALLOWED_OPTIONS = {"model", "family", "checkpoint", "device", "dtype", "head", "cache_dir"}


def _validate(options: Mapping[str, Any]) -> None:
    unknown = set(options) - _ALLOWED_OPTIONS
    if unknown:
        raise ValueError(f"unknown MACE adapter options: {sorted(unknown)}")
    if ("model" in options) == ("checkpoint" in options):
        raise ValueError("MACE requires exactly one of 'model' or 'checkpoint'")
    family = options.get("family", "mp")
    if family not in ("mp", "off"):
        raise ValueError("MACE family must be 'mp' or 'off'")
    if "model" in options:
        model = options["model"]
        if not isinstance(model, str) or not model:
            raise ValueError("MACE model must be an explicit non-empty model name")
    for name in ("checkpoint", "cache_dir"):
        if name in options:
            value = options[name]
            if not isinstance(value, str) or not value or not Path(value).is_absolute():
                raise ValueError(f"MACE {name} must be an absolute path")
    if options.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("MACE device must be 'cpu' or 'cuda'")
    if options.get("dtype", "float64") not in ("float32", "float64"):
        raise ValueError("MACE dtype must be 'float32' or 'float64'")
    if "head" in options and (not isinstance(options["head"], str) or not options["head"]):
        raise ValueError("MACE head must be a non-empty string")


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    """Resolve a local checkpoint, downloading a named model only when allowed."""

    _validate(options)
    require_package("mace-torch", MACE_VERSION, "mace")
    require_package("e3nn", "0.4.4", "mace")
    if "checkpoint" in options:
        path = Path(options["checkpoint"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MACE checkpoint does not exist: {path}")
    else:
        from mace.calculators.foundations_models import mace_mp_urls, mace_off_urls

        family = options.get("family", "mp")
        registry = mace_mp_urls if family == "mp" else mace_off_urls
        model = options["model"]
        if model not in registry:
            raise ValueError(
                f"unknown MACE {family} model {model!r}; choose from {sorted(registry)}"
            )
        url = registry[model]
        path = cached_download(url, model_cache(options, "mace"), download=download)
    return {"checkpoint": path}


class MACEAdapter(TorchModelAdapter):
    """One pinned MACE calculator for every trajectory and pathway stage."""

    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        super().__init__(
            trajectory=trajectory,
            options=options,
            files=prepare(options, download=False),
            packages=("mace-torch", "e3nn"),
            kind="reactionflow.mace",
        )

    def _new_calculator(self) -> Calculator:
        require_package("mace-torch", MACE_VERSION, "mace")
        require_package("e3nn", "0.4.4", "mace")
        from mace.calculators import MACECalculator

        kwargs = {
            "model_paths": [str(self.files["checkpoint"])],
            "device": self.device,
            "default_dtype": self.options.get("dtype", "float64"),
        }
        if "head" in self.options:
            kwargs["head"] = self.options["head"]
        calculator = MACECalculator(**kwargs)
        if "head" in self.options and calculator.head != self.options["head"]:
            raise ValueError(
                f"MACE did not select requested head {self.options['head']!r}; "
                f"resolved to {calculator.head!r}"
            )
        return calculator

    def _extra_metadata(self, calculator: Calculator) -> dict[str, Any]:
        return {"resolved_head": calculator.head}


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> MACEAdapter:
    return MACEAdapter(trajectory=trajectory, options=options)
