"""SevenNet 0.13.0 checkpoints with explicit modality and offline inference."""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

from ..campaign import TrajectorySpec
from ._model_files import cached_download, model_cache, require_package
from ._torch import TorchModelAdapter

VERSION = "0.13.0"
_BASE = "https://github.com/MDIL-SNU/SevenNet/releases/download/"
# Paths and release URLs from the pinned wheel's sevenn._const registry. Resolving
# through package metadata keeps preparation independent of Torch/model imports.
_BUNDLED = {
    "7net-0": "SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth",
    "7net-0_22may2024": "SevenNet_0__22May2024/checkpoint_sevennet_0.pth",
    "7net-l3i5": "SevenNet_l3i5/checkpoint_l3i5.pth",
    "7net-mf-0": "SevenNet_MF_0/checkpoint_sevennet_mf_0.pth",
}
_DOWNLOADS = {
    "7net-mf-ompa": "v0.11.0.cp/checkpoint_sevennet_mf_ompa.pth",
    "7net-omat": "v0.11.0.cp/checkpoint_sevennet_omat.pth",
    "7net-omni": "v0.12.0.cp/checkpoint_sevennet_omni.pth",
    "7net-omni-i8": "v0.12.1.cp/checkpoint_sevennet_omni_i8.pth",
    "7net-omni-i12": "v0.12.1.cp/checkpoint_sevennet_omni_i12.pth",
}
_OMNI_MODALS = (
    "mpa",
    "omat24",
    "matpes_pbe",
    "matpes_r2scan",
    "mp_r2scan",
    "oc20",
    "oc22",
    "odac23",
    "omol25_low",
    "omol25_high",
    "spice",
    "qcml",
    "pet_mad",
)
_MODALS = {
    "7net-mf-0": ("PBE", "R2SCAN"),
    "7net-mf-ompa": ("mpa", "omat24"),
    "7net-omni": _OMNI_MODALS,
    "7net-omni-i8": _OMNI_MODALS,
    "7net-omni-i12": _OMNI_MODALS,
}
_OPTIONS = {"model", "checkpoint", "modal", "device", "cache_dir"}


def catalog() -> dict[str, Any]:
    return {
        "backend": "sevennet",
        "factory": "reactionflow.adapters.sevennet:create_adapter",
        "package": f"sevenn=={VERSION}",
        "models": [
            {
                "name": name,
                "description": (
                    "Cross-domain multitask model" if "omni" in name else "Materials model"
                ),
                "modals": list(_MODALS.get(name, ())),
                "bundled": name in _BUNDLED,
            }
            for name in (*_BUNDLED, *_DOWNLOADS)
        ],
        "options": sorted(_OPTIONS),
        "notes": [
            "Multitask models require an explicit modal; the loaded checkpoint must contain it.",
            "Uses float32 e3nn inference with conservative forces/stress and no extra D3.",
            "Charge/spin are not explicit calculator inputs; modal selects training fidelity.",
        ],
        "sources": [
            "https://github.com/MDIL-SNU/SevenNet",
            "https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html",
        ],
    }


def _validate(options: Mapping[str, Any]) -> None:
    unknown = set(options) - _OPTIONS
    if unknown:
        raise ValueError(f"unknown SevenNet options: {sorted(unknown)}")
    if ("model" in options) == ("checkpoint" in options):
        raise ValueError("SevenNet requires exactly one of model or checkpoint")
    if "model" in options:
        model = options["model"]
        if not isinstance(model, str) or model not in (*_BUNDLED, *_DOWNLOADS):
            raise ValueError(
                f"unknown SevenNet model; choose from {list((*_BUNDLED, *_DOWNLOADS))}"
            )
        modals = _MODALS.get(model, ())
        if modals and options.get("modal") not in modals:
            raise ValueError(f"SevenNet {model} requires explicit modal from {list(modals)}")
        if not modals and "modal" in options:
            raise ValueError(f"SevenNet {model} has no modal; omit that option")
    if "modal" in options and (not isinstance(options["modal"], str) or not options["modal"]):
        raise ValueError("SevenNet modal must be a nonempty string")
    for name in ("checkpoint", "cache_dir"):
        if name in options:
            value = options[name]
            if not isinstance(value, str) or not value or not Path(value).is_absolute():
                raise ValueError(f"SevenNet {name} must be an absolute path")
    if options.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("SevenNet device must be cpu or cuda")


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    _validate(options)
    require_package("sevenn", VERSION, "sevennet")
    if "checkpoint" in options:
        path = Path(options["checkpoint"])
    elif options["model"] in _BUNDLED:
        path = Path(
            distribution("sevenn").locate_file(
                f"sevenn/pretrained_potentials/{_BUNDLED[options['model']]}"
            )
        )
    else:
        path = cached_download(
            _BASE + _DOWNLOADS[options["model"]],
            model_cache(options, "sevennet"),
            download=download,
        )
    if not path.is_file():
        raise FileNotFoundError(f"SevenNet checkpoint is missing: {path}")
    return {"checkpoint": path.resolve()}


class SevenNetAdapter(TorchModelAdapter):
    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        super().__init__(
            trajectory=trajectory,
            options=options,
            files=prepare(options, download=False),
            packages=("sevenn", "e3nn", "torch-geometric", "matscipy", "scipy"),
            kind="reactionflow.sevennet",
        )

    def _new_calculator(self) -> Calculator:
        require_package("sevenn", VERSION, "sevennet")
        overrides = [
            name
            for name in (
                "SEVENNET_ENABLE_CUEQ",
                "SEVENNET_ENABLE_FLASH",
                "SEVENNET_ENABLE_OEQ",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            )
            if os.environ.get(name) == "1"
        ]
        if overrides:
            raise ValueError(f"unset SevenNet acceleration overrides for exact runs: {overrides}")
        import torch

        # The native graph converter always produces float32 tensors.
        torch.set_default_dtype(torch.float32)
        from sevenn.calculator import SevenNetCalculator

        calculator = SevenNetCalculator(
            model=str(self.files["checkpoint"]),
            file_type="checkpoint",
            device=self.device,
            modal=self.options.get("modal"),
            enable_cueq=False,
            enable_flash=False,
            enable_oeq=False,
            compute_atomic_virial=False,
        )
        if calculator.modal != self.options.get("modal"):
            raise ValueError("SevenNet checkpoint did not select the requested modal")
        return calculator

    def _extra_metadata(self, calculator: Calculator) -> dict[str, Any]:
        return {"dtype": "float32", "backend": "e3nn", "resolved_modal": calculator.modal}


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> SevenNetAdapter:
    return SevenNetAdapter(trajectory=trajectory, options=options)
