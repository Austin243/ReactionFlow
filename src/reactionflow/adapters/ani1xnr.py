"""Pinned ANI-1xnr weights with offline calculator construction."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

from .._durable import file_digest
from ..campaign import TrajectorySpec
from ._model_files import model_cache, require_package
from ._torch import TorchModelAdapter

TORCH_VERSION = "2.11.0"
TORCHANI_VERSION = "2.8.4"
MODEL_REPOSITORY = "roitberg-group/ani1xnr"
MODEL_REVISION = "234bb748b853eeed456d55a0c90bf8e95ed4f392"
MODEL_FILENAME = "ani1xnr.pt"
MODEL_SHA256 = "beef541802e4cb3d23b6cfdfdf9df1e42ddd3805399fb15f01e275aba4f099b3"

_ALLOWED_OPTIONS = {"device", "dtype", "model_index", "strategy", "cache_dir"}


def catalog() -> dict[str, Any]:
    """Describe the pinned model without importing Torch or downloading weights."""

    return {
        "backend": "ani1xnr",
        "factory": "reactionflow.adapters.ani1xnr:create_adapter",
        "package": f"torchani=={TORCHANI_VERSION}",
        "models": [
            {
                "name": "ani1xnr",
                "description": "ANI-1xnr reactive potential for H, C, N and O; BLYP/TZV2P",
            }
        ],
        "notes": [
            "The adapter always loads the pinned weights; there is no model or checkpoint option.",
            "The model is an ensemble of eight networks; model_index selects one (default 0).",
            "dtype is float32 (default) or float64; strategy is pyaev (default) or cuaev.",
            f"Requires torch {TORCH_VERSION}.",
        ],
        "sources": [
            "https://www.nature.com/articles/s41557-023-01427-3",
            f"https://huggingface.co/{MODEL_REPOSITORY}",
        ],
    }


def _validate(options: Mapping[str, Any]) -> None:
    unknown = set(options) - _ALLOWED_OPTIONS
    if unknown:
        raise ValueError(f"unknown ANI-1xnr adapter options: {sorted(unknown)}")
    if options.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("ANI-1xnr device must be 'cpu' or 'cuda'")
    if options.get("dtype", "float32") not in ("float32", "float64"):
        raise ValueError("ANI-1xnr dtype must be 'float32' or 'float64'")
    model_index = options.get("model_index", 0)
    if isinstance(model_index, bool) or not isinstance(model_index, int) or model_index < 0:
        raise ValueError("ANI-1xnr model_index must be a non-negative integer")
    if options.get("strategy", "pyaev") not in ("pyaev", "cuaev"):
        raise ValueError("ANI-1xnr strategy must be 'pyaev' or 'cuaev'")


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    """Resolve the pinned weights, downloading them only when allowed, and verify them."""

    _validate(options)
    require_package("torchani", TORCHANI_VERSION, "ani1xnr")
    # TorchANI reads built-in state dicts from <data directory>/StateDicts.
    path = model_cache(options, "ani1xnr") / "StateDicts" / MODEL_FILENAME
    if not path.is_file():
        if not download:
            raise FileNotFoundError(
                "ANI-1xnr weights are not cached; run 'reactionflow prepare campaign.json' "
                "or 'reactionflow run campaign.json --download' first"
            )
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_dir=path.parent,
        )
    digest = file_digest(path)
    if digest != MODEL_SHA256:
        raise ValueError(
            f"ANI-1xnr weights failed SHA-256 verification: {path} has {digest}, "
            f"expected {MODEL_SHA256}"
        )
    return {"checkpoint": path}


class ANI1xnrAdapter(TorchModelAdapter):
    """One pinned ANI-1xnr model for MD and every refinement stage."""

    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        super().__init__(
            trajectory=trajectory,
            options=options,
            files=prepare(options, download=False),
            packages=("torchani",),
            kind="reactionflow.ani1xnr",
        )
        self.dtype = options.get("dtype", "float32")
        self.model_index = options.get("model_index", 0)
        self.strategy = options.get("strategy", "pyaev")

    def _new_calculator(self) -> Calculator:
        require_package("torchani", TORCHANI_VERSION, "ani1xnr")
        import torch
        from torchani.models import ANI1xnr

        if torch.__version__.split("+", 1)[0] != TORCH_VERSION:
            raise RuntimeError(
                f"ANI-1xnr requires torch {TORCH_VERSION}, found {torch.__version__}"
            )
        # TorchANI loads the state dict from its data directory when the model is built, so
        # point it at the hashed weights rather than at whatever the process inherited.
        os.environ["TORCHANI_DATA_DIR"] = str(self.files["checkpoint"].parents[1])
        model = ANI1xnr(
            model_index=self.model_index,
            strategy=self.strategy,
            periodic_table_index=True,
            device=self.device,
            dtype=getattr(torch, self.dtype),
        )
        return model.ase(stress_kind="scaling")

    def _extra_metadata(self, calculator: Calculator) -> dict[str, Any]:
        return {
            "model_revision": MODEL_REVISION,
            "model_index": self.model_index,
            "dtype": self.dtype,
            "strategy": self.strategy,
            "stress_kind": "scaling",
        }


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> ANI1xnrAdapter:
    return ANI1xnrAdapter(trajectory=trajectory, options=options)


__all__ = [
    "MODEL_FILENAME",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "MODEL_SHA256",
    "TORCHANI_VERSION",
    "TORCH_VERSION",
    "ANI1xnrAdapter",
    "catalog",
    "create_adapter",
    "prepare",
]
