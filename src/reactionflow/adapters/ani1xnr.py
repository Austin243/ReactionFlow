"""Pinned ANI-1xnr calculator and exactly restartable ASE NPT runtime."""

from __future__ import annotations

import hashlib
import os
import platform
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

from ..campaign import TrajectorySpec
from ..restart import ComponentState
from .ase import ASELangevinBAOABAdapter

TORCH_VERSION = "2.11.0"
TORCHANI_VERSION = "2.8.4"
MODEL_REPOSITORY = "roitberg-group/ani1xnr"
MODEL_REVISION = "234bb748b853eeed456d55a0c90bf8e95ed4f392"
MODEL_FILENAME = "ani1xnr.pt"
MODEL_SHA256 = "beef541802e4cb3d23b6cfdfdf9df1e42ddd3805399fb15f01e275aba4f099b3"

_CALCULATOR_KIND = "reactionflow.ani1xnr"
_ALLOWED_OPTIONS = {"device", "dtype", "model_index", "strategy"}


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _source_sha256(path: Path) -> str:
    return _sha256(path.resolve())


def _package_sha256(root: Path) -> str:
    checksum = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        checksum.update(str(path.relative_to(root)).encode())
        checksum.update(b"\0")
        checksum.update(path.read_bytes())
        checksum.update(b"\0")
    return checksum.hexdigest()


def _base_version(value: object) -> str:
    return str(value).split("+", 1)[0]


class ANI1xnrAdapter(ASELangevinBAOABAdapter):
    """One pinned ANI-1xnr model with Langevin BAOAB NVT/NPT dynamics."""

    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        unknown_options = set(options) - _ALLOWED_OPTIONS
        if unknown_options:
            raise ValueError(f"unknown ANI-1xnr adapter options: {sorted(unknown_options)}")
        super().__init__(trajectory=trajectory)
        self.device = str(options.get("device", "cuda"))
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("ANI-1xnr device must be 'cpu' or 'cuda'")
        self.dtype = str(options.get("dtype", "float32"))
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("ANI-1xnr dtype must be 'float32' or 'float64'")
        model_index = options.get("model_index", 0)
        if isinstance(model_index, bool) or not isinstance(model_index, int) or model_index < 0:
            raise ValueError("ANI-1xnr model_index must be a non-negative integer")
        self.model_index = model_index
        self.strategy = str(options.get("strategy", "pyaev"))
        if self.strategy not in {"pyaev", "cuaev"}:
            raise ValueError("ANI-1xnr strategy must be 'pyaev' or 'cuaev'")

    def _model_path(self) -> Path:
        data_dir = os.environ.get("TORCHANI_DATA_DIR")
        if not data_dir:
            raise RuntimeError(
                "TORCHANI_DATA_DIR must point to the pinned ANI-1xnr cache; "
                "run scripts/setup-perlmutter-ani1xnr.sh"
            )
        path = Path(data_dir).expanduser().resolve() / "StateDicts" / MODEL_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"pinned ANI-1xnr weights are missing: {path}; run the setup script"
            )
        digest = _sha256(path)
        if digest != MODEL_SHA256:
            raise ValueError(
                f"ANI-1xnr weights failed SHA-256 verification: {digest} != {MODEL_SHA256}"
            )
        return path

    def _load_backend(self) -> tuple[Any, Any]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
            raise RuntimeError("exact ANI restart requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        try:
            import torch
            from torchani.models import ANI1xnr
        except ImportError as error:
            raise RuntimeError(
                "ANI-1xnr support requires torch 2.11.0 and torchani 2.8.4; "
                "install the 'ani1xnr' extra or run the Perlmutter setup script"
            ) from error
        installed_torchani = version("torchani")
        if _base_version(torch.__version__) != TORCH_VERSION:
            raise RuntimeError(
                f"ANI-1xnr requires torch {TORCH_VERSION}, found {torch.__version__}"
            )
        if installed_torchani != TORCHANI_VERSION:
            raise RuntimeError(
                f"ANI-1xnr requires torchani {TORCHANI_VERSION}, found {installed_torchani}"
            )
        if self.device == "cuda" and (
            not torch.cuda.is_available() or torch.cuda.device_count() != 1
        ):
            raise RuntimeError("each ANI-1xnr worker must see exactly one usable CUDA device")
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return torch, ANI1xnr

    def _calculator_state(self) -> ComponentState:
        torch, _ = self._load_backend()
        model_path = self._model_path()
        device_metadata: dict[str, object] = {}
        if self.device == "cuda":
            properties = torch.cuda.get_device_properties(0)
            device_metadata = {
                "cuda_version": str(torch.version.cuda),
                "device_name": str(properties.name),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
        return ComponentState(
            kind=_CALCULATOR_KIND,
            metadata={
                "torch_version": _base_version(torch.__version__),
                "torchani_version": version("torchani"),
                "model_repository": MODEL_REPOSITORY,
                "model_revision": MODEL_REVISION,
                "model_filename": MODEL_FILENAME,
                "model_sha256": _sha256(model_path),
                "model_index": self.model_index,
                "device": self.device,
                "dtype": self.dtype,
                "strategy": self.strategy,
                "deterministic_algorithms": True,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "allow_tf32": False,
                "python_version": platform.python_version(),
                "reactionflow_source_sha256": _package_sha256(Path(__file__).parents[1]),
                "adapter_source_sha256": _source_sha256(Path(__file__)),
                "ase_npt_source_sha256": _source_sha256(Path(__file__).parents[1] / "ase_npt.py"),
                **device_metadata,
            },
        )

    def _new_calculator(self, torch: Any, factory: Any) -> Calculator:
        dtype = torch.float32 if self.dtype == "float32" else torch.float64
        model = factory(
            model_index=self.model_index,
            strategy=self.strategy,
            periodic_table_index=True,
            device=self.device,
            dtype=dtype,
        )
        return model.ase(stress_kind="scaling")

    @contextmanager
    def _lease(self) -> Iterator[tuple[Calculator, ComponentState]]:
        torch, factory = self._load_backend()
        state = self._calculator_state()
        calculator = self._new_calculator(torch, factory)
        try:
            yield calculator, state
        finally:
            del calculator
            if self.device == "cuda":
                torch.cuda.empty_cache()


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> ANI1xnrAdapter:
    """Campaign factory for the pinned built-in ANI-1xnr adapter."""

    return ANI1xnrAdapter(trajectory=trajectory, options=options)


__all__ = [
    "MODEL_FILENAME",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "MODEL_SHA256",
    "TORCHANI_VERSION",
    "TORCH_VERSION",
    "ANI1xnrAdapter",
    "create_adapter",
]
