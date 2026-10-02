"""Shared deterministic inference and restart identity for optional Torch models."""

from __future__ import annotations

import inspect
import os
import platform
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import ase
import numpy as np
from ase.calculators.calculator import Calculator

from ..campaign import TrajectorySpec
from ..restart import ComponentState
from .ase import ASELangevinBAOABAdapter, _sha256


def _load_torch(device: str) -> Any:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise RuntimeError("deterministic model inference requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install the selected model extra before running it") from error
    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("each model worker must see exactly one usable CUDA device")
    if device == "cuda":
        torch.cuda.init()
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch


def _torch_environment(torch: Any, device: str) -> dict[str, object]:
    settings = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    required = {
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    changed = [name for name, value in required.items() if settings[name] != value]
    if changed:
        raise RuntimeError(f"model backend changed required Torch inference settings: {changed}")
    result: dict[str, object] = {
        "torch_version": str(torch.__version__),
        "device": device,
        **settings,
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "platform": platform.platform(),
    }
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        result.update(
            cuda_version=str(torch.version.cuda),
            device_name=str(properties.name),
            compute_capability=[int(properties.major), int(properties.minor)],
        )
    return result


@contextmanager
def _preserve_rng(torch: Any) -> Iterator[None]:
    """Model construction must not reseed its caller or change the default precision."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        torch.set_default_dtype(dtype)


class TorchModelAdapter(ASELangevinBAOABAdapter):
    """Reuse the ASE runtime while binding every lease to the same local model."""

    def __init__(
        self,
        *,
        trajectory: TrajectorySpec,
        options: Mapping[str, Any],
        files: Mapping[str, Path],
        packages: tuple[str, ...],
        kind: str,
    ) -> None:
        super().__init__(trajectory=trajectory)
        self.options = dict(options)
        self.files = {name: path.resolve() for name, path in files.items()}
        self.device = options.get("device", "cuda")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("model device must be 'cpu' or 'cuda'")
        self.packages = packages
        self.kind = kind
        self._contract: ComponentState | None = None

    def _new_calculator(self) -> Calculator:
        raise NotImplementedError

    def _extra_metadata(self, calculator: Calculator) -> dict[str, object]:
        return {}

    @contextmanager
    def _lease(self) -> Iterator[tuple[Calculator, ComponentState]]:
        torch = _load_torch(self.device)
        # Rehash before every new calculator: replacing a local checkpoint mid-run must
        # not let MD and refinement use different potentials under one recorded identity.
        files = {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in self.files.items()
        }
        with _preserve_rng(torch):
            calculator = self._new_calculator()
        if not isinstance(calculator, Calculator):
            raise TypeError("model backend must return an ASE Calculator")
        try:
            if any(_sha256(path) != files[name]["sha256"] for name, path in self.files.items()):
                raise ValueError("model files changed while the calculator was being constructed")
            state = ComponentState(
                kind=self.kind,
                metadata={
                    "options": self.options,
                    "model_files": files,
                    "package_versions": {name: version(name) for name in self.packages},
                    "python_version": platform.python_version(),
                    "ase_version": ase.__version__,
                    "numpy_version": np.__version__,
                    "adapter_source_sha256": _sha256(Path(inspect.getfile(type(self)))),
                    "torch_adapter_source_sha256": _sha256(Path(__file__)),
                    "ase_runtime_source_sha256": _sha256(Path(__file__).with_name("ase.py")),
                    "integrator_source_sha256": _sha256(Path(__file__).parents[1] / "ase_npt.py"),
                    **_torch_environment(torch, self.device),
                    **self._extra_metadata(calculator),
                },
            )
            if self._contract is not None and state != self._contract:
                raise ValueError("model or inference environment changed during the run")
            self._contract = state
            yield calculator, state
        finally:
            close = getattr(calculator, "close", None)
            if callable(close):
                close()
