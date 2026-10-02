"""CHGNet checkpoints bundled with the optional package, or explicit local files."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ..campaign import TrajectorySpec
from ._model_files import require_package
from ._torch import TorchModelAdapter

VERSION = "0.4.2"
_MODELS = {
    "0.3.0": "0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
    "r2scan": "r2scan/chgnet_r2scan_transfer_learning_e15f36s161m23.pth.tar",
}


def catalog() -> dict[str, Any]:
    return {
        "backend": "chgnet",
        "factory": "reactionflow.adapters.chgnet:create_adapter",
        "package": f"chgnet=={VERSION}",
        "models": [
            {"name": "0.3.0", "description": "MPtrj PBE+U materials model"},
            {"name": "r2scan", "description": "MP-r2SCAN transfer-learning model"},
        ],
        "notes": [
            "Weights are included in the package; no separate download is needed.",
            "Requires fully periodic structures; isolated atoms fail explicitly.",
        ],
        "sources": ["https://github.com/CederGroupHub/chgnet"],
    }


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    unknown = set(options) - {"model", "checkpoint", "device"}
    if unknown:
        raise ValueError(f"unknown CHGNet options: {sorted(unknown)}")
    if ("model" in options) == ("checkpoint" in options):
        raise ValueError("CHGNet requires exactly one of model or checkpoint")
    if options.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("CHGNet device must be cpu or cuda")
    if "checkpoint" in options and (
        not isinstance(options["checkpoint"], str) or not Path(options["checkpoint"]).is_absolute()
    ):
        raise ValueError("CHGNet checkpoint must be an absolute path")
    if "model" in options and options["model"] not in _MODELS:
        raise ValueError(f"unknown CHGNet model; choose from {sorted(_MODELS)}")
    require_package("chgnet", VERSION, "chgnet")
    path = (
        Path(options["checkpoint"])
        if "checkpoint" in options
        else Path(
            distribution("chgnet").locate_file(f"chgnet/pretrained/{_MODELS[options['model']]}")
        )
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CHGNet checkpoint is missing: {path}")
    return {"checkpoint": path}


def _periodic(atoms: Atoms) -> None:
    if not atoms.pbc.all() or atoms.cell.rank != 3:
        raise ValueError("CHGNet requires a fully periodic structure with a 3D cell")


class CHGNetAdapter(TorchModelAdapter):
    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        super().__init__(
            trajectory=trajectory,
            options=options,
            files=prepare(options, download=False),
            packages=("chgnet", "pymatgen"),
            kind="reactionflow.chgnet",
        )

    def preflight(self, atoms: Atoms) -> None:
        _periodic(atoms)

    def _new_calculator(self) -> Calculator:
        require_package("chgnet", VERSION, "chgnet")
        import torch

        # CHGNet's graph converter emits float32 tensors regardless of the caller default.
        torch.set_default_dtype(torch.float32)
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model.model import CHGNet

        # Match CHGNet.load's named-model settings while loading only the hashed path.
        settings = (
            {"version": self.options["model"], "mlp_out_bias": False}
            if "model" in self.options
            else {}
        )
        model = CHGNet.from_file(str(self.files["checkpoint"]), **settings)

        class PeriodicCalculator(CHGNetCalculator):
            def check_state(self, atoms, tol=1e-15):
                _periodic(atoms)
                return super().check_state(atoms, tol=tol)

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                _periodic(atoms if atoms is not None else self.atoms)
                return super().calculate(atoms, properties, system_changes)

        return PeriodicCalculator(
            model=model, use_device=self.device, check_cuda_mem=False, on_isolated_atoms="error"
        )

    def _extra_metadata(self, calculator: Calculator) -> dict[str, Any]:
        return {"dtype": "float32"}


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> CHGNetAdapter:
    return CHGNetAdapter(trajectory=trajectory, options=options)
