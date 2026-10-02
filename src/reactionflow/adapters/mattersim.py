"""Prepared MatterSim materials potentials with local-only inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ..campaign import TrajectorySpec
from ._model_files import cached_download, model_cache, require_package
from ._torch import TorchModelAdapter

VERSION = "1.2.5"
_RELEASE = "40a1eb8f1189a53af310957b4f2c5dfbfe68d647"
_MODELS = ("mattersim-v1.0.0-1M", "mattersim-v1.0.0-5M")


def catalog() -> dict[str, Any]:
    return {
        "backend": "mattersim",
        "factory": "reactionflow.adapters.mattersim:create_adapter",
        "package": f"mattersim=={VERSION}",
        "models": [{"name": name} for name in _MODELS],
        "notes": [
            "Periodic materials; PBE training across elements, temperatures and pressures.",
            "Requires fully periodic structures. dtype: float32 (default) or float64.",
        ],
        "sources": ["https://github.com/microsoft/mattersim"],
    }


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    unknown = set(options) - {"model", "checkpoint", "device", "dtype", "cache_dir"}
    if unknown:
        raise ValueError(f"unknown MatterSim options: {sorted(unknown)}")
    if ("model" in options) == ("checkpoint" in options):
        raise ValueError("MatterSim requires exactly one of model or checkpoint")
    if options.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("MatterSim device must be cpu or cuda")
    if options.get("dtype", "float32") not in ("float32", "float64"):
        raise ValueError("MatterSim dtype must be float32 or float64")
    for key in ("checkpoint", "cache_dir"):
        if key in options and (
            not isinstance(options[key], str) or not Path(options[key]).is_absolute()
        ):
            raise ValueError(f"MatterSim {key} must be an absolute path")
    if "model" in options and options["model"] not in _MODELS:
        raise ValueError(f"unknown MatterSim model; choose from {_MODELS}")
    require_package("mattersim", VERSION, "mattersim")
    if "checkpoint" in options:
        path = Path(options["checkpoint"]).resolve()
    else:
        url = (
            f"https://raw.githubusercontent.com/microsoft/mattersim/{_RELEASE}/"
            f"pretrained_models/{options['model']}.pth"
        )
        path = cached_download(url, model_cache(options, "mattersim"), download=download)
    if not path.is_file():
        raise FileNotFoundError(f"MatterSim checkpoint is missing: {path}")
    return {"checkpoint": path}


def _periodic(atoms: Atoms) -> None:
    if not atoms.pbc.all() or atoms.cell.rank != 3:
        raise ValueError("MatterSim requires a fully periodic structure with a 3D cell")


class MatterSimAdapter(TorchModelAdapter):
    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        super().__init__(
            trajectory=trajectory,
            options=options,
            files=prepare(options, download=False),
            packages=("mattersim", "e3nn", "torch-geometric", "pymatgen"),
            kind="reactionflow.mattersim",
        )

    def preflight(self, atoms: Atoms) -> None:
        _periodic(atoms)

    def _new_calculator(self) -> Calculator:
        require_package("mattersim", VERSION, "mattersim")
        import torch
        from mattersim.forcefield import MatterSimCalculator, Potential
        from mattersim.forcefield.m3gnet.m3gnet import M3Gnet

        # from_checkpoint creates an upstream home-directory cache even for local files.
        # Build from the hashed checkpoint directly, preserving its exact architecture.
        checkpoint = torch.load(
            self.files["checkpoint"], map_location=self.device, weights_only=False
        )
        if checkpoint.get("model_name") != "m3gnet":
            raise ValueError("MatterSim requires an m3gnet checkpoint")
        torch.set_default_dtype(torch.float32)
        model = M3Gnet(device=self.device, **checkpoint["model_args"]).to(self.device)
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        # Released v1 weights predate this deterministic, nonlearned basis buffer.
        if set(incompatible.missing_keys) - {"sbf.coef"} or incompatible.unexpected_keys:
            raise ValueError(
                f"MatterSim checkpoint does not match its architecture: {incompatible}"
            )
        model.eval()
        potential = Potential(model, device=self.device, allow_tf32=False, model_name="m3gnet")
        dtype = getattr(torch, self.options.get("dtype", "float32"))

        class PeriodicCalculator(MatterSimCalculator):
            def check_state(self, atoms, tol=1e-15):
                _periodic(atoms)
                return super().check_state(atoms, tol=tol)

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                _periodic(atoms if atoms is not None else self.atoms)
                # Upstream creates strain tensors from the global default dtype.
                previous_dtype = torch.get_default_dtype()
                try:
                    torch.set_default_dtype(dtype)
                    return super().calculate(atoms, properties, system_changes)
                finally:
                    torch.set_default_dtype(previous_dtype)

        return PeriodicCalculator(
            potential=potential,
            device=self.device,
            dtype=self.options.get("dtype", "float32"),
            compute_stress=True,
            compile=False,
            batch_converter=False,
            direct_graph=False,
        )


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> MatterSimAdapter:
    return MatterSimAdapter(trajectory=trajectory, options=options)
