"""Prepared UMA checkpoints with explicit tasks and local-only runtime loading."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from importlib.metadata import distribution
from numbers import Integral
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ..campaign import TrajectorySpec
from ._model_files import model_cache, require_package
from ._torch import TorchModelAdapter

_VERSION = "2.23.0"
_OPTIONS = {
    "model",
    "checkpoint",
    "atom_refs",
    "form_elem_refs",
    "task",
    "device",
    "cache_dir",
    "revision",
}
_INFERENCE_SETTINGS = {
    "merge_mole": False,
    "compile": False,
    "tf32": False,
    "execution_mode": "general",
    "base_precision_dtype": "float32",
    "activation_checkpointing": False,
    "auto_add_default_untrained_tasks": True,
}
_TASKS = {
    "omol": "OMol25, wB97M-V; molecules with explicit total charge and spin multiplicity",
    "omat": "OMat24, PBE/PBE+U; inorganic materials",
    "omc": "OMC25, PBE+D3; molecular crystals",
    "odac": "ODAC23, PBE+D3; CO2/H2O adsorption in metal-organic frameworks",
    "oc20": "OC20, RPBE; heterogeneous catalysis",
    "oc22": "OC22, PBE+U; oxide catalysis (UMA 1.2 and 1.2.1)",
    "oc25": "OC25, RPBE+D3; electrolyte/inorganic interfaces (UMA 1.2 and 1.2.1)",
}
_MODEL_TASKS = {
    name: tuple(task for task in _TASKS if "1p2" in name or task not in {"oc22", "oc25"})
    for name in ("uma-s-1p2p1", "uma-s-1p2", "uma-s-1p1", "uma-m-1p1")
}


def catalog() -> dict[str, Any]:
    """Describe the pinned UMA models and tasks without optional imports or downloads."""

    return {
        "backend": "uma",
        "factory": "reactionflow.adapters.uma:create_adapter",
        "package": f"fairchem-core=={_VERSION}",
        "models": [{"name": name, "tasks": list(tasks)} for name, tasks in _MODEL_TASKS.items()],
        "tasks": dict(_TASKS),
        "notes": [
            "Every model requires an explicit task; the loaded checkpoint validates availability.",
            "OC22 and OC25 require UMA 1.2 or 1.2.1. Other listed tasks also exist in UMA 1.1.",
            "OMol requires integer atoms.info charge and positive integer spin multiplicity.",
            "Stress availability does not establish training or suitability for NPT in every task.",
            "Named checkpoints require access to the gated facebook/UMA Hugging Face repository.",
        ],
        "sources": [
            "https://facebookresearch.github.io/fairchem/uma/",
            "https://huggingface.co/facebook/UMA",
        ],
    }


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"UMA {name} must be an absolute path string")
    return Path(value)


def _validate_options(options: Mapping[str, Any]) -> None:
    unknown = set(options) - _OPTIONS
    if unknown:
        raise ValueError(f"unknown UMA adapter options: {sorted(unknown)}")
    if ("model" in options) == ("checkpoint" in options):
        raise ValueError("UMA requires exactly one of model or checkpoint")
    task = options.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("UMA requires an explicit task")
    if options.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("UMA device must be cpu or cuda")
    if "cache_dir" in options:
        _absolute_path(options["cache_dir"], "cache_dir")
    if "model" in options:
        if not isinstance(options["model"], str) or not options["model"]:
            raise ValueError("UMA model must name an official UMA checkpoint")
        if task not in _TASKS:
            raise ValueError(f"unknown named UMA task {task!r}; choose from {list(_TASKS)}")
        supported = _MODEL_TASKS.get(options["model"])
        if supported is not None and task not in supported:
            raise ValueError(
                f"UMA model {options['model']!r} was not trained for task {task!r}; "
                f"choose from {list(supported)} or use a UMA 1.2 model"
            )
        if {"atom_refs", "form_elem_refs"} & options.keys():
            raise ValueError("UMA local reference paths are only valid with checkpoint")
        if "revision" in options and (
            not isinstance(options["revision"], str) or not options["revision"]
        ):
            raise ValueError("UMA revision must be a non-empty Hugging Face revision")
    elif "revision" in options:
        raise ValueError("UMA revision is only valid with a named model")


def _registry() -> dict[str, Any]:
    # Read the pinned distribution's data without importing FAIR-Chem, Torch, or a predictor.
    path = distribution("fairchem-core").locate_file(
        "fairchem/core/calculate/pretrained_models.json"
    )
    registry = json.loads(path.read_text(encoding="utf-8"))
    return {name: spec for name, spec in registry.items() if spec["repo_id"] == "facebook/UMA"}


def _cached_revision(path: Path, cache: Path, repo_id: str) -> str:
    # Keep the cache snapshot path here: resolving its symlink first loses the revision.
    snapshots = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    try:
        revision = path.relative_to(snapshots).parts[0]
    except (ValueError, IndexError) as error:
        raise ValueError(
            "UMA checkpoint did not resolve to a Hugging Face cache snapshot"
        ) from error
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("UMA checkpoint cache snapshot has no immutable commit revision")
    return revision


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    """Resolve local artifacts, optionally downloading a named checkpoint and its references."""

    _validate_options(options)
    require_package("fairchem-core", _VERSION, "uma")
    if "checkpoint" in options:
        files = {
            name: _absolute_path(options[name], name)
            for name in ("checkpoint", "atom_refs", "form_elem_refs")
            if name in options
        }
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

        registry = _registry()
        name = options["model"]
        if name not in registry:
            raise ValueError(f"unknown UMA model {name!r}; available models: {sorted(registry)}")
        spec = registry[name]
        cache = model_cache(options, "uma")
        revision = options.get("revision", spec.get("revision"))

        def fetch(file_spec: Mapping[str, Any]) -> Path:
            try:
                return Path(
                    hf_hub_download(
                        repo_id=spec["repo_id"],
                        filename=file_spec["filename"],
                        subfolder=file_spec.get("subfolder"),
                        revision=revision,
                        cache_dir=str(cache),
                        local_files_only=not download,
                    )
                )
            except LocalEntryNotFoundError as error:
                raise FileNotFoundError(
                    f"UMA artifact {file_spec['filename']!r} is unavailable; "
                    "prepare this model before running offline"
                ) from error
            except HfHubHTTPError as error:
                if getattr(error.response, "status_code", None) in (401, 403):
                    raise RuntimeError(
                        "UMA download requires access to https://huggingface.co/facebook/UMA; "
                        "accept its access conditions and authenticate with hf auth login "
                        "or HF_TOKEN, then retry preparation"
                    ) from error
                raise

        checkpoint = fetch(spec)
        revision = _cached_revision(checkpoint, cache, spec["repo_id"])
        # The checkpoint selects one immutable revision for every reference file, even if
        # the requested branch moves during preparation or a previous download was interrupted.
        files = {"checkpoint": checkpoint}
        for key in ("atom_refs", "form_elem_refs"):
            if spec.get(key) is not None:
                files[key] = fetch(spec[key])
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"UMA {name} file is missing: {path}")
    return {name: path.resolve() for name, path in files.items()}


def _validate_omol(atoms: Atoms) -> None:
    for name in ("charge", "spin"):
        value = atoms.info.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"UMA omol requires explicit integer atoms.info[{name!r}]")
        if name == "spin" and value < 1:
            raise ValueError("UMA omol spin must be a positive multiplicity")


class UMAAdapter(TorchModelAdapter):
    """Use a prepared UMA model with the shared Langevin runtime."""

    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        super().__init__(
            trajectory=trajectory,
            options=options,
            files=prepare(options, download=False),
            packages=("fairchem-core", "e3nn", "hydra-core", "omegaconf"),
            kind="reactionflow.uma",
        )

    def _new_calculator(self) -> Calculator:
        require_package("fairchem-core", _VERSION, "uma")
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        from omegaconf import OmegaConf

        atom_refs = OmegaConf.load(self.files["atom_refs"]) if "atom_refs" in self.files else None
        form_elem_refs = (
            OmegaConf.load(self.files["form_elem_refs"])["refs"]
            if "form_elem_refs" in self.files
            else None
        )
        predictor = load_predict_unit(
            self.files["checkpoint"],
            inference_settings=InferenceSettings(**_INFERENCE_SETTINGS),
            atom_refs=atom_refs,
            form_elem_refs=form_elem_refs,
            device=self.device,
            workers=1,
            seed=0,
        )
        task = self.options["task"]
        if task != "omol":
            return FAIRChemCalculator(predictor, task_name=task)

        class ExplicitOMOLCalculator(FAIRChemCalculator):
            def check_state(self, atoms, tol=1e-15):
                _validate_omol(atoms)
                return super().check_state(atoms, tol=tol)

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                _validate_omol(atoms if atoms is not None else self.atoms)
                return super().calculate(atoms, properties, system_changes)

        return ExplicitOMOLCalculator(predictor, task_name=task)

    def _extra_metadata(self, calculator: Calculator) -> dict[str, Any]:
        return {"inference_settings": dict(_INFERENCE_SETTINGS), "predictor_seed": 0}


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> UMAAdapter:
    return UMAAdapter(trajectory=trajectory, options=options)


__all__ = ["UMAAdapter", "catalog", "create_adapter", "prepare"]
