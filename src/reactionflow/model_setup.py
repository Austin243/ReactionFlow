"""Explicit preparation for supported pretrained models, separate from MD execution."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .campaign import AdapterSpec, CampaignConfig

_PACKAGES = {
    "reactionflow.adapters.mace:create_adapter": "mace-torch==0.3.16",
    "reactionflow.adapters.uma:create_adapter": "fairchem-core==2.23.0",
}


def _selected_adapters(campaign: CampaignConfig, index: int | None) -> tuple[AdapterSpec, ...]:
    indices = range(len(campaign.trajectories)) if index is None else (index,)
    unique = {}
    for item in indices:
        spec = campaign.adapter_for(item)
        key = (spec.factory, json.dumps(dict(spec.options), sort_keys=True, allow_nan=False))
        unique.setdefault(key, spec)
    return tuple(unique.values())


def prepare_adapter(spec: AdapterSpec) -> dict[str, Path]:
    """Download/resolve one supported model without constructing a calculator."""

    if spec.factory not in _PACKAGES:
        raise ValueError(
            "--download supports the built-in MACE and UMA adapters; "
            f"{spec.factory!r} manages its own setup"
        )
    module = import_module(spec.factory.split(":", 1)[0])
    return module.prepare(dict(spec.options), download=True)


def prepare_campaign(campaign: CampaignConfig, *, index: int | None = None) -> list[dict]:
    """Prepare each referenced adapter once, or only the selected trajectory's adapter."""

    results = []
    for spec in _selected_adapters(campaign, index):
        if spec.factory not in _PACKAGES:
            results.append(
                {
                    "factory": spec.factory,
                    "status": "external_setup",
                    "message": "This adapter manages its own dependencies and model files.",
                }
            )
            continue
        files = prepare_adapter(spec)
        results.append(
            {
                "factory": spec.factory,
                "status": "prepared",
                "files": {name: str(path) for name, path in files.items()},
            }
        )
    return results


def install_and_prepare(campaign: CampaignConfig, *, index: int | None = None) -> int:
    """Install selected optional packages, then prepare models in a fresh interpreter."""

    packages = list(
        dict.fromkeys(
            _PACKAGES[spec.factory]
            for spec in _selected_adapters(campaign, index)
            if spec.factory in _PACKAGES
        )
    )
    mace = "mace-torch==0.3.16" in packages
    uma = "fairchem-core==2.23.0" in packages
    if mace and uma:
        raise ValueError(
            "MACE 0.3.16 requires e3nn==0.4.4, but UMA's FAIR-Chem 2.23.0 requires "
            "e3nn>=0.5; use --index to prepare each in a separate Python environment"
        )
    conflicts = []
    if uma:
        conflicts.extend([("mace-torch", "0.3.16"), ("torchani", "2.8.4")])
    if mace:
        conflicts.append(("fairchem-core", "2.23.0"))
    for name, pinned in conflicts:
        try:
            installed_version = version(name)
        except PackageNotFoundError:
            continue
        if installed_version == pinned:
            raise ValueError(
                f"installing {packages[0]} would conflict with installed {name}=={pinned}; "
                "use a fresh Python environment to preserve the existing model setup"
            )
    if packages:
        installed = subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=False)
        if installed.returncode:
            return installed.returncode
    # pip may have replaced NumPy or another already-imported package. Do not import a
    # model backend in this process after installation, even when pip succeeds.
    command = [sys.executable, "-m", "reactionflow.cli", "prepare", str(campaign.source)]
    if index is not None:
        command.extend(["--index", str(index)])
    return subprocess.run(command, check=False).returncode


__all__ = ["install_and_prepare", "prepare_adapter", "prepare_campaign"]
