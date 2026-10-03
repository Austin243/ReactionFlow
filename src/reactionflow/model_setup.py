"""Explicit preparation for supported pretrained models, separate from MD execution."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from .campaign import AdapterSpec, CampaignConfig

_FIELD_COMMIT = "136e4ef040d7c51a5b051a7a609ffb1f29307478"
_FIELD_PACKAGE = f"mace-torch @ git+https://github.com/mdi-group/mace-field.git@{_FIELD_COMMIT}"
_POLAR_PACKAGE = (
    "graph_longrange @ git+https://github.com/WillBaldwin0/graph_electrostatics.git"
    "@0e21d5546c482d08388a08eb4d948e833227ce47"
)
_PACKAGES = {
    "reactionflow.adapters.mace:create_adapter": "mace-torch==0.3.16",
    "reactionflow.adapters.uma:create_adapter": "fairchem-core==2.23.0",
    "reactionflow.adapters.aimnet2:create_adapter": "aimnet[ase]==0.2.0",
    "reactionflow.adapters.orb:create_adapter": "orb-models==0.7.0",
    "reactionflow.adapters.mattersim:create_adapter": "mattersim==1.2.5",
    "reactionflow.adapters.chgnet:create_adapter": "chgnet==0.4.2",
    "reactionflow.adapters.mace_field:create_adapter": _FIELD_PACKAGE,
    "reactionflow.adapters.sevennet:create_adapter": "sevenn==0.13.0",
    "reactionflow.adapters.nep:create_adapter": "calorine==4.0",
    "reactionflow.adapters.ani1xnr:create_adapter": "torchani==2.8.4",
}


def model_catalog(backend: str | None = None) -> list[dict]:
    """List supported choices without importing model libraries or accessing the network."""

    modules = {
        factory.split(":")[0].rsplit(".", 1)[1]: factory.split(":")[0] for factory in _PACKAGES
    }
    if backend is not None:
        backend = backend.replace("-", "_")
    if backend is not None and backend not in modules:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(modules)}")
    selected = modules.values() if backend is None else (modules[backend],)
    return [
        {
            "backend": module.rsplit(".", 1)[1],
            "factory": f"{module}:create_adapter",
            **import_module(module).catalog(),
        }
        for module in selected
    ]


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
            "--download supports the built-in model adapters; "
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


def backend_conflict(factories: Iterable[str]) -> str | None:
    """Return why these built-in backends cannot share one Python environment, or None."""

    packages = {_PACKAGES[factory] for factory in factories if factory in _PACKAGES}
    mace = "mace-torch==0.3.16" in packages
    field = _FIELD_PACKAGE in packages
    uma = "fairchem-core==2.23.0" in packages
    orb = "orb-models==0.7.0" in packages
    mattersim = "mattersim==1.2.5" in packages
    sevennet = "sevenn==0.13.0" in packages
    ani = "torchani==2.8.4" in packages
    if field and mace:
        return "MACE-FIELD and MACE/POLAR install different mace packages"
    if (mace or field) and (uma or mattersim or sevennet):
        return (
            "MACE and MACE-FIELD require e3nn==0.4.4, "
            "but UMA, MatterSim and SevenNet require e3nn>=0.5"
        )
    if orb and (uma or mattersim):
        return "UMA/MatterSim and ORB use different tested nvalchemi dependency stacks"
    if ani and uma:
        return "ANI-1xnr pins Torch 2.11, but UMA's FAIR-Chem version requires Torch 2.13"
    return None


def install_and_prepare(
    campaign: CampaignConfig,
    *,
    index: int | None = None,
    extra_packages: Iterable[str] = (),
) -> int:
    """Install selected optional packages, then prepare models in a fresh interpreter."""

    selected = _selected_adapters(campaign, index)
    conflict = backend_conflict(spec.factory for spec in selected)
    if conflict:
        raise ValueError(
            f"{conflict}; use --index to prepare each in a separate Python environment"
        )
    packages = list(
        dict.fromkeys(_PACKAGES[spec.factory] for spec in selected if spec.factory in _PACKAGES)
    )
    mace = "mace-torch==0.3.16" in packages
    field = _FIELD_PACKAGE in packages
    uma = "fairchem-core==2.23.0" in packages
    orb = "orb-models==0.7.0" in packages
    mattersim = "mattersim==1.2.5" in packages
    sevennet = "sevenn==0.13.0" in packages
    ani = "torchani==2.8.4" in packages
    conflicts = []
    if ani:
        conflicts.append(("fairchem-core", "2.23.0"))
    if uma:
        conflicts.extend(
            [("mace-torch", "0.3.16"), ("mace-torch", "0.3.15"), ("torchani", "2.8.4")]
        )
    if mattersim or sevennet:
        conflicts.extend([("mace-torch", "0.3.16"), ("mace-torch", "0.3.15")])
    if uma or mattersim:
        conflicts.append(("orb-models", "0.7.0"))
    if orb:
        conflicts.extend([("fairchem-core", "2.23.0"), ("mattersim", "1.2.5")])
    if mace or field:
        conflicts.extend(
            [("fairchem-core", "2.23.0"), ("mattersim", "1.2.5"), ("sevenn", "0.13.0")]
        )
    for name, pinned in conflicts:
        try:
            installed_version = version(name)
        except PackageNotFoundError:
            continue
        if installed_version == pinned:
            raise ValueError(
                f"selected backends conflict with installed {name}=={pinned}; "
                "use a fresh Python environment to preserve the existing model setup"
            )
    if mace or field:
        try:
            installed_mace = version("mace-torch")
        except PackageNotFoundError:
            installed_mace = None
        if installed_mace is not None:
            info = json.loads(distribution("mace-torch").read_text("direct_url.json") or "{}")
            is_field = info.get("vcs_info", {}).get("commit_id") == _FIELD_COMMIT
            if (field and not is_field) or (mace and installed_mace != "0.3.16"):
                raise ValueError(
                    "MACE-FIELD and official MACE use the same package namespace; "
                    "use a fresh Python environment"
                )
    if any(
        spec.factory == "reactionflow.adapters.mace:create_adapter"
        and spec.options.get("family") == "polar"
        for spec in selected
    ):
        packages.append(_POLAR_PACKAGE)
    if sevennet:
        packages.append("torch>=2.8,<3")
    if ani:
        packages.append("torch==2.11.0")
    packages.extend(extra_packages)
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


__all__ = [
    "backend_conflict",
    "install_and_prepare",
    "model_catalog",
    "prepare_adapter",
    "prepare_campaign",
]
