"""Explicit MACE checkpoints with offline calculator construction."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.fd import calculate_numerical_stress

from ..campaign import TrajectorySpec
from ._model_files import cached_download, model_cache, require_package
from ._torch import TorchModelAdapter

MACE_VERSION = "0.3.16"
_POLAR_STRAIN = 1e-5

_ALLOWED_OPTIONS = {"model", "family", "checkpoint", "device", "dtype", "head", "cache_dir"}

_MH1_HEADS = {
    "omat_pbe": "OMat24 PBE/PBE+U; inorganic materials and cross-domain chemistry",
    "omol": "OMol25 neutral subset, wB97M-V; molecular and organometallic chemistry",
    "spice_wB97M": "SPICE, wB97M-D3(BJ); molecular chemistry",
    "oc20_usemppbe": "OC20, PBE; adsorbates and surfaces",
    "matpes_r2scan": "MatPES, r2SCAN; inorganic materials",
    "mp_pbe_refit_add": "Materials Project trajectories, PBE/PBE+U; inorganic materials",
}
_MP_MODELS = (
    "small",
    "medium",
    "large",
    "small-0b",
    "medium-0b",
    "small-0b2",
    "medium-0b2",
    "large-0b2",
    "medium-0b3",
    "medium-mpa-0",
    "small-omat-0",
    "medium-omat-0",
    "mace-matpes-pbe-0",
    "mace-matpes-r2scan-0",
    "mh-0",
    "mh-1",
)


def catalog() -> dict[str, Any]:
    """Describe the pinned models without importing MACE or downloading weights."""

    models = [{"name": name, "family": "mp"} for name in _MP_MODELS]
    models.extend(
        {"name": name, "family": "off", "description": "MACE-OFF23 organic molecules"}
        for name in ("small", "medium", "large")
    )
    models.extend(
        {"name": f"polar-1-{size}", "family": "polar", "description": f"MACE-POLAR-1 {label}"}
        for size, label in (("s", "small"), ("m", "medium"), ("l", "large"))
    )
    for model in models:
        if model["name"] == "mh-1":
            model.update(
                description="MACE-MH-1 multi-head foundation model", heads=list(_MH1_HEADS)
            )
        elif model["name"] == "mh-0":
            model.update(description="MACE-MH-0; select an explicit checkpoint head")
    return {
        "backend": "mace",
        "factory": "reactionflow.adapters.mace:create_adapter",
        "package": f"mace-torch=={MACE_VERSION}",
        "models": models,
        "heads": dict(_MH1_HEADS),
        "notes": [
            "Named mh-0 and mh-1 models require an explicit head.",
            "These six heads were verified in the released MH-1 checkpoint. "
            "rgd1_b3lyp belongs to MH-0, not the current MH-1 release.",
            "Local checkpoints may expose other heads; the loaded checkpoint is authoritative.",
            "No additional D3 dispersion correction is enabled by this adapter.",
            "Polar models require graph_longrange 0.4.0; "
            "prepare --install installs the pinned source.",
            "Polar inputs: explicit integer atoms.info charge, positive integer spin multiplicity, "
            "and optional finite external_field [Ex, Ey, Ez] in V/angstrom (zero by default).",
            "Polar stress uses central differences of the full energy (strain 1e-5), "
            "retaining analytic forces; stress/NPT require float64 and cost twelve energy probes.",
        ],
        "sources": [
            "https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mh_1",
            "https://github.com/ACEsuit/mace/discussions/1462",
            "https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html",
            "https://github.com/ACEsuit/mace/issues/1642",
        ],
    }


def _validate(options: Mapping[str, Any]) -> None:
    unknown = set(options) - _ALLOWED_OPTIONS
    if unknown:
        raise ValueError(f"unknown MACE adapter options: {sorted(unknown)}")
    if ("model" in options) == ("checkpoint" in options):
        raise ValueError("MACE requires exactly one of 'model' or 'checkpoint'")
    family = options.get("family", "mp")
    if family not in ("mp", "off", "polar"):
        raise ValueError("MACE family must be 'mp', 'off', or 'polar'")
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
    if family == "mp" and options.get("model") in {"mh-0", "mh-1"}:
        if "head" not in options:
            raise ValueError(
                "MACE multi-head models require an explicit head; run reactionflow models"
            )
        if options["model"] == "mh-1" and options["head"] not in _MH1_HEADS:
            raise ValueError(
                f"unknown MACE-MH-1 head {options['head']!r}; choose from {list(_MH1_HEADS)}"
            )


def prepare(options: Mapping[str, Any], *, download: bool = True) -> dict[str, Path]:
    """Resolve a local checkpoint, downloading a named model only when allowed."""

    _validate(options)
    require_package("mace-torch", MACE_VERSION, "mace")
    require_package("e3nn", "0.4.4", "mace")
    if options.get("family") == "polar":
        _require_polar()
    if "checkpoint" in options:
        path = Path(options["checkpoint"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MACE checkpoint does not exist: {path}")
    else:
        from mace.calculators.foundations_models import (
            mace_mp_urls,
            mace_off_urls,
            polar_model_urls,
        )

        family = options.get("family", "mp")
        registry = {"mp": mace_mp_urls, "off": mace_off_urls, "polar": polar_model_urls}[family]
        model = options["model"]
        if model not in registry:
            raise ValueError(
                f"unknown MACE {family} model {model!r}; choose from {sorted(registry)}"
            )
        url = registry[model]
        path = cached_download(url, model_cache(options, "mace"), download=download)
    return {"checkpoint": path}


def _require_polar() -> None:
    try:
        require_package("graph_longrange", "0.4.0", "mace")
    except RuntimeError as error:
        raise RuntimeError(
            "MACE-POLAR requires graph_longrange 0.4.0; run "
            "'reactionflow prepare campaign.json --install' to install its pinned source"
        ) from error


def _polar_inputs(atoms: Atoms) -> tuple[int, int, tuple[float, ...]]:
    for name in ("charge", "spin"):
        value = atoms.info.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"MACE-POLAR requires explicit integer atoms.info[{name!r}]")
        if name == "spin" and value < 1:
            raise ValueError("MACE-POLAR spin must be a positive multiplicity")
    field = np.asarray(atoms.info.get("external_field", [0.0, 0.0, 0.0]), dtype=float)
    if field.shape != (3,) or not np.isfinite(field).all():
        raise ValueError("MACE-POLAR external_field must be a finite three-vector")
    return int(atoms.info["charge"]), int(atoms.info["spin"]), tuple(map(float, field))


class _NumericalPolarStressCalculator(Calculator):
    """Keep analytic energy/forces and differentiate the full energy for stress."""

    def __init__(self, calculator: Calculator, *, dtype: str) -> None:
        super().__init__()
        self.inner = calculator
        self.head = calculator.head
        self.dtype = dtype
        self.implemented_properties = [
            name for name in calculator.implemented_properties if name != "stresses"
        ]
        self._polar_signature = None

    def check_state(self, atoms, tol=1e-15):
        signature = _polar_inputs(atoms)
        changes = super().check_state(atoms, tol=tol)
        if signature != self._polar_signature:
            changes.append("info")
        return changes

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        signature = _polar_inputs(atoms if atoms is not None else self.atoms)
        if "stress" in properties and self.dtype != "float64":
            raise ValueError("MACE-POLAR numerical stress requires dtype='float64'")
        super().calculate(atoms, properties, system_changes)
        self.results = {}
        probe = self.atoms.copy()
        # ASE applies user constraints outside the calculator. Their energy or
        # strain projection must not enter the raw model stress a second time.
        probe.set_constraint()
        probe.calc = self.inner
        probe.get_potential_energy()
        # Inner results are for the original geometry. Never publish its analytic
        # stress or alias arrays overwritten while evaluating strained geometries.
        results = deepcopy(self.inner.results)
        results.pop("stress", None)
        results.pop("stresses", None)
        if "stress" in properties:
            results["stress"] = calculate_numerical_stress(
                probe, eps=_POLAR_STRAIN, force_consistent=True
            )
        self.results = results
        self._polar_signature = signature


class MACEAdapter(TorchModelAdapter):
    """One pinned MACE calculator for every trajectory and pathway stage."""

    def __init__(self, *, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> None:
        if (
            options.get("family") == "polar"
            and trajectory.pressure_GPa is not None
            and options.get("dtype", "float64") != "float64"
        ):
            raise ValueError("MACE-POLAR NPT requires dtype='float64' for numerical stress")
        super().__init__(
            trajectory=trajectory,
            options=options,
            files=prepare(options, download=False),
            packages=("mace-torch", "e3nn")
            + (("graph_longrange",) if options.get("family") == "polar" else ()),
            kind="reactionflow.mace",
        )

    def _new_calculator(self) -> Calculator:
        require_package("mace-torch", MACE_VERSION, "mace")
        require_package("e3nn", "0.4.4", "mace")
        from mace.calculators import MACECalculator
        from mace.modules import PolarMACE

        calculator_class = MACECalculator
        kwargs = {
            "model_paths": [str(self.files["checkpoint"])],
            "device": self.device,
            "default_dtype": self.options.get("dtype", "float64"),
        }
        if self.options.get("family") == "polar":
            _require_polar()
            kwargs["model_type"] = "PolarMACE"

            class ExplicitPolarCalculator(MACECalculator):
                def check_state(self, atoms, tol=1e-15):
                    signature = _polar_inputs(atoms)
                    changes = super().check_state(atoms, tol=tol)
                    # Upstream ignores ndarray-valued info; retain a value copy so an
                    # in-place field update cannot reuse forces from the previous field.
                    if (
                        signature != getattr(self, "_polar_signature", None)
                        and "info" not in changes
                    ):
                        changes.append("info")
                    return changes

                def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                    signature = _polar_inputs(atoms if atoms is not None else self.atoms)
                    result = super().calculate(atoms, properties, system_changes)
                    self._polar_signature = signature
                    return result

            calculator_class = ExplicitPolarCalculator
        if "head" in self.options:
            kwargs["head"] = self.options["head"]
        calculator = calculator_class(**kwargs)
        polar = self.options.get("family") == "polar"
        if any(isinstance(model, PolarMACE) != polar for model in calculator.models):
            raise ValueError(
                "MACE checkpoint architecture does not match family: PolarMACE checkpoints "
                "require family='polar'; other checkpoints must "
                "use family='mp' or 'off'"
            )
        if "head" in self.options and calculator.head != self.options["head"]:
            raise ValueError(
                f"MACE did not select requested head {self.options['head']!r}; "
                f"resolved to {calculator.head!r}"
            )
        return (
            _NumericalPolarStressCalculator(calculator, dtype=self.options.get("dtype", "float64"))
            if polar
            else calculator
        )

    def _extra_metadata(self, calculator: Calculator) -> dict[str, Any]:
        metadata = {"resolved_head": calculator.head}
        if self.options.get("family") == "polar":
            metadata["stress"] = {
                "method": "central_finite_difference_free_energy",
                "eps_strain": _POLAR_STRAIN,
                "required_dtype": "float64",
            }
        return metadata


def create_adapter(*, trajectory: TrajectorySpec, options: Mapping[str, Any]) -> MACEAdapter:
    return MACEAdapter(trajectory=trajectory, options=options)
