"""Exact restart support for ASE's Langevin BAOAB NPT integrator."""

from __future__ import annotations

from typing import Any

import ase
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.md.langevinbaoab import LangevinBAOAB

from .restart import ComponentState

_STATE_KIND = "ase.langevin_baoab"


def _copy_attribute(dynamics: LangevinBAOAB, name: str) -> np.ndarray:
    return np.asarray(getattr(dynamics, name)).copy()


def snapshot_langevin_baoab(dynamics: LangevinBAOAB) -> ComponentState:
    """Capture every mutable quantity used by the next BAOAB step."""

    if not isinstance(dynamics, LangevinBAOAB):
        raise TypeError("expected an ASE LangevinBAOAB dynamics object")

    metadata: dict[str, Any] = {
        "ase_version": ase.__version__,
        "numpy_version": np.__version__,
        "nsteps": int(dynamics.nsteps),
        "configuration": {
            "timestep": float(dynamics.dt),
            "temperature_K": dynamics.temperature_K,
            "externalstress": dynamics.externalstress,
            "hydrostatic": bool(dynamics.hydrostatic),
            "T_tau": dynamics.T_tau,
            "P_tau": getattr(dynamics, "P_tau", None),
            "P_mass": getattr(dynamics, "barostat_mass", None),
            "disable_cell_langevin": bool(dynamics.disable_cell_langevin),
        },
    }
    if dynamics.temperature_K is not None:
        metadata["rng_bit_generator"] = type(dynamics.rng.bit_generator).__name__
        metadata["rng_state"] = dynamics.rng.bit_generator.state

    arrays = {"accel": _copy_attribute(dynamics, "accel")}
    if dynamics.externalstress is not None:
        arrays.update(
            p_eps=_copy_attribute(dynamics, "p_eps"),
            force_eps=_copy_attribute(dynamics, "force_eps"),
            gamma_mod=_copy_attribute(dynamics, "gamma_mod"),
        )
    return ComponentState(kind=_STATE_KIND, metadata=metadata, arrays=arrays)


def _restore_rng(metadata: dict[str, Any]) -> np.random.Generator | None:
    if metadata["configuration"]["temperature_K"] is None:
        return None
    name = metadata.get("rng_bit_generator")
    bit_generator_type = getattr(np.random, str(name), None)
    if bit_generator_type is None:
        raise ValueError(f"unsupported NumPy bit generator {name!r}")
    rng = np.random.Generator(bit_generator_type())
    rng.bit_generator.state = metadata["rng_state"]
    return rng


def restore_langevin_baoab(
    atoms: Atoms,
    calculator: Calculator,
    state: ComponentState,
) -> LangevinBAOAB:
    """Reconstruct BAOAB so its next step matches an uninterrupted run."""

    if state.kind != _STATE_KIND or state.version != 1 or not state.exact:
        raise ValueError("unsupported or inexact LangevinBAOAB restart state")
    metadata = dict(state.metadata)
    versions = {
        "ASE": (metadata.get("ase_version"), ase.__version__),
        "NumPy": (metadata.get("numpy_version"), np.__version__),
    }
    for name, (recorded, installed) in versions.items():
        if recorded != installed:
            raise ValueError(
                f"exact LangevinBAOAB restart requires {name} {recorded}, "
                f"but {installed} is installed"
            )
    configuration = dict(metadata["configuration"])
    atoms.calc = calculator
    dynamics = LangevinBAOAB(
        atoms,
        timestep=configuration["timestep"],
        temperature_K=configuration["temperature_K"],
        externalstress=configuration["externalstress"],
        hydrostatic=configuration["hydrostatic"],
        T_tau=configuration["T_tau"],
        P_tau=configuration["P_tau"],
        P_mass=configuration["P_mass"],
        disable_cell_langevin=configuration["disable_cell_langevin"],
        rng=_restore_rng(metadata),
        logfile=None,
    )
    dynamics.nsteps = int(metadata["nsteps"])
    dynamics.accel = state.arrays["accel"].copy()
    if dynamics.externalstress is not None:
        dynamics.p_eps = state.arrays["p_eps"].copy()
        if dynamics.hydrostatic:
            dynamics.p_eps = float(dynamics.p_eps)
        dynamics.force_eps = state.arrays["force_eps"].copy()
        if dynamics.hydrostatic:
            dynamics.force_eps = float(dynamics.force_eps)
        dynamics.gamma_mod = state.arrays["gamma_mod"].copy()
        if dynamics.hydrostatic:
            dynamics.gamma_mod = float(dynamics.gamma_mod)
    return dynamics


__all__ = ["restore_langevin_baoab", "snapshot_langevin_baoab"]
