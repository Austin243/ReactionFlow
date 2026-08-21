"""Narrow extension boundary for user-selected MLIP implementations."""

from __future__ import annotations

from contextlib import AbstractContextManager
from importlib import import_module
from typing import Protocol

from ase.calculators.calculator import Calculator

from .campaign import AdapterSpec, TrajectorySpec
from .runtime import ExactRuntimeProvider


class MLIPAdapter(ExactRuntimeProvider, Protocol):
    """Exact MD runtime plus sequential calculators for pathway refinement."""

    def calculator(self, stage: str) -> AbstractContextManager[Calculator]: ...


def load_mlip_adapter(spec: AdapterSpec, trajectory: TrajectorySpec) -> MLIPAdapter:
    """Create an adapter from an explicit ``module:factory`` reference."""

    module_name, factory_name = spec.factory.split(":", 1)
    module = import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TypeError(f"MLIP adapter factory {spec.factory!r} is not callable")
    adapter = factory(trajectory=trajectory, options=dict(spec.options))
    missing = [
        name
        for name in ("start", "restore", "calculator")
        if not callable(getattr(adapter, name, None))
    ]
    if missing:
        raise TypeError(f"MLIP adapter is missing callable methods: {', '.join(missing)}")
    return adapter


__all__ = ["MLIPAdapter", "load_mlip_adapter"]
