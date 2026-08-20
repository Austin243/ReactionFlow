"""Small interfaces for an exactly restartable molecular-dynamics runtime."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from ase import Atoms

from .restart import ExactRestartSnapshot


class ExactDynamicsRuntime(Protocol):
    """One live trajectory whose next step can be reproduced exactly."""

    @property
    def atoms(self) -> Atoms: ...

    @property
    def nsteps(self) -> int: ...

    def run(self, steps: int) -> None: ...

    def snapshot(self) -> ExactRestartSnapshot: ...


class ExactRuntimeProvider(Protocol):
    """Acquire either a fresh or restored runtime, releasing it on context exit."""

    def start(self, atoms: Atoms) -> AbstractContextManager[ExactDynamicsRuntime]: ...

    def restore(
        self,
        snapshot: ExactRestartSnapshot,
    ) -> AbstractContextManager[ExactDynamicsRuntime]: ...


__all__ = ["ExactDynamicsRuntime", "ExactRuntimeProvider"]
