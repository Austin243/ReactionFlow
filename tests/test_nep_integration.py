"""Opt-in NEP89 CPU check; no downloads.

Set REACTIONFLOW_TEST_NEP_CHECKPOINT to an existing absolute nep.txt path, then run
pytest -q tests/test_nep_integration.py in an environment with the nep extra.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from reactionflow.adapters.nep import NEPAdapter
from reactionflow.campaign import TrajectorySpec


def test_real_nep_npt_restart_is_bitwise_identical():
    configured = os.environ.get("REACTIONFLOW_TEST_NEP_CHECKPOINT")
    checkpoint = Path(configured) if configured else None
    if checkpoint is None or not checkpoint.is_absolute() or not checkpoint.is_file():
        pytest.skip("set REACTIONFLOW_TEST_NEP_CHECKPOINT to an existing absolute checkpoint")
    trajectory = TrajectorySpec(
        id="real-nep", total_steps=2, timestep_fs=0.1, temperature_K=100, pressure_GPa=0, seed=17
    )
    options = {"checkpoint": str(checkpoint), "device": "cpu"}
    atoms = bulk("Si", "diamond", a=5.43)
    adapter = NEPAdapter(trajectory=trajectory, options=options)
    adapter.preflight(atoms)
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(2)
        assert np.isfinite(runtime.atoms.get_potential_energy())
        assert runtime.atoms.get_potential_energy(force_consistent=True) == (
            runtime.atoms.get_potential_energy()
        )
        assert np.isfinite(runtime.atoms.get_forces()).all()
        assert np.isfinite(runtime.atoms.get_stress()).all()
        full = runtime.snapshot()
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(1)
        split = runtime.snapshot()
    fresh = NEPAdapter(trajectory=trajectory, options=options)
    with fresh.restore(split) as runtime:
        runtime.run(1)
        resumed = runtime.snapshot()
    assert resumed.calculator == full.calculator
    assert resumed.dynamics.metadata == full.dynamics.metadata
    for name in full.atoms.arrays:
        np.testing.assert_array_equal(resumed.atoms.arrays[name], full.atoms.arrays[name])
    np.testing.assert_array_equal(resumed.atoms.cell.array, full.atoms.cell.array)
    for name in full.dynamics.arrays:
        np.testing.assert_array_equal(resumed.dynamics.arrays[name], full.dynamics.arrays[name])
