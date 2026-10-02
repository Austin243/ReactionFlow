"""Opt-in AIMNet2 CPU tests; no downloads.

Set REACTIONFLOW_TEST_AIMNET2_CHECKPOINT and/or REACTIONFLOW_TEST_AIMNET2_RXN_CHECKPOINT
to existing absolute checkpoint paths. Run pytest -q tests/test_aimnet2_integration.py
in an environment with the aimnet2 extra installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule

from reactionflow.adapters.aimnet2 import AIMNet2Adapter
from reactionflow.campaign import TrajectorySpec


@pytest.mark.parametrize(
    "variable", ["REACTIONFLOW_TEST_AIMNET2_CHECKPOINT", "REACTIONFLOW_TEST_AIMNET2_RXN_CHECKPOINT"]
)
@pytest.mark.parametrize("pressure", [None, 0.0])
def test_real_aimnet2_restart_is_bitwise_identical(variable, pressure):
    configured = os.environ.get(variable)
    checkpoint = Path(configured) if configured else None
    if checkpoint is None or not checkpoint.is_absolute() or not checkpoint.is_file():
        pytest.skip(f"set {variable} to an existing absolute checkpoint path")
    trajectory = TrajectorySpec(
        id="real-aimnet2",
        total_steps=2,
        timestep_fs=0.1,
        temperature_K=100.0,
        pressure_GPa=pressure,
        seed=17,
    )
    options = {"checkpoint": str(checkpoint), "device": "cpu"}
    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    if pressure is not None:
        atoms.set_cell([12] * 3)
        atoms.center()
        atoms.pbc = True
    adapter = AIMNet2Adapter(trajectory=trajectory, options=options)
    adapter.preflight(atoms)
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(2)
        assert np.isfinite(runtime.atoms.get_potential_energy())
        assert np.isfinite(runtime.atoms.get_forces()).all()
        if pressure is not None:
            assert np.isfinite(runtime.atoms.get_stress()).all()
        full = runtime.snapshot()
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(1)
        split = runtime.snapshot()
    fresh = AIMNet2Adapter(trajectory=trajectory, options=options)
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
