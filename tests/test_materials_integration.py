"""Opt-in real CPU checks using installed CHGNet and an already prepared MatterSim file."""

from __future__ import annotations

import os
from importlib import import_module

import numpy as np
import pytest
from ase.build import bulk

from reactionflow.campaign import TrajectorySpec


@pytest.mark.parametrize(
    "backend,model,dtype",
    [
        ("chgnet", "0.3.0", None),
        ("chgnet", "r2scan", None),
        ("mattersim", None, "float32"),
        ("mattersim", None, "float64"),
    ],
)
def test_real_materials_npt_restart(backend, model, dtype):
    if os.environ.get("REACTIONFLOW_TEST_MATERIALS") != "1":
        pytest.skip("set REACTIONFLOW_TEST_MATERIALS=1 in the optional materials environment")
    options = {"device": "cpu"}
    if dtype is not None:
        options["dtype"] = dtype
    if model is None:
        checkpoint = os.environ.get("REACTIONFLOW_TEST_MATTERSIM_CHECKPOINT")
        if not checkpoint:
            pytest.skip("set REACTIONFLOW_TEST_MATTERSIM_CHECKPOINT to an existing absolute file")
        options["checkpoint"] = checkpoint
    else:
        options["model"] = model
    module = import_module(f"reactionflow.adapters.{backend}")
    trajectory = TrajectorySpec(
        id="materials", total_steps=2, timestep_fs=0.1, temperature_K=100, pressure_GPa=0, seed=17
    )
    atoms = bulk("Si", "diamond", a=5.43)
    adapter = module.create_adapter(trajectory=trajectory, options=options)
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(2)
        assert np.isfinite(runtime.atoms.get_potential_energy())
        assert np.isfinite(runtime.atoms.get_forces()).all()
        assert np.isfinite(runtime.atoms.get_stress()).all()
        full = runtime.snapshot()
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(1)
        split = runtime.snapshot()
    fresh = module.create_adapter(trajectory=trajectory, options=options)
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
