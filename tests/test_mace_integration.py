"""Optional local-weight MACE check; never downloads a checkpoint.

Set REACTIONFLOW_TEST_MACE_CHECKPOINT to an absolute existing model path, then run
``python -m pytest -q tests/test_mace_integration.py`` in the MACE environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from reactionflow.adapters.mace import MACEAdapter
from reactionflow.campaign import TrajectorySpec


def test_real_mace_npt_restart_is_bitwise_identical() -> None:
    configured = os.environ.get("REACTIONFLOW_TEST_MACE_CHECKPOINT")
    checkpoint = Path(configured) if configured else None
    if checkpoint is None or not checkpoint.is_absolute() or not checkpoint.is_file():
        pytest.skip("set REACTIONFLOW_TEST_MACE_CHECKPOINT to an existing absolute checkpoint path")

    trajectory = TrajectorySpec.from_dict(
        {
            "id": "real-mace-restart",
            "total_steps": 2,
            "timestep_fs": 0.1,
            "temperature_K": 100.0,
            "pressure_GPa": 0.0,
            "seed": 17,
        }
    )
    options = {"checkpoint": str(checkpoint), "device": "cpu", "dtype": "float64"}
    atoms = bulk("Si", "diamond", a=5.43)
    assert len(atoms) == 2

    adapter = MACEAdapter(trajectory=trajectory, options=options)
    with adapter.start(atoms.copy()) as runtime:
        runtime.run(2)
        assert np.isfinite(runtime.atoms.get_potential_energy())
        assert np.isfinite(runtime.atoms.get_forces()).all()
        assert np.isfinite(runtime.atoms.get_stress()).all()
        full = runtime.snapshot()

    with adapter.start(atoms.copy()) as runtime:
        runtime.run(1)
        split = runtime.snapshot()
    fresh = MACEAdapter(trajectory=trajectory, options=options)
    with fresh.restore(split) as runtime:
        runtime.run(1)
        resumed = runtime.snapshot()

    assert resumed.calculator == full.calculator
    assert resumed.dynamics.metadata == full.dynamics.metadata
    assert resumed.atoms.arrays.keys() == full.atoms.arrays.keys()
    for name in full.atoms.arrays:
        np.testing.assert_array_equal(resumed.atoms.arrays[name], full.atoms.arrays[name])
    np.testing.assert_array_equal(resumed.atoms.cell.array, full.atoms.cell.array)
    np.testing.assert_array_equal(resumed.atoms.pbc, full.atoms.pbc)
    assert resumed.dynamics.arrays.keys() == full.dynamics.arrays.keys()
    for name in full.dynamics.arrays:
        np.testing.assert_array_equal(resumed.dynamics.arrays[name], full.dynamics.arrays[name])
