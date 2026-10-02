"""Opt-in real SevenNet CPU checks; uses bundled or already prepared local weights.

Set REACTIONFLOW_TEST_SEVENNET=1 in the optional SevenNet environment. To also
check Omni, set REACTIONFLOW_TEST_SEVENNET_OMNI_CHECKPOINT to an absolute file.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from reactionflow.adapters.sevennet import create_adapter
from reactionflow.campaign import TrajectorySpec


@pytest.mark.parametrize(
    "model,modal",
    [("7net-0", None), ("7net-mf-0", "PBE"), ("7net-mf-0", "R2SCAN"), ("7net-omni", "mpa")],
)
def test_real_sevennet_npt_restart_is_bitwise_identical(model, modal, monkeypatch):
    if os.environ.get("REACTIONFLOW_TEST_SEVENNET") != "1":
        pytest.skip("set REACTIONFLOW_TEST_SEVENNET=1 in the optional SevenNet environment")
    options = {"model": model, "device": "cpu"}
    if modal is not None:
        options["modal"] = modal
    if model == "7net-omni":
        configured = os.environ.get("REACTIONFLOW_TEST_SEVENNET_OMNI_CHECKPOINT")
        if not configured:
            pytest.skip(
                "set REACTIONFLOW_TEST_SEVENNET_OMNI_CHECKPOINT to an existing absolute file"
            )
        checkpoint = Path(configured)
        assert checkpoint.is_absolute() and checkpoint.is_file()
        options.pop("model")
        options["checkpoint"] = str(checkpoint)

    import torch

    def no_network(*args, **kwargs):
        pytest.fail("prepared SevenNet runtime must not access the network")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    previous_dtype, previous_threads = torch.get_default_dtype(), torch.get_num_threads()
    # Check that native float32 inference also works for float64 callers.
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    try:
        trajectory = TrajectorySpec(
            id="real-sevennet",
            total_steps=2,
            timestep_fs=0.1,
            temperature_K=50,
            pressure_GPa=0.1,
            seed=77,
        )
        atoms = bulk("Si", "diamond", a=5.43)
        adapter = create_adapter(trajectory=trajectory, options=options)
        with adapter.start(atoms.copy()) as runtime:
            runtime.run(2)
            assert np.isfinite(runtime.atoms.get_potential_energy())
            assert np.isfinite(runtime.atoms.get_forces()).all()
            assert np.isfinite(runtime.atoms.get_stress()).all()
            full = runtime.snapshot()
        with adapter.start(atoms.copy()) as runtime:
            runtime.run(1)
            split = runtime.snapshot()
        fresh = create_adapter(trajectory=trajectory, options=options)
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
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)
