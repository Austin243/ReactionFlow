"""Opt-in real CPU checks using prepared weights, with network access disabled.

Set REACTIONFLOW_TEST_ORB_CHECKPOINT to the released OrbMol-v2 checkpoint, or
REACTIONFLOW_TEST_MACE_FIELD_CHECKPOINT to the released MACE-Field checkpoint.
Run this file separately in each backend's pinned environment.
"""

from __future__ import annotations

import os
import socket
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk, molecule

from reactionflow.campaign import TrajectorySpec


@pytest.mark.parametrize("backend,periodic", [("orb", False), ("orb", True), ("mace_field", True)])
def test_real_orb_field_restart_is_bitwise_identical(backend, periodic, monkeypatch):
    variable = f"REACTIONFLOW_TEST_{backend.upper()}_CHECKPOINT"
    configured = os.environ.get(variable)
    if not configured:
        pytest.skip(f"set {variable} to an existing absolute checkpoint path")
    checkpoint = Path(configured)
    assert checkpoint.is_absolute() and checkpoint.is_file()

    import torch

    def no_network(*args, **kwargs):
        pytest.fail("prepared model execution must not access the network")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    previous_threads = torch.get_num_threads()
    previous_dtype = torch.get_default_dtype()
    previous_load_policy = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    torch.set_num_threads(1)
    try:
        module = import_module(f"reactionflow.adapters.{backend}")
        trajectory = TrajectorySpec(
            id=f"real-{backend}",
            total_steps=2,
            timestep_fs=0.1,
            temperature_K=50,
            pressure_GPa=0.1 if periodic else None,
            seed=77,
        )
        options = {"checkpoint": str(checkpoint), "device": "cpu"}
        if backend == "orb":
            options.update(model="orbmol-v2", precision="float32-highest")
            atoms = molecule("H2O")
            atoms.set_cell([8 if periodic else 20] * 3)
            atoms.pbc = periodic
            atoms.info.update(charge=0, spin=1)
        else:
            options.update(head="mp-dielectric", electric_field=[0, 0, 0.01], dtype="float64")
            atoms = bulk("NaCl", "rocksalt", a=5.64)
            # A field under pressure needs the lower-triangular cell pathways use.
            atoms.set_cell(atoms.cell.standard_form()[0], scale_atoms=True)
        adapter = module.create_adapter(trajectory=trajectory, options=options)
        adapter.preflight(atoms)
        with adapter.start(atoms.copy()) as runtime:
            runtime.run(2)
            assert np.isfinite(runtime.atoms.get_potential_energy())
            assert np.isfinite(runtime.atoms.get_forces()).all()
            if periodic:
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
        assert torch.get_default_dtype() == previous_dtype
        assert os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") == previous_load_policy
    finally:
        torch.set_num_threads(previous_threads)
