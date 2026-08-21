from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from reactionflow import ComponentState
from reactionflow.adapters.ani1xnr import ANI1xnrAdapter
from reactionflow.campaign import CampaignConfig, TrajectorySpec


class HarmonicSolid(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.positions
        self.results = {
            "energy": float(0.5 * np.sum(positions**2)),
            "forces": -positions.copy(),
            "stress": np.zeros(6),
        }


def _trajectory(**overrides) -> TrajectorySpec:
    value = {
        "id": "test",
        "total_steps": 6,
        "timestep_fs": 0.25,
        "temperature_K": 300.0,
        "pressure_GPa": 0.1,
        "seed": 77,
        "conditions": {
            "hydrostatic": True,
            "thermostat_tau_fs": 25.0,
            "barostat_tau_fs": 250.0,
        },
        **overrides,
    }
    return TrajectorySpec.from_dict(value)


def _atoms() -> Atoms:
    return Atoms(
        "H2",
        positions=[[0.2, 0.1, 0.3], [1.1, 0.4, 0.2]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )


def _fake_backend(monkeypatch) -> None:
    state = ComponentState(kind="reactionflow.ani1xnr", metadata={"fake": True})
    monkeypatch.setattr(ANI1xnrAdapter, "_load_backend", lambda self: (object(), object()))
    monkeypatch.setattr(ANI1xnrAdapter, "_calculator_state", lambda self: state)
    monkeypatch.setattr(
        ANI1xnrAdapter,
        "_new_calculator",
        lambda self, torch, factory: HarmonicSolid(),
    )


def test_adapter_validates_configuration_without_importing_optional_backend() -> None:
    adapter = ANI1xnrAdapter(
        trajectory=_trajectory(),
        options={"device": "cuda", "model_index": 0, "dtype": "float32"},
    )
    assert (adapter.device, adapter.model_index, adapter.dtype) == ("cuda", 0, "float32")

    with pytest.raises(ValueError, match="unknown ANI-1xnr adapter options"):
        ANI1xnrAdapter(trajectory=_trajectory(), options={"mystery": True})
    with pytest.raises(ValueError, match="unknown ANI-1xnr trajectory conditions"):
        ANI1xnrAdapter(
            trajectory=_trajectory(conditions={"unknown": True}),
            options={},
        )


def test_adapter_restart_matches_uninterrupted_npt_exactly(monkeypatch) -> None:
    _fake_backend(monkeypatch)
    adapter = ANI1xnrAdapter(trajectory=_trajectory(), options={"device": "cpu"})

    uninterrupted_atoms = _atoms()
    with adapter.start(uninterrupted_atoms) as uninterrupted:
        uninterrupted.run(6)
        uninterrupted_snapshot = uninterrupted.snapshot()

    split_atoms = _atoms()
    with adapter.start(split_atoms) as split:
        split.run(2)
        checkpoint = split.snapshot()
    with adapter.restore(checkpoint) as resumed:
        resumed.run(4)
        resumed_snapshot = resumed.snapshot()

    assert resumed_snapshot.calculator == uninterrupted_snapshot.calculator
    assert resumed_snapshot.dynamics.metadata == uninterrupted_snapshot.dynamics.metadata
    for name in resumed_snapshot.atoms.arrays:
        np.testing.assert_array_equal(
            resumed_snapshot.atoms.arrays[name],
            uninterrupted_snapshot.atoms.arrays[name],
        )
    np.testing.assert_array_equal(
        resumed_snapshot.atoms.cell.array,
        uninterrupted_snapshot.atoms.cell.array,
    )
    for name in resumed_snapshot.dynamics.arrays:
        np.testing.assert_array_equal(
            resumed_snapshot.dynamics.arrays[name],
            uninterrupted_snapshot.dynamics.arrays[name],
        )


def test_adapter_rejects_changed_calculator_contract(monkeypatch) -> None:
    _fake_backend(monkeypatch)
    adapter = ANI1xnrAdapter(trajectory=_trajectory(), options={"device": "cpu"})
    with adapter.start(_atoms()) as runtime:
        checkpoint = runtime.snapshot()

    monkeypatch.setattr(
        ANI1xnrAdapter,
        "_calculator_state",
        lambda self: ComponentState(kind="reactionflow.ani1xnr", metadata={"fake": False}),
    )
    with pytest.raises(ValueError, match="environment differs"), adapter.restore(checkpoint):
        pass


def test_checked_in_acn_campaign_and_slurm_shape() -> None:
    root = Path(__file__).parents[1]
    example = root / "examples/perlmutter/acn_20gpa_ani1xnr"
    campaign = CampaignConfig.load(example / "campaign.json")

    assert len(campaign.trajectories) == 4
    assert [trajectory.temperature_K for trajectory in campaign.trajectories] == [
        100.0,
        300.0,
        500.0,
        700.0,
    ]
    assert [trajectory.seed for trajectory in campaign.trajectories] == [11, 22, 33, 44]
    assert {trajectory.pressure_GPa for trajectory in campaign.trajectories} == {20.0}
    assert {trajectory.total_steps for trajectory in campaign.trajectories} == {1000}
    assert campaign.adapter.factory == "reactionflow.adapters.ani1xnr:create_adapter"
    assert len(campaign.structure.read_text(encoding="utf-8").splitlines()) == 194

    script = (example / "submit.sbatch").read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=4",
        "#SBATCH --ntasks-per-node=4",
        "#SBATCH --gpus-per-task=1",
        "#SBATCH --cpus-per-task=32",
    ):
        assert directive in script
    assert "module load pytorch/2.11.0" in script
    assert "srun --kill-on-bad-exit=1 --cpu-bind=cores" in script

    raw = json.loads((example / "campaign.json").read_text(encoding="utf-8"))
    assert raw["adapter"]["options"]["model_index"] == 0


def test_setup_uses_nersc_pytorch_module_and_pinned_weights() -> None:
    root = Path(__file__).parents[1]
    setup = (root / "scripts/setup-perlmutter-ani1xnr.sh").read_text(encoding="utf-8")
    cache = (root / "scripts/cache_ani1xnr.py").read_text(encoding="utf-8")

    assert "module load pytorch/2.11.0" in setup
    assert "PYTHONUSERBASE" in setup
    assert "MODEL_REVISION" in cache
    assert "MODEL_SHA256" in cache
