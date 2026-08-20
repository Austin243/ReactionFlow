from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from reactionflow import ComponentState, ExactRestartSnapshot
from reactionflow.campaign import CampaignConfig
from reactionflow.cli import main, resolve_task_index, run_selected_trajectory, visible_gpu


class _TestRuntime:
    def __init__(self, atoms: Atoms, *, nsteps: int = 0) -> None:
        self._atoms = atoms
        self._nsteps = nsteps

    @property
    def atoms(self) -> Atoms:
        return self._atoms

    @property
    def nsteps(self) -> int:
        return self._nsteps

    def run(self, steps: int) -> None:
        self._nsteps += steps

    def snapshot(self) -> ExactRestartSnapshot:
        return ExactRestartSnapshot(
            atoms=self._atoms,
            dynamics=ComponentState(
                kind="test.campaign-runtime",
                metadata={"nsteps": self._nsteps},
            ),
            calculator=ComponentState(kind="test.campaign-calculator"),
        )


class _TestAdapter:
    def __init__(self, trajectory, options) -> None:
        self.trajectory = trajectory
        self.options = options

    @contextmanager
    def start(self, atoms: Atoms):
        yield _TestRuntime(atoms)

    @contextmanager
    def restore(self, snapshot: ExactRestartSnapshot):
        yield _TestRuntime(snapshot.atoms, nsteps=int(snapshot.dynamics.metadata["nsteps"]))

    @contextmanager
    def calculator(self, _stage: str):
        raise AssertionError("a single inert atom should not launch a pathway")
        yield


FACTORY_CALLS: list[tuple[str, dict[str, object]]] = []


def create_test_adapter(*, trajectory, options):
    FACTORY_CALLS.append((trajectory.id, options))
    return _TestAdapter(trajectory, options)


def _campaign(tmp_path, *, count: int = 1, require_gpu: bool = False):
    structure = Atoms("He", positions=[[0, 0, 0]])
    structure.set_array("source_marker", np.asarray([7]))
    write(tmp_path / "structure.extxyz", structure)
    value = {
        "schema_version": 1,
        "structure": "structure.extxyz",
        "output_root": "runs",
        "require_gpu": require_gpu,
        "adapter": {
            "factory": f"{__name__}:create_test_adapter",
            "options": {"model": "test-model"},
        },
        "reaction_run": {"observation_interval": 1},
        "trajectories": [
            {
                "id": f"trajectory-{index:04d}",
                "total_steps": 2,
                "timestep_fs": 1.0,
                "temperature_K": 100.0 + index,
                "pressure_GPa": 20.0,
                "seed": index,
                "conditions": {"integrator": "npt"},
            }
            for index in range(count)
        ],
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_campaign_loads_relative_paths_and_arbitrary_trajectory_count(tmp_path) -> None:
    campaign = CampaignConfig.load(_campaign(tmp_path, count=130))

    assert campaign.structure == tmp_path / "structure.extxyz"
    assert campaign.output_root == tmp_path / "runs"
    assert len(campaign.trajectories) == 130
    assert campaign.trajectory(129).temperature_K == 229.0
    assert campaign.trajectory(129).conditions == {"integrator": "npt"}
    assert campaign.reaction_run.observation_interval == 1


def test_task_mapping_requires_exactly_one_process_per_trajectory() -> None:
    assert (
        resolve_task_index(
            130,
            None,
            environment={"SLURM_NTASKS": "130", "SLURM_PROCID": "129"},
        )
        == 129
    )
    assert resolve_task_index(1, None, environment={}) == 0
    with pytest.raises(ValueError, match="exactly one task per trajectory"):
        resolve_task_index(
            130,
            None,
            environment={"SLURM_NTASKS": "128", "SLURM_PROCID": "0"},
        )
    with pytest.raises(ValueError, match="use --index"):
        resolve_task_index(2, None, environment={})
    with pytest.raises(ValueError, match="conflicts"):
        resolve_task_index(2, 0, environment={"SLURM_PROCID": "1"})


def test_gpu_worker_accepts_one_visible_device_and_rejects_shared_visibility() -> None:
    assert visible_gpu({"CUDA_VISIBLE_DEVICES": "GPU-abc"}) == "GPU-abc"
    with pytest.raises(RuntimeError, match="exactly one"):
        visible_gpu({"CUDA_VISIBLE_DEVICES": "0,1"})
    with pytest.raises(RuntimeError, match="exactly one"):
        visible_gpu({})


def test_selected_trajectory_runs_and_relaunch_is_idempotent(tmp_path) -> None:
    FACTORY_CALLS.clear()
    path = _campaign(tmp_path)
    campaign = CampaignConfig.load(path)

    first = run_selected_trajectory(campaign, index=0, environment={})
    second = run_selected_trajectory(campaign, index=0, environment={})

    assert (first.phase, first.global_step) == ("completed", 2)
    assert second == first
    assert FACTORY_CALLS == [
        ("trajectory-0000", {"model": "test-model"}),
        ("trajectory-0000", {"model": "test-model"}),
    ]
    state = json.loads((tmp_path / "runs/trajectory-0000/state.json").read_text())
    assert state["phase"] == "completed"
    contract = json.loads((tmp_path / "runs/trajectory-0000/trajectory-contract.json").read_text())
    assert contract["trajectory_id"] == "trajectory-0000"

    changed = json.loads(path.read_text())
    changed["trajectories"][0]["temperature_K"] = 999.0
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="scientific configuration changed"):
        run_selected_trajectory(CampaignConfig.load(path), index=0, environment={})


def test_cli_plan_has_no_campaign_size_ceiling(tmp_path, capsys) -> None:
    path = _campaign(tmp_path, count=130)

    assert main(["plan", str(path), "--gpus-per-node", "4"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan == {
        "atoms": 1,
        "campaign": str(path),
        "gpus": 130,
        "gpus_per_node": 4,
        "minimum_nodes": 33,
        "require_gpu": False,
        "tasks": 130,
        "trajectories": 130,
    }


def test_cli_run_maps_slurm_rank_and_one_gpu_to_one_trajectory(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    path = _campaign(tmp_path, require_gpu=True)
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert main(["run", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["trajectory_id"] == "trajectory-0000"
    assert result["trajectory_index"] == 0
    assert result["phase"] == "completed"
    assert result["global_step"] == 2


def test_perlmutter_template_maps_four_local_cpu_monitors_to_four_gpus() -> None:
    script = (Path(__file__).parents[1] / "examples/perlmutter/run-campaign.sbatch").read_text()

    assert "#SBATCH --ntasks-per-node=4" in script
    assert "#SBATCH --gpus-per-task=1" in script
    assert "#SBATCH --cpus-per-task=32" in script
    assert 'srun --cpu-bind=cores reactionflow run "$campaign"' in script
    assert "--ntasks=32" not in script
