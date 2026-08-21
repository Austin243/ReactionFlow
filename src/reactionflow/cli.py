"""Command-line entry point for independent trajectory campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from numbers import Integral
from pathlib import Path
from uuid import uuid4

from ase.io import read

from .campaign import CampaignConfig
from .mlip import load_mlip_adapter
from .run import ReactionRun, RunSummary


def _file_digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _trajectory_contract(campaign: CampaignConfig, index: int) -> dict[str, object]:
    trajectory = campaign.trajectory(index)
    scientific_config = {
        "adapter": {
            "factory": campaign.adapter.factory,
            "options": dict(campaign.adapter.options),
        },
        "reaction_run": campaign.reaction_run.to_dict(),
        "trajectory": asdict(trajectory),
    }
    encoded = json.dumps(
        scientific_config,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return {
        "schema_version": 1,
        "trajectory_id": trajectory.id,
        "configuration_sha256": hashlib.sha256(encoded).hexdigest(),
        "structure_sha256": _file_digest(campaign.structure),
    }


def _bind_trajectory_contract(root: Path, contract: dict[str, object]) -> None:
    path = root / "trajectory-contract.json"
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored == contract:
            return
        if (root / "state.json").exists():
            raise ValueError(
                "campaign structure or scientific configuration changed for an existing trajectory"
            )
    if (root / "state.json").exists():
        raise ValueError("existing trajectory is missing its campaign contract")
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".trajectory-contract-{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _environment_integer(environment: Mapping[str, str], name: str) -> int | None:
    raw = environment.get(name)
    if raw is None:
        return None
    if not raw.isdigit():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(raw)


def resolve_task_index(
    campaign_size: int,
    explicit_index: int | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Map one Slurm process—or one explicit local index—to one trajectory."""

    if (
        isinstance(campaign_size, bool)
        or not isinstance(campaign_size, Integral)
        or campaign_size < 1
    ):
        raise ValueError("campaign size must be a positive integer")
    variables = os.environ if environment is None else environment
    slurm_tasks = _environment_integer(variables, "SLURM_NTASKS")
    slurm_index = _environment_integer(variables, "SLURM_PROCID")
    if slurm_tasks is not None and slurm_tasks != campaign_size:
        raise ValueError(
            f"Slurm launched {slurm_tasks} tasks for {campaign_size} trajectories; "
            "use exactly one task per trajectory"
        )
    if explicit_index is None:
        if slurm_index is not None:
            index = slurm_index
        elif campaign_size == 1:
            index = 0
        else:
            raise ValueError("use --index outside Slurm when a campaign has multiple trajectories")
    else:
        if isinstance(explicit_index, bool) or not isinstance(explicit_index, Integral):
            raise TypeError("trajectory index must be an integer")
        index = int(explicit_index)
        if slurm_index is not None and index != slurm_index:
            raise ValueError("--index conflicts with SLURM_PROCID")
    if not 0 <= index < campaign_size:
        raise IndexError(f"trajectory index {index} is outside this campaign")
    return index


def visible_gpu(environment: Mapping[str, str] | None = None) -> str:
    """Require exactly one CUDA device in a GPU worker's process environment."""

    variables = os.environ if environment is None else environment
    raw = variables.get("CUDA_VISIBLE_DEVICES")
    devices = [] if raw is None else [device.strip() for device in raw.split(",")]
    if len(devices) != 1 or not devices[0] or devices[0] in {"-1", "NoDevFiles"}:
        raise RuntimeError(
            "each GPU trajectory worker requires exactly one CUDA_VISIBLE_DEVICES entry"
        )
    return devices[0]


def run_selected_trajectory(
    campaign: CampaignConfig,
    *,
    index: int,
    environment: Mapping[str, str] | None = None,
) -> RunSummary:
    """Run or exactly resume one selected trajectory without spawning workers."""

    trajectory = campaign.trajectory(index)
    if campaign.require_gpu:
        visible_gpu(environment)
    root = campaign.output_root / trajectory.id
    state_path = root / "state.json"
    if state_path.is_file():
        _bind_trajectory_contract(root, _trajectory_contract(campaign, index))
        adapter = load_mlip_adapter(campaign.adapter, trajectory)
        run = ReactionRun.open(root)
        atoms = None
    else:
        atoms = read(campaign.structure)
        adapter = load_mlip_adapter(campaign.adapter, trajectory)
        _bind_trajectory_contract(root, _trajectory_contract(campaign, index))
        run = ReactionRun.create(root, config=campaign.reaction_run)
    return run.run_exact(
        atoms,
        runtime_provider=adapter,
        pathway_calculator_provider=adapter.calculator,
        total_steps=trajectory.total_steps,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reactionflow")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate", help="validate a campaign without importing its MLIP"
    )
    validate.add_argument("campaign", type=Path)

    plan = commands.add_parser("plan", help="print scheduler-neutral campaign sizing")
    plan.add_argument("campaign", type=Path)
    plan.add_argument("--gpus-per-node", type=int, default=4)

    run = commands.add_parser("run", help="run one campaign trajectory in this process")
    run.add_argument("campaign", type=Path)
    run.add_argument("--index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    campaign = CampaignConfig.load(arguments.campaign)
    if arguments.command in {"validate", "plan"}:
        atoms = read(campaign.structure)
        payload = {
            "campaign": str(campaign.source),
            "trajectories": len(campaign.trajectories),
            "require_gpu": campaign.require_gpu,
            "atoms": len(atoms),
        }
        if arguments.command == "plan":
            if arguments.gpus_per_node < 1:
                parser.error("--gpus-per-node must be positive")
            payload.update(
                tasks=len(campaign.trajectories),
                gpus=len(campaign.trajectories),
                gpus_per_node=arguments.gpus_per_node,
                minimum_nodes=math.ceil(len(campaign.trajectories) / arguments.gpus_per_node),
            )
        print(json.dumps(payload, sort_keys=True))
        return 0

    index = resolve_task_index(len(campaign.trajectories), arguments.index)
    summary = run_selected_trajectory(campaign, index=index)
    payload = {
        "trajectory_id": campaign.trajectory(index).id,
        "trajectory_index": index,
        "hostname": socket.gethostname(),
        **asdict(summary),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "resolve_task_index", "run_selected_trajectory", "visible_gpu"]
