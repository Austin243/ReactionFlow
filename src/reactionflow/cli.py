"""Command-line entry point for independent trajectory campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, replace
from numbers import Integral
from pathlib import Path
from uuid import uuid4

from ase.io import read

from ._durable import ensure_directory, file_digest, publish, sync_directory
from ._version import __version__
from .campaign import CampaignConfig
from .mlip import load_mlip_adapter
from .model_setup import install_and_prepare, model_catalog, prepare_adapter, prepare_campaign
from .pathway import check_saddle
from .run import ReactionRun, ReactionRunConfig, RunSummary
from .scan import ScanResult, file_frames, recorded_occurrences, run_frames, scan_frames
from .status import campaign_status, format_status
from .wizard import create_campaign


def _trajectory_contract(campaign: CampaignConfig, index: int) -> dict[str, object]:
    trajectory = campaign.trajectory(index)
    adapter = campaign.adapter_for(index)
    scientific_config = {
        "adapter": {
            "factory": adapter.factory,
            "options": dict(adapter.options),
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
        "structure_sha256": file_digest(campaign.structure_for(index)),
    }


def _bind_trajectory_contract(root: Path, contract: dict[str, object]) -> None:
    path = root / "trajectory-contract.json"
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored == contract:
            sync_directory(root)
            return
        if (root / "state.json").exists():
            raise ValueError(
                "campaign structure or scientific configuration changed for an existing trajectory"
            )
    if (root / "state.json").exists():
        raise ValueError("existing trajectory is missing its campaign contract")
    ensure_directory(root)
    temporary = root / f".trajectory-contract-{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish(temporary, path)
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
    download: bool = False,
) -> RunSummary:
    """Run or exactly resume one selected trajectory without spawning workers."""

    trajectory = campaign.trajectory(index)
    adapter_spec = campaign.adapter_for(index)
    if campaign.require_gpu:
        visible_gpu(environment)
    if download:
        prepare_adapter(adapter_spec)
    root = campaign.output_root / trajectory.id
    state_path = root / "state.json"
    if state_path.is_file():
        _bind_trajectory_contract(root, _trajectory_contract(campaign, index))
        adapter = load_mlip_adapter(adapter_spec, trajectory)
        run = ReactionRun.open(root)
        atoms = read(campaign.structure_for(index)) if run.phase == "new" else None
    else:
        atoms = read(campaign.structure_for(index))
        adapter = load_mlip_adapter(adapter_spec, trajectory)
        preflight = getattr(adapter, "preflight", None)
        if callable(preflight):
            preflight(atoms)
        _bind_trajectory_contract(root, _trajectory_contract(campaign, index))
        run = ReactionRun.create(root, config=campaign.reaction_run)
    return run.run(
        atoms,
        runtime_provider=adapter,
        pathway_calculator_provider=adapter.calculator,
        total_steps=trajectory.total_steps,
        pressure_GPa=trajectory.pressure_GPa,
    )


def check_band(
    campaign: CampaignConfig,
    band: Path,
    *,
    index: int,
    pressure_GPa: float | None,
) -> tuple[Path, dict[str, object]]:
    """Test a band's saddle with one trajectory's model and write the verdict beside the band."""

    images = read(band, ":")
    trajectory = campaign.trajectory(index)
    adapter = load_mlip_adapter(campaign.adapter_for(index), trajectory)
    with adapter.calculator("check") as calculator:
        outcome = check_saddle(
            images,
            calculator,
            detector_config=campaign.reaction_run.detector,
            config=campaign.reaction_run.pathway,
            pressure_GPa=pressure_GPa,
        )
    result: dict[str, object] = {
        "band": str(band.resolve()),
        "trajectory_id": trajectory.id,
        "reactionflow_version": __version__,
        **outcome.to_dict(),
    }
    del result["method"]  # how the band was made is not known here
    path = band.with_name(f"{band.stem}.check.json")
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        publish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path, result


def _scan_settings(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace, base: ReactionRunConfig
) -> dict[str, object]:
    """A run's detection and tracking settings, with the ones the command line changes."""

    radius = base.candidate_region_radius
    if arguments.radius == "none":
        radius = None
    elif arguments.radius is not None:
        if not arguments.radius.isdigit():
            parser.error("--radius takes a non-negative integer or none")
        radius = int(arguments.radius)
    persistence = arguments.persistence
    if persistence is None:
        persistence = base.detector.persistence_frames
    try:
        detector = replace(
            base.detector,
            persistence_frames=persistence,
            pair_thresholds={
                **base.detector.pair_thresholds,
                **{pair: (form, breaking) for pair, form, breaking in arguments.pair},
            },
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    stability = arguments.stability
    if stability is None:
        stability = base.candidate_stability_frames
    elif stability < 1:
        parser.error("--stability must be positive")
    if arguments.every < 1:
        parser.error("--every must be positive")
    return {
        "detector_config": detector,
        "stability_frames": stability,
        "region_radius": radius,
        "species_every": arguments.every,
    }


def _print_scan(path: Path, result: ScanResult, **identity: object) -> None:
    counts = {key: value for key, value in result.summary.items() if key != "settings"}
    print(json.dumps({"scan": str(path), **identity, **counts}, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reactionflow")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="write a campaign file by answering questions")
    init.add_argument("campaign", type=Path, nargs="?", default=Path("campaign.json"))

    models = commands.add_parser("models", help="list built-in models, heads and tasks offline")
    models.add_argument("--backend", help="show only one backend, for example mace or uma")

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
    run.add_argument(
        "--download", action="store_true", help="prepare the selected built-in model before running"
    )

    prepare = commands.add_parser(
        "prepare", help="download built-in model files without creating a trajectory"
    )
    prepare.add_argument("campaign", type=Path)
    prepare.add_argument("--index", type=int, help="prepare only this trajectory's adapter")
    prepare.add_argument(
        "--install", action="store_true", help="install the selected optional backend packages"
    )

    check = commands.add_parser(
        "check", help="test whether a band's highest image is a saddle between its two ends"
    )
    check.add_argument("campaign", type=Path)
    check.add_argument("band", type=Path, help="a band file ASE can read, both ends included")
    check.add_argument("--index", type=int, help="the trajectory whose model and pressure to use")
    check.add_argument("--pressure", type=float, help="GPa, in place of the trajectory's pressure")

    scan = commands.add_parser(
        "scan", help="replay saved frames through bond detection without loading the model"
    )
    scan.add_argument(
        "campaign",
        type=Path,
        metavar="source",
        help="a campaign .json file, or any trajectory file ASE can read",
    )
    scan.add_argument("--index", type=int, help="scan only this campaign trajectory")
    scan.add_argument("--radius", help="region radius in bonds, or none for whole components")
    scan.add_argument("--persistence", type=int, help="observations a bond change must last")
    scan.add_argument("--stability", type=int, help="observations a reaction region must hold")
    scan.add_argument(
        "--pair",
        nargs=3,
        action="append",
        default=[],
        metavar=("A-B", "FORM", "BREAK"),
        help="formation and break distances in Å for one element pair",
    )
    scan.add_argument("--every", type=int, default=10, help="frames between species counts")
    scan.add_argument("--frame-ps", type=float, help="ps between the frames of a trajectory file")
    scan.add_argument("--out", type=Path, help="output directory")

    status = commands.add_parser(
        "status", help="summarize trajectories and reactions without changing any run"
    )
    status.add_argument("campaign", type=Path)
    status.add_argument("--json", action="store_true", help="print the report as JSON")
    status.add_argument(
        "--radius",
        type=int,
        default=2,
        help="bonds around a bond change that define its reaction class (default 2)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "init":
        return create_campaign(arguments.campaign)
    if arguments.command == "models":
        try:
            catalog = model_catalog(arguments.backend)
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps({"backends": catalog}, indent=2, sort_keys=True))
        return 0
    if arguments.command == "scan" and arguments.campaign.suffix != ".json":
        if arguments.index is not None:
            parser.error("--index picks a campaign trajectory")
        frame_ps = arguments.frame_ps
        if frame_ps is not None and not (math.isfinite(frame_ps) and frame_ps > 0):
            parser.error("--frame-ps must be a positive number")
        settings = _scan_settings(parser, arguments, ReactionRunConfig())
        result = scan_frames(file_frames(arguments.campaign, frame_ps=frame_ps), **settings)
        source = arguments.campaign
        _print_scan(result.write(arguments.out or source.with_name(f"{source.name}-scan")), result)
        return 0
    raw = json.loads(arguments.campaign.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("campaign must be a JSON object")
    mode = raw.get("mode", "md")
    if arguments.command == "check" and mode != "md":
        parser.error("check takes an MD campaign file; it supplies the model and bond thresholds")
    if arguments.command == "scan" and mode != "md":
        parser.error("scan reads the frames an MD campaign saves; an EON campaign saves none")
    if mode == "eon":
        from .search.cli import handle_command

        return handle_command(arguments)
    if mode != "md":
        raise ValueError("campaign.mode must be 'md' or 'eon'")
    campaign = CampaignConfig.load(arguments.campaign)
    if arguments.command == "prepare":
        if arguments.install:
            return install_and_prepare(campaign, index=arguments.index)
        models = prepare_campaign(campaign, index=arguments.index)
        print(json.dumps({"campaign": str(campaign.source), "models": models}, sort_keys=True))
        return 0
    if arguments.command == "status":
        if arguments.radius < 0:
            parser.error("--radius must not be negative")
        report = campaign_status(campaign, radius=arguments.radius)
        print(
            json.dumps(report, indent=2, sort_keys=True)
            if arguments.json
            else format_status(report)
        )
        return 0
    if arguments.command == "scan":
        if arguments.frame_ps is not None:
            parser.error("--frame-ps is for a trajectory file; a run records its own times")
        indices = range(len(campaign.trajectories))
        if arguments.index is not None:
            indices = [resolve_task_index(len(indices), arguments.index, environment={})]
        for index in indices:
            trajectory = campaign.trajectory(index)
            root = campaign.output_root / trajectory.id
            if not (root / "segments").is_dir():
                missing = {"scan": None, "trajectory_id": trajectory.id, "message": "not started"}
                print(json.dumps(missing))
                continue
            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            base = ReactionRunConfig.from_dict(state["config"])
            settings = _scan_settings(parser, arguments, base)
            result = scan_frames(
                run_frames(root, timestep_fs=trajectory.timestep_fs),
                occurrences=recorded_occurrences(root),
                **settings,
            )
            out = root / "scan" if arguments.out is None else arguments.out / trajectory.id
            _print_scan(result.write(out), result, trajectory_id=trajectory.id)
        return 0
    if arguments.command in {"validate", "plan"}:
        payload = {
            "campaign": str(campaign.source),
            "trajectories": len(campaign.trajectories),
            "require_gpu": campaign.require_gpu,
            "structure_atoms": {
                os.path.relpath(path, campaign.source.parent): len(read(path))
                for path in dict.fromkeys(campaign.trajectory_structures)
            },
            "adapter_profile_counts": dict(Counter(campaign.trajectory_profiles)),
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

    if arguments.command == "check":
        if arguments.index is None and len(campaign.trajectories) > 1:
            parser.error("--index picks the trajectory when a campaign has several")
        # Slurm tasks pick trajectories for run only, so a check inside any job reads --index.
        index = resolve_task_index(len(campaign.trajectories), arguments.index, environment={})
        pressure = arguments.pressure
        if pressure is None:
            pressure = campaign.trajectory(index).pressure_GPa
        elif not math.isfinite(pressure):
            parser.error("--pressure must be a finite number")
        path, result = check_band(campaign, arguments.band, index=index, pressure_GPa=pressure)
        summary = {key: result[key] for key in ("status", "barrier_eV", "message")}
        print(json.dumps({"check": str(path), **summary}, sort_keys=True))
        return 0

    index = resolve_task_index(len(campaign.trajectories), arguments.index)
    trajectory_id = campaign.trajectory(index).id
    payload: dict[str, object] = {
        "trajectory_id": trajectory_id,
        "trajectory_index": index,
        "hostname": socket.gethostname(),
    }
    # Every Slurm task shares one log, so each worker also records its own error by trajectory.
    error_path = campaign.output_root / trajectory_id / "last-error.json"
    with suppress(OSError):
        error_path.unlink()
    try:
        summary = run_selected_trajectory(campaign, index=index, download=arguments.download)
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(payload, sort_keys=True))
        temporary = error_path.with_name(f".last-error-{uuid4().hex}.tmp")
        with suppress(OSError):
            try:
                error_path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                publish(temporary, error_path)
            finally:
                temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({**payload, **asdict(summary)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["check_band", "main", "resolve_task_index", "run_selected_trajectory", "visible_gpu"]
