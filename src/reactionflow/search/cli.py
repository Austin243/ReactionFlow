"""EON campaign commands; validation and status never load a model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read

from ..campaign import AdapterSpec, TrajectorySpec
from ..detection import canonical_copy
from ..mlip import load_mlip_adapter
from ..model_setup import install_and_prepare, prepare_adapter, prepare_campaign
from .akmc import AKMCConfig, confidence, in_window, rate
from .eon import EON_VERSION, EONBackend, EONSettings, free_rotation
from .labels import describe
from .records import SearchStore
from .runner import read_search_state, run_exploration


def _object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _settings(cls, value: Any, label: str):
    data = _object(value, label)
    try:
        return cls(**data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {label} settings: {error}") from error


@dataclass(frozen=True)
class SearchCampaign:
    source: Path
    structure: Path
    output_root: Path
    adapter: AdapterSpec
    require_gpu: bool
    akmc: AKMCConfig
    eon: EONSettings

    @classmethod
    def load(cls, path: str | Path) -> SearchCampaign:
        source = Path(path).resolve()
        data = _object(json.loads(source.read_text(encoding="utf-8")), "campaign")
        json.dumps(data, allow_nan=False)
        if type(data.get("schema_version")) is not int or data["schema_version"] != 2:
            raise ValueError("unsupported campaign schema; use schema_version 2")
        if data.get("mode") != "eon":
            raise ValueError("search campaign requires mode 'eon'")
        unknown = data.keys() - {
            "schema_version",
            "mode",
            "structure",
            "output_root",
            "adapter",
            "require_gpu",
            "akmc",
            "eon",
        }
        if unknown:
            raise ValueError(f"unknown EON campaign keys: {sorted(unknown)}")

        def resolve(value: Any, label: str) -> Path:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty path string")
            return (source.parent / Path(value).expanduser()).resolve()

        require_gpu = data.get("require_gpu", True)
        if type(require_gpu) is not bool:
            raise TypeError("campaign.require_gpu must be a boolean")
        return cls(
            source=source,
            structure=resolve(data.get("structure"), "structure"),
            output_root=resolve(data.get("output_root"), "output_root"),
            adapter=AdapterSpec.from_dict(data.get("adapter")),
            require_gpu=require_gpu,
            akmc=_settings(AKMCConfig, data.get("akmc", {}), "akmc"),
            eon=_settings(EONSettings, data.get("eon", {}), "eon"),
        )

    @property
    def trajectories(self) -> tuple[TrajectorySpec]:
        """One fixed-cell search, shaped like a one-trajectory campaign for model setup.

        Adapters are built for MD. EON uses only their calculator, so the MD fields are unused.
        """

        return (
            TrajectorySpec(
                id="eon",
                total_steps=1,
                timestep_fs=1.0,
                temperature_K=self.akmc.temperature_K,
                pressure_GPa=None,
                seed=self.akmc.seed,
            ),
        )

    def adapter_for(self, index: int) -> AdapterSpec:
        if index != 0:
            raise IndexError(f"EON campaigns have one task; index {index} is outside it")
        return self.adapter

    def starting_atoms(self) -> Atoms:
        atoms = canonical_copy(read(self.structure))
        if free_rotation(atoms) and self.eon.prefactor is None:
            raise ValueError(
                "harmonic prefactors need a structure that cannot turn as a whole: a free "
                "molecule's rate also depends on its rotations, which EON does not include. "
                "Give eon.prefactor a fixed value, fix three atoms not on one line, or use a "
                "periodic cell"
            )
        return atoms


PINNED = (
    "the fixed atoms do not stop the structure turning about them, and EON removes rotation "
    "only when no atom is fixed; fix at least three atoms not on one line, or none"
)


def search_status(campaign: SearchCampaign) -> dict[str, Any]:
    """Read progress and records without creating directories or loading resources."""

    root = campaign.output_root
    report: dict[str, Any] = {"mode": "eon", "output_root": str(root)}
    if not (root / "search-state.json").exists():
        return report | {"status": "not_started", "steps": 0, "time_s": 0.0, "attempts": 0}
    state = read_search_state(root)
    config = AKMCConfig(**json.loads((root / "search-contract.json").read_text())["akmc"])
    store = SearchStore(root)
    processes = {pid: store.read("processes", pid).data for pid in store.ids("processes")}
    attempts = [store.read("attempts", name).data for name in store.ids("attempts")]
    steps = [store.read("steps", name).data for name in store.ids("steps")]
    current = state["current"]
    counters = state["states"][current]
    table = [{**processes[pid], "id": pid} for pid in counters["processes"]]
    lowest = min((process["barrier_eV"] for process in table), default=0.0)
    return report | {
        "status": "stopped" if state["stop_reason"] else "incomplete",
        "stop_reason": state["stop_reason"],
        "temperature_K": config.temperature_K,
        "steps": state["steps"],
        "time_s": state["time_s"],
        "current": current,
        "current_confidence": confidence(counters["repeats"]),
        "current_searches": counters["searches"],
        "active": state["active"],
        "states": len(state["states"]),
        "processes": len(processes),
        "attempts": state["attempts"],
        "attempt_statuses": dict(Counter(item["status"] for item in attempts)),
        "step_rows": [
            {
                "step": index,
                **{key: step[key] for key in ("from", "to", "process", "dt_s", "time_s")},
                "barrier_eV": processes[step["process"]]["barrier_eV"],
                "bonds": describe(processes[step["process"]]["bonds"]),
            }
            for index, step in enumerate(steps)
        ],
        "current_processes": [
            {
                "process": process["id"],
                "to": process["product"],
                "barrier_eV": process["barrier_eV"],
                "prefactor_s": process["prefactor_s"],
                "rate_s": rate(process["prefactor_s"], process["barrier_eV"], config.kT),
                "in_window": in_window(process["barrier_eV"], lowest, config),
                "bonds": describe(process["bonds"]),
            }
            for process in table
        ],
    }


def _format_status(report: dict[str, Any]) -> str:
    if report["status"] == "not_started":
        return "EON AKMC: not started"
    lines = [
        f"EON AKMC: {report['status']} ({report['stop_reason'] or 'no stop reason'}) "
        f"at {report['temperature_K']:g} K",
        f"Steps: {report['steps']}; simulated time {report['time_s']:.4g} s",
        f"Current {report['current']}: confidence {report['current_confidence']:.3f} after "
        f"{report['current_searches']} searches; known processes: "
        f"{len(report['current_processes'])}",
        f"States: {report['states']}; processes: {report['processes']}; searches: "
        f"{report['attempts']} {json.dumps(report['attempt_statuses'], sort_keys=True)}",
    ]
    for row in report["step_rows"]:
        lines.append(
            f"step {row['step']}: {row['from']} -> {row['to']} via {row['process']} "
            f"({row['barrier_eV']:.4g} eV; {row['bonds']}), dt {row['dt_s']:.4g} s, "
            f"t {row['time_s']:.4g} s"
        )
    return "\n".join(lines)


@contextmanager
def _eon_output(path: Path | None) -> Iterator[None]:
    """Send what EON's own code writes to standard output into `path`, when one is given."""

    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with path.open("a", encoding="utf-8") as log:
            os.dup2(log.fileno(), 1)
            try:
                yield
            finally:
                # EON writes its log from a background thread; let it finish first.
                time.sleep(0.2)
                os.dup2(saved, 1)
    finally:
        os.close(saved)


def handle_command(arguments: argparse.Namespace) -> int:
    campaign = SearchCampaign.load(arguments.campaign)
    command = arguments.command
    if command == "prepare":
        if arguments.install:
            return install_and_prepare(
                campaign,
                index=arguments.index,
                extra_packages=(f"pyeonclient[ase]=={EON_VERSION}",),
            )
        models = prepare_campaign(campaign, index=arguments.index)
        print(json.dumps({"campaign": str(campaign.source), "models": models}, sort_keys=True))
        return 0
    if command == "status":
        report = search_status(campaign)
        print(
            json.dumps(report, indent=2, sort_keys=True)
            if arguments.json
            else _format_status(report)
        )
        return 0
    if command == "run":
        from ..cli import resolve_task_index, visible_gpu

        resolve_task_index(1, arguments.index)
        if campaign.require_gpu:
            visible_gpu()
        if arguments.download:
            prepare_adapter(campaign.adapter)
    atoms = campaign.starting_atoms()
    warnings = [PINNED] if free_rotation(atoms) == "pinned" else []
    if command in {"validate", "plan"}:
        print(
            json.dumps(
                {
                    "campaign": str(campaign.source),
                    "mode": "eon",
                    "atoms": len(atoms),
                    "output_root": str(campaign.output_root),
                    "tasks": 1,
                    "kmc_steps": campaign.akmc.steps,
                    "temperature_K": campaign.akmc.temperature_K,
                    "adapter_factory": campaign.adapter.factory,
                    "require_gpu": campaign.require_gpu,
                    "rigid_rotation": campaign.eon.rotation_for(atoms),
                    "warnings": warnings,
                },
                sort_keys=True,
            )
        )
        return 0
    adapter = load_mlip_adapter(campaign.adapter, campaign.trajectories[0])
    preflight = getattr(adapter, "preflight", None)
    if callable(preflight):
        preflight(atoms)
    lease = getattr(adapter, "_lease", None)
    if lease is None:
        raise TypeError("EON needs a built-in model adapter or reactionflow.adapters.ase")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    log = campaign.output_root / "eon.log" if campaign.eon.log == "file" else None
    # One model load gives the calculator and the same identity an MD restart records.
    with lease() as (calculator, identity), _eon_output(log):
        backend = EONBackend(calculator, campaign.eon, campaign.akmc.temperature_K)
        run_exploration(
            campaign.output_root,
            atoms,
            backend=backend,
            contract={
                "adapter": asdict(campaign.adapter),
                "calculator": asdict(identity),
                "eon": backend.contract,
            },
            config=campaign.akmc,
        )
    print(json.dumps(search_status(campaign), sort_keys=True))
    return 0
