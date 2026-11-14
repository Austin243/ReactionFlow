"""EON campaign commands; validation and status never load a model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read

from ..campaign import AdapterSpec, TrajectorySpec, _reject_unknown
from ..detection import canonical_copy
from ..mlip import load_mlip_adapter
from ..model_setup import install_and_prepare, prepare_adapter, prepare_campaign
from .akmc import AKMCConfig, confidence, rate
from .eon import EON_VERSION, EONBackend, EONSettings, free_rotation
from .labels import describe
from .records import SearchStore
from .runner import kept_processes, read_search_state, run_exploration


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
        known = {
            "schema_version",
            "mode",
            "structure",
            "output_root",
            "adapter",
            "require_gpu",
            "akmc",
            "eon",
        }
        _reject_unknown(data, known, "EON campaign keys")

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

        spec = TrajectorySpec(
            id="eon",
            total_steps=1,
            timestep_fs=1.0,
            temperature_K=self.akmc.temperature_K,
            pressure_GPa=None,
            seed=self.akmc.seed,
        )
        return (spec,)

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


def _last_error(root: Path) -> dict[str, str]:
    """The error that ended the last run, as a report entry, or nothing."""

    try:
        return {"last_error": json.loads((root / "last-error.json").read_text())["error"]}
    except FileNotFoundError:
        return {}
    except Exception as error:
        return {"last_error": f"unreadable last-error.json: {type(error).__name__}"}


def search_status(campaign: SearchCampaign) -> dict[str, Any]:
    """Read progress and records without creating directories or loading resources."""

    root = campaign.output_root
    report: dict[str, Any] = {"mode": "eon", "output_root": str(root), **_last_error(root)}
    if not (root / "search-state.json").exists():
        records = sum(
            1
            for kind in ("states", "processes", "attempts", "steps")
            for _ in (root / kind).glob("*")
        )
        status = (
            {"status": "state_missing", "records": records}
            if records
            else {"status": "not_started"}
        )
        return report | status | {"steps": 0, "time_s": 0.0, "attempts": 0}
    state = read_search_state(root)
    config = AKMCConfig(**json.loads((root / "search-contract.json").read_text())["akmc"])
    store = SearchStore(root)
    processes = {pid: store.read("processes", pid).data for pid in store.ids("processes")}
    attempts = [store.read("attempts", name).data for name in store.ids("attempts")]
    steps = [store.read("steps", name).data for name in store.ids("steps")]
    current = state["current"]
    counters = state["states"][current]
    table = [{**processes[pid], "id": pid} for pid in counters["processes"]]
    kept = {process["id"] for process in kept_processes(table, config)}
    saddles = [item["minima"] for item in attempts if "minima" in item]
    reasons = Counter(
        str(item.get("message") or "no message") for item in attempts if item["status"] == "failed"
    )
    if reasons:
        report["failure_reasons"] = dict(reasons.most_common())
    return report | {
        "status": "stopped" if state["stop_reason"] else "incomplete",
        "stop_reason": state["stop_reason"],
        "temperature_K": config.temperature_K,
        "basins": config.basins,
        "chemical_steps": state.get("chemical_steps"),
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
        "failed_with_saddle": len(saddles),
        "failed_with_bond_change": sum(
            any(side["bonds"]["formed"] or side["bonds"]["broken"] for side in minima)
            for minima in saddles
        ),
        "step_rows": [
            {
                "step": index,
                **{key: step[key] for key in ("from", "to", "process", "dt_s", "time_s")},
                "barrier_eV": processes[step["process"]]["barrier_eV"],
                "bonds": describe(processes[step["process"]]["bonds"]),
                "chemical": step.get("chemical"),
                "basin_states": step.get("basin_states"),
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
                "in_window": process["id"] in kept,
                "bonds": describe(process["bonds"]),
            }
            for process in table
        ],
    }


def _format_status(report: dict[str, Any]) -> str:
    error = [f"Last error: {report['last_error']}"] if "last_error" in report else []
    if report["status"] == "not_started":
        return "\n".join(["EON AKMC: not started", *error])
    if report["status"] == "state_missing":
        line = f"EON AKMC: search-state.json is missing, with {report['records']} records present"
        return "\n".join([line, *error])
    chemical = report["chemical_steps"]
    steps = report["steps"] if chemical is None else f"{chemical} chemical of {report['steps']}"
    lines = [
        f"EON AKMC: {report['status']} ({report['stop_reason'] or 'no stop reason'}) "
        f"at {report['temperature_K']:g} K",
        f"Steps: {steps}; simulated time {report['time_s']:.4g} s",
        f"Current {report['current']}: confidence {report['current_confidence']:.3f} after "
        f"{report['current_searches']} searches; known processes: "
        f"{len(report['current_processes'])}",
        f"States: {report['states']}; processes: {report['processes']}; searches: "
        f"{report['attempts']} {json.dumps(report['attempt_statuses'], sort_keys=True)}",
    ]
    lines += error
    reasons = report.get("failure_reasons", {})
    if reasons:
        shown = [f"{count} {reason}" for reason, count in list(reasons.items())[:3]]
        more = f"; and {len(reasons) - 3} other reasons" if len(reasons) > 3 else ""
        lines.append(f"Failed searches: {'; '.join(shown)}{more}")
    if report["failed_with_saddle"]:
        lines.append(
            f"Failed searches that kept a saddle: {report['failed_with_saddle']}, of which "
            f"{report['failed_with_bond_change']} reach a minimum with a bond change"
        )
    for row in report["step_rows"]:
        kind = ""
        if row["chemical"] is not None:
            name = "chemical" if row["chemical"] else "into unexplored conformer"
            kind = f"{name}, basin of {row['basin_states']}; "
        lines.append(
            f"step {row['step']}: {row['from']} -> {row['to']} via {row['process']} "
            f"({kind}{row['barrier_eV']:.4g} eV; {row['bonds']}), dt {row['dt_s']:.4g} s, "
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


@contextmanager
def _keeping_last_error(root: Path) -> Iterator[None]:
    """Clear the last run's error, and keep this run's for `status` if it ends in one."""

    path = root / "last-error.json"
    with suppress(OSError):
        path.unlink()
    try:
        yield
    except Exception as error:
        with suppress(OSError):
            root.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"error": f"{type(error).__name__}: {error}"}) + "\n")
        raise


def _validate(campaign: SearchCampaign) -> None:
    atoms = campaign.starting_atoms()
    report = {
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
        "warnings": [PINNED] if free_rotation(atoms) == "pinned" else [],
    }
    print(json.dumps(report, sort_keys=True))


def _run(campaign: SearchCampaign, arguments: argparse.Namespace) -> None:
    from ..cli import resolve_task_index, visible_gpu

    resolve_task_index(1, arguments.index)
    if campaign.require_gpu:
        visible_gpu()
    if arguments.download:
        prepare_adapter(campaign.adapter)
    atoms = campaign.starting_atoms()
    adapter = load_mlip_adapter(campaign.adapter, campaign.trajectories[0])
    preflight = getattr(adapter, "preflight", None)
    if callable(preflight):
        preflight(atoms)
    field = getattr(adapter, "external_field", None)
    if campaign.eon.rotation_for(atoms) and callable(field) and field(atoms).any():
        raise ValueError("an external field breaks rotation symmetry; set eon.rigid_rotation false")
    lease = getattr(adapter, "_lease", None)
    if lease is None:
        raise TypeError("EON needs a built-in model adapter or reactionflow.adapters.ase")
    if free_rotation(atoms) == "pinned":
        print(f"warning: {PINNED}", file=sys.stderr)
    log = campaign.output_root / "eon.log" if campaign.eon.log == "file" else None
    # One model load gives the calculator and the same identity an MD restart records.
    with lease() as (calculator, identity), _eon_output(log):
        backend = EONBackend(calculator, campaign.eon, campaign.akmc.temperature_K)
        contract = {
            "adapter": asdict(campaign.adapter),
            "calculator": asdict(identity),
            "eon": backend.contract,
        }
        run_exploration(
            campaign.output_root, atoms, backend=backend, contract=contract, config=campaign.akmc
        )
    print(json.dumps(search_status(campaign), sort_keys=True))


def handle_command(arguments: argparse.Namespace) -> int:
    campaign = SearchCampaign.load(arguments.campaign)
    command = arguments.command
    if command == "prepare":
        if arguments.install:
            extra = (f"pyeonclient[ase]=={EON_VERSION}",)
            return install_and_prepare(campaign, index=arguments.index, extra_packages=extra)
        models = prepare_campaign(campaign, index=arguments.index)
        print(json.dumps({"campaign": str(campaign.source), "models": models}, sort_keys=True))
    elif command == "status":
        report = search_status(campaign)
        if arguments.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(_format_status(report))
    elif command == "validate":
        _validate(campaign)
    else:
        with _keeping_last_error(campaign.output_root):
            _run(campaign, arguments)
    return 0
