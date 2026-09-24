"""Read-only campaign status and reaction summary."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

from .campaign import CampaignConfig
from .candidates import ReactionCandidate, reaction_key, same_reaction
from .detection import atom_ids
from .store import _read_bundle


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _occurrences(root: Path) -> list[sqlite3.Row]:
    database = root / "reactions.sqlite3"
    if not database.is_file():
        return []
    # Read-only, so a status check can never change a run, including one that is running.
    uri = f"{database.as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        return db.execute(
            "SELECT occurrence_id, class_id, representative, bundle FROM occurrences "
            "ORDER BY sequence"
        ).fetchall()


def _describe(candidate: ReactionCandidate) -> str:
    symbols = dict(
        zip(atom_ids(candidate.reactant), candidate.reactant.get_chemical_symbols(), strict=True)
    )
    changes = Counter(
        f"{kind} {'-'.join(sorted(symbols[atom_id] for atom_id in bond))}"
        for kind, bonds in (
            ("formed", candidate.product_bonds - candidate.reactant_bonds),
            ("broken", candidate.reactant_bonds - candidate.product_bonds),
        )
        for bond in bonds
    )
    return "; ".join(
        label if count == 1 else f"{label} x{count}" for label, count in sorted(changes.items())
    )


def _trajectory(root: Path, row: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Fill one trajectory's status row and return its classes as (candidate, key, events, result).

    The row is filled as reading progresses, so a damaged file later on still leaves the phase and
    step from state.json in place.
    """

    state = _read_json(root / "state.json")
    row.update(phase=state["phase"], global_step=state["global_step"])
    failure = state.get("failure")
    if state["phase"] == "failed" and failure:
        row["error"] = f"{failure.get('type')}: {failure.get('message')}"
    elif (root / "last-error.json").is_file():
        row["error"] = _read_json(root / "last-error.json").get("error")
    records = _occurrences(root)
    members: dict[str, list[sqlite3.Row]] = {}
    for record in records:
        members.setdefault(record["class_id"], []).append(record)
    classes = []
    pathways: Counter[str] = Counter()
    for occurrences in members.values():
        _, candidate = _read_bundle(root / occurrences[0]["bundle"])
        result = None
        representative = next((item for item in occurrences if item["representative"]), None)
        if representative is not None:
            path = root / "pathways" / representative["occurrence_id"] / "result.json"
            if path.is_file():
                result = _read_json(path)
                pathways[result["status"]] += 1
        classes.append((candidate, reaction_key(candidate), len(occurrences), result))
    row.update(events=len(records), pathways=dict(pathways))
    return classes


def campaign_status(campaign: CampaignConfig) -> dict[str, Any]:
    """Summarize every trajectory and merge reaction classes by model and pressure."""

    trajectories: list[dict[str, Any]] = []
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for index, spec in enumerate(campaign.trajectories):
        model = campaign.adapter_profile_for(index)
        row: dict[str, Any] = {
            "trajectory": spec.id,
            "model": model,
            "temperature_K": spec.temperature_K,
            "pressure_GPa": spec.pressure_GPa,
            "phase": "not started",
            "global_step": 0,
            "total_steps": spec.total_steps,
            "events": 0,
            "pathways": {},
            "error": None,
        }
        trajectories.append(row)
        root = campaign.output_root / spec.id
        if not (root / "state.json").is_file():
            continue
        try:
            classes = _trajectory(root, row)
        except Exception as error:  # a damaged trajectory must not hide the rest
            row["error"] = f"unreadable: {type(error).__name__}: {error}"
            continue
        for candidate, key, events, result in classes:
            bucket = buckets.setdefault((model, spec.pressure_GPa, key), [])
            entry = next(
                (item for item in bucket if same_reaction(candidate, item["_candidate"])), None
            )
            if entry is None:
                entry = {
                    "_candidate": candidate,
                    "reaction": _describe(candidate),
                    "model": model,
                    "pressure_GPa": spec.pressure_GPa,
                    "barrier_quantity": (
                        "potential_energy" if spec.pressure_GPa is None else "enthalpy"
                    ),
                    "trajectories": [],
                    "events": 0,
                    "pathways": Counter(),
                    "frequency": Counter(),
                    "barriers_eV": [],
                }
                bucket.append(entry)
            if spec.id not in entry["trajectories"]:
                entry["trajectories"].append(spec.id)
            entry["events"] += events
            if result is not None:
                entry["pathways"][result["status"]] += 1
                validation = result.get("frequency_validation")
                if validation:
                    entry["frequency"][validation["status"]] += 1
                if result["status"] == "ci_neb_converged" and result.get("barrier_eV") is not None:
                    entry["barriers_eV"].append(result["barrier_eV"])
    reactions = [
        {
            **{key: value for key, value in entry.items() if key != "_candidate"},
            "pathways": dict(entry["pathways"]),
            "frequency": dict(entry["frequency"]),
        }
        for bucket in buckets.values()
        for entry in bucket
    ]
    reactions.sort(
        key=lambda item: (
            item["model"] or "",
            item["pressure_GPa"] is not None,
            item["pressure_GPa"] or 0.0,
            -item["events"],
            item["reaction"],
        )
    )
    return {"campaign": str(campaign.source), "trajectories": trajectories, "reactions": reactions}


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(item) for item in column) for column in zip(header, *rows, strict=True)]
    return "\n".join(
        "  ".join(item.ljust(width) for item, width in zip(line, widths, strict=True)).rstrip()
        for line in (header, *rows)
    )


def _pressure(value: float | None) -> str:
    return "NVT" if value is None else f"{value:g}"


def _converged(pathways: dict[str, int]) -> str:
    total = sum(pathways.values())
    return f"{pathways.get('ci_neb_converged', 0)}/{total}" if total else "-"


def format_status(report: dict[str, Any]) -> str:
    """Render a status report as a trajectory table and a reaction table."""

    trajectory_rows = [
        [
            row["trajectory"],
            row["model"] or "-",
            _pressure(row["pressure_GPa"]),
            row["phase"],
            f"{row['global_step']}/{row['total_steps']}",
            str(row["events"]),
            _converged(row["pathways"]),
            "" if not row["error"] else row["error"][:90],
        ]
        for row in report["trajectories"]
    ]
    phases = Counter(row["phase"] for row in report["trajectories"])
    lines = [
        _table(
            ["trajectory", "model", "P (GPa)", "phase", "steps", "events", "converged", "error"],
            trajectory_rows,
        ),
        "",
        f"{len(trajectory_rows)} trajectories: "
        + ", ".join(f"{count} {phase}" for phase, count in sorted(phases.items())),
        "",
    ]
    if not report["reactions"]:
        lines.append("No reactions recorded yet.")
        return "\n".join(lines)
    reaction_rows = []
    for entry in report["reactions"]:
        barriers = entry["barriers_eV"]
        reaction_rows.append(
            [
                entry["reaction"],
                entry["model"] or "-",
                _pressure(entry["pressure_GPa"]),
                str(len(entry["trajectories"])),
                str(entry["events"]),
                _converged(entry["pathways"]),
                str(entry["frequency"].get("one_imaginary_mode", 0)),
                "-"
                if not barriers
                else " / ".join(
                    f"{value:.3f}"
                    for value in (min(barriers), statistics.median(barriers), max(barriers))
                ),
            ]
        )
    lines.append(
        _table(
            [
                "reaction",
                "model",
                "P (GPa)",
                "trajectories",
                "events",
                "converged",
                "one imaginary mode",
                "barrier eV (min / median / max)",
            ],
            reaction_rows,
        )
    )
    lines.append("")
    lines.append("NVT barriers are potential energies; NPT barriers are enthalpies.")
    return "\n".join(lines)


__all__ = ["campaign_status", "format_status"]
