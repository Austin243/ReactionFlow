"""Read-only campaign status and reaction summary."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

from ase.formula import Formula

from .campaign import CampaignConfig
from .candidates import ReactionCandidate, ReactionClasses, _reaction_graph
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


def _describe(candidate: ReactionCandidate, radius: int) -> str:
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
    changed = "; ".join(
        label if count == 1 else f"{label} x{count}" for label, count in sorted(changes.items())
    )
    # Classes with the same bond changes differ in the atoms around them, so name those atoms.
    graph = _reaction_graph(candidate, radius=radius)
    elements = [element for _, element in graph.nodes(data="element")]
    return f"{changed} [{Formula.from_list(elements).format('hill')}]"


def _trajectory(root: Path, row: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Fill one trajectory's status row and return its classes.

    Each class is (candidate, events, results, occurrence ID of the candidate).

    The row is filled as reading progresses, so a damaged file later on still leaves the phase and
    step from state.json in place.
    """

    state = _read_json(root / "state.json")
    row.update(phase=state["phase"], global_step=state["global_step"])
    failure = state.get("failure")
    if state["phase"] == "failed" and failure:
        row["error"] = f"{failure.get('type')}: {failure.get('message')}"
    records = _occurrences(root)
    members: dict[str, list[sqlite3.Row]] = {}
    for record in records:
        members.setdefault(record["class_id"], []).append(record)
    classes = []
    pathways: Counter[str] = Counter()
    for occurrences in members.values():
        _, candidate = _read_bundle(root / occurrences[0]["bundle"])
        results = []
        for occurrence in occurrences:
            path = root / "pathways" / occurrence["occurrence_id"] / "result.json"
            if path.is_file():
                result = _read_json(path)
                results.append(result)
                pathways[result["status"]] += 1
        classes.append((candidate, len(occurrences), results, occurrences[0]["occurrence_id"]))
    row.update(events=len(records), pathways=dict(pathways))
    return classes


def campaign_status(campaign: CampaignConfig, *, radius: int = 2) -> dict[str, Any]:
    """Summarize every trajectory and merge reaction classes by model and pressure.

    A class is a bond change together with the atoms within ``radius`` bonds of it, so the same
    local reaction counts once however large the molecule or polymer it happens in. Each
    trajectory's store keeps exact whole-region classes, and those merge here.
    """

    trajectories: list[dict[str, Any]] = []
    # Only barriers from the same model and pressure are comparable, so classes merge within those.
    known: dict[tuple[Any, ...], ReactionClasses] = {}
    entries: dict[str, dict[str, Any]] = {}
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
        classes = []
        if (root / "state.json").is_file():
            try:
                classes = _trajectory(root, row)
            except Exception as error:  # a damaged trajectory must not hide the rest
                row["error"] = f"unreadable: {type(error).__name__}: {error}"
        # A worker can fail before it writes state.json, so its last error is read for every
        # trajectory, and read last so that a damaged file cannot hide the reactions above.
        if row["error"] is None and (root / "last-error.json").is_file():
            try:
                row["error"] = _read_json(root / "last-error.json").get("error")
            except Exception as error:
                row["error"] = f"unreadable last-error.json: {type(error).__name__}"
        merged = known.setdefault((model, spec.pressure_GPa), ReactionClasses(radius))
        for candidate, events, results, occurrence_id in classes:
            name = merged.find(candidate)
            if name is None:
                name = str(len(entries))
                merged.add(name, candidate)
                entries[name] = {
                    "reaction": _describe(candidate, radius),
                    "example": {"trajectory": spec.id, "occurrence_id": occurrence_id},
                    "model": model,
                    "pressure_GPa": spec.pressure_GPa,
                    "barrier_quantity": (
                        "potential_energy" if spec.pressure_GPa is None else "enthalpy"
                    ),
                    "trajectories": [],
                    "events": 0,
                    "pathways": Counter(),
                    "frequency": Counter(),
                    "connectivity": Counter(),
                    "barriers_eV": [],
                }
            entry = entries[name]
            if spec.id not in entry["trajectories"]:
                entry["trajectories"].append(spec.id)
            entry["events"] += events
            for result in results:
                entry["pathways"][result["status"]] += 1
                validation = result.get("frequency_validation")
                if validation:
                    entry["frequency"][validation["status"]] += 1
                connectivity = result.get("connectivity")
                if connectivity:
                    entry["connectivity"][connectivity["status"]] += 1
                if result["status"] == "ci_neb_converged" and result.get("barrier_eV") is not None:
                    entry["barriers_eV"].append(result["barrier_eV"])
    reactions = [
        {
            **entry,
            "pathways": dict(entry["pathways"]),
            "frequency": dict(entry["frequency"]),
            "connectivity": dict(entry["connectivity"]),
        }
        for entry in entries.values()
    ]
    reactions.sort(
        key=lambda item: (
            item["model"],
            item["pressure_GPa"] is not None,
            item["pressure_GPa"] or 0.0,
            -item["events"],
            item["reaction"],
        )
    )
    return {
        "campaign": str(campaign.source),
        "radius": radius,
        "trajectories": trajectories,
        "reactions": reactions,
    }


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


def _connects(checks: dict[str, int]) -> str:
    total = sum(checks.values())
    return f"{checks.get('connects_endpoints', 0)}/{total}" if total else "-"


def format_status(report: dict[str, Any]) -> str:
    """Render a status report as a trajectory table and a reaction table."""

    trajectory_rows = [
        [
            row["trajectory"],
            row["model"],
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
                entry["model"],
                _pressure(entry["pressure_GPa"]),
                str(len(entry["trajectories"])),
                str(entry["events"]),
                _converged(entry["pathways"]),
                str(entry["frequency"].get("one_imaginary_mode", 0)),
                _connects(entry["connectivity"]),
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
                "saddle connects",
                "barrier eV (min / median / max)",
            ],
            reaction_rows,
        )
    )
    lines.append("")
    checks = sum((Counter(entry["connectivity"]) for entry in report["reactions"]), Counter())
    if checks:
        lines.append(
            f"Saddle connectivity checks: {checks['connects_endpoints']} connect, "
            f"{checks['does_not_connect']} do not connect, {checks['inconclusive']} inconclusive, "
            f"{checks['failed']} failed."
        )
    radius = report["radius"]
    lines.append(
        f"A class is a bond change with the atoms within {radius} bond{'' if radius == 1 else 's'}"
        " of it; brackets give their formula."
    )
    lines.append("NVT barriers are potential energies; NPT barriers are enthalpies.")
    return "\n".join(lines)


__all__ = ["campaign_status", "format_status"]
