"""Read-only campaign status and reaction summary."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

from ase.formula import Formula

from .campaign import CampaignConfig
from .candidates import ReactionCandidate, ReactionClasses, _isomorphic, _reaction_graph
from .detection import _symbols_by_id
from .pathway import CONVERGED_STATUSES
from .run import frame_ranges, missing_frames
from .store import _read_bundle

# Converged bands, including those refined before the saddle check, which have no verdict.
_CONVERGED = CONVERGED_STATUSES | {"ci_neb_converged"}


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
    symbols = _symbols_by_id(candidate.reactant)
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

    Each class is (candidate, events, results, occurrence ID of the candidate, and for each result
    the event it refined, for its direction). Each step of an event refined in steps is one more, an
    event of its own with the step's bonds. The candidate is None when no bundle of the class can be
    read, as in a copy of a run's results alone, so that its counts and barriers still show.

    The row is filled as reading progresses, so a damaged file later on still leaves the phase and
    step from state.json in place. Files that cannot be read are named in ``row["unreadable"]``,
    and each costs only what depends on it.
    """

    state = _read_json(root / "state.json")
    row.update(phase=state["phase"], global_step=state["global_step"])
    failure = state.get("failure")
    if state["phase"] == "failed" and failure:
        row["error"] = f"{failure.get('type')}: {failure.get('message')}"
    records = _occurrences(root)
    unreadable: Counter[str] = Counter()
    lost = missing_frames(root, state["global_frame"])
    if lost:
        row["missing_frames"] = lost
    if state.get("pending_pathway_ids"):
        row["queued"] = len(state["pending_pathway_ids"])
    changes = root / "restart-environment.jsonl"
    if changes.is_file():
        try:
            lines = [json.loads(line) for line in changes.read_text().splitlines()]
            fields = sorted({field for line in lines for field in line["changed"]})
            row["environment_changes"] = {"resumes": len(lines), "fields": fields}
        except Exception:
            unreadable["restart-environment.jsonl"] += 1
    members: dict[str, list[sqlite3.Row]] = {}
    for record in records:
        members.setdefault(record["class_id"], []).append(record)
    classes = []
    steps = []
    pathways: Counter[str] = Counter()

    read: dict[str, ReactionCandidate | None] = {}

    def bundle(occurrence: sqlite3.Row) -> ReactionCandidate | None:
        name = occurrence["bundle"]
        if name not in read:
            try:
                read[name] = _read_bundle(root / name)[1]
            except Exception:
                read[name] = None
                unreadable["candidate bundle"] += 1
        return read[name]

    def as_refined(
        occurrence: sqlite3.Row, candidate: ReactionCandidate | None
    ) -> ReactionCandidate | None:
        """The class's candidate with this occurrence's bonds, read without its structures, since
        each atom's element is the same throughout a trajectory."""

        if candidate is None:
            return None
        try:
            data = _read_json(root / occurrence["bundle"] / "candidate.json")["candidate"]
            return replace(
                candidate,
                atom_ids=tuple(data["atom_ids"]),
                reactant_bonds=frozenset(map(tuple, data["reactant_bonds"])),
                product_bonds=frozenset(map(tuple, data["product_bonds"])),
            )
        except Exception:
            unreadable["candidate bundle"] += 1
            return None

    def result_file(path: Path) -> dict[str, Any] | None:
        try:
            result = _read_json(path)
            pathways[result["status"]] += 1
            return result
        except Exception:
            unreadable["result file"] += 1
            return None

    for occurrences in members.values():
        # Any occurrence can stand for its class.
        candidate, example = None, occurrences[0]["occurrence_id"]
        for occurrence in occurrences:
            candidate = bundle(occurrence)
            if candidate is not None:
                example = occurrence["occurrence_id"]
                break
        results, refined = [], []
        for occurrence in occurrences:
            path = root / "pathways" / occurrence["occurrence_id"] / "result.json"
            result = result_file(path) if path.is_file() else None
            if result is None:
                continue
            results.append(result)
            validated = result["status"] == "saddle_validated"
            refined.append(as_refined(occurrence, candidate) if validated else None)
            if result.get("steps"):
                event = bundle(occurrence)
                for step in result["steps"]:
                    step_path = path.parent / f"step-{step['step']}" / "result.json"
                    step_result = result_file(step_path)
                    if step_result is None:
                        continue
                    part = event and replace(
                        event,
                        **{
                            key: frozenset(map(tuple, step_result[key]))
                            for key in ("reactant_bonds", "product_bonds")
                        },
                    )
                    steps.append((part, 1, [step_result], occurrence["occurrence_id"], [part]))
        classes.append((candidate, len(occurrences), results, example, refined))
    row.update(events=len(records), pathways=dict(pathways))
    notes = [
        what if what.endswith("jsonl") else f"{count} {what}{'' if count == 1 else 's'}"
        for what, count in unreadable.items()
    ]
    if not (root / "reactions.sqlite3").is_file():
        bundles = root.glob("candidates/[!.]*/candidate.json")
        if count := sum(1 for _ in bundles):
            notes.append(f"reactions.sqlite3 is missing, with {count} on disk")
    if notes:
        row["unreadable"] = notes
    return classes + steps


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
    labelled: dict[str, Any] = {}  # the graph of the direction each class is named for
    no_band: Counter[tuple[str, str]] = Counter()
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
        notes = [row["error"]] if row["error"] else []
        if "missing_frames" in row:
            lost = frame_ranges(row["missing_frames"])
            notes.insert(0, f"saved frames {lost} missing or unreadable")
        if "unreadable" in row:
            notes.insert(0, "unreadable: " + ", ".join(row["unreadable"]))
        row["error"] = "; ".join(notes) or None
        merged = known.setdefault((model, spec.pressure_GPa), ReactionClasses(radius))
        for candidate, events, results, occurrence_id, refined in classes:
            # A class with no readable candidate has no graph to merge by, so it keeps its own row.
            name = None if candidate is None else merged.find(candidate)
            if name is None:
                name = str(len(entries))
                if candidate is not None:
                    merged.add(name, candidate)
                    labelled[name] = _reaction_graph(candidate, radius=radius)
                entries[name] = {
                    "reaction": (
                        f"class of {occurrence_id} (no readable candidate, direction unknown)"
                        if candidate is None
                        else _describe(candidate, radius)
                    ),
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
            for result, event in zip(results, refined, strict=True):
                entry["pathways"][result["status"]] += 1
                if result["status"] not in _CONVERGED | {"stepwise", "saddle_validated"}:
                    no_band[result["status"], result.get("message") or "no message"] += 1
                validation = result.get("frequency_validation")
                if validation:
                    entry["frequency"][validation["status"]] += 1
                connectivity = result.get("connectivity")
                if connectivity:
                    entry["connectivity"][connectivity["status"]] += 1
                if result["status"] == "saddle_validated":
                    key = "barriers_eV"
                    # A barrier is measured from each event's own reactant, so a class that runs
                    # both ways has two barriers. A row with no candidate names no direction.
                    if candidate is not None:
                        if event is None:
                            key = "unknown_direction_barriers_eV"
                        elif not _isomorphic(labelled[name], _reaction_graph(event, radius=radius)):
                            key = "reverse_barriers_eV"
                    entry.setdefault(key, []).append(result["barrier_eV"])
    for entry in entries.values():
        for key in ("pathways", "frequency", "connectivity"):
            entry[key] = dict(entry[key])
    reactions = sorted(
        entries.values(),
        key=lambda item: (
            item["model"],
            item["pressure_GPa"] is not None,
            item["pressure_GPa"] or 0.0,
            -item["events"],
            item["reaction"],
        ),
    )
    report = {
        "campaign": str(campaign.source),
        "radius": radius,
        "trajectories": trajectories,
        "reactions": reactions,
    }
    if no_band:
        report["no_band"] = [
            {"status": status, "message": message, "events": count}
            for (status, message), count in no_band.most_common()
        ]
    return report


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(item) for item in column) for column in zip(header, *rows, strict=True)]
    return "\n".join(
        "  ".join(item.ljust(width) for item, width in zip(line, widths, strict=True)).rstrip()
        for line in (header, *rows)
    )


def _pressure(value: float | None) -> str:
    return "NVT" if value is None else f"{value:g}"


def _spread(barriers: list[float]) -> str:
    if not barriers:
        return "-"
    values = (min(barriers), statistics.median(barriers), max(barriers))
    return " / ".join(f"{value:.3f}" for value in values)


def _converged(pathways: dict[str, int]) -> str:
    # An event refined in steps has no band of its own; its steps are counted.
    total = sum(count for status, count in pathways.items() if status != "stepwise")
    converged = sum(count for status, count in pathways.items() if status in _CONVERGED)
    return f"{converged}/{total}" if total else "-"


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
    queued = [row for row in report["trajectories"] if row.get("queued")]
    if queued:
        where = ", ".join(f"{row['queued']} in {row['trajectory']}" for row in queued[:5])
        more = f", and {len(queued) - 5} more trajectories" if len(queued) > 5 else ""
        lines += [f"Waiting to be refined, not yet in the converged column: {where}{more}.", ""]
    for row in report["trajectories"]:
        if "environment_changes" in row:
            changes = row["environment_changes"]
            lines += [
                f"{row['trajectory']} resumed {changes['resumes']} time(s) under another "
                f"environment, so it is no longer bitwise exact: {', '.join(changes['fields'])}. "
                "A changed calculator source can mean a changed potential.",
                "",
            ]
    incomplete = sum("unreadable: " in (row["error"] or "") for row in report["trajectories"])
    if incomplete:
        lines += [
            f"Incomplete: files of {incomplete} "
            f"{'trajectory' if incomplete == 1 else 'trajectories'} cannot be read, so what "
            "follows leaves them out; the error column and --json say which.",
            "",
        ]
    if not report["reactions"]:
        lines.append("No reactions could be read." if incomplete else "No reactions recorded yet.")
        return "\n".join(lines)
    both_ways = any("reverse_barriers_eV" in entry for entry in report["reactions"])
    reaction_rows = []
    for entry in report["reactions"]:
        reaction_rows.append(
            [
                entry["reaction"],
                entry["model"],
                _pressure(entry["pressure_GPa"]),
                str(len(entry["trajectories"])),
                str(entry["events"]),
                _converged(entry["pathways"]),
                str(entry["pathways"].get("saddle_validated", 0)),
                _spread(entry["barriers_eV"]),
                *([_spread(entry.get("reverse_barriers_eV", []))] if both_ways else []),
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
                "validated",
                "barrier eV (min / median / max)",
                *(["reverse barrier eV (min / median / max)"] if both_ways else []),
            ],
            reaction_rows,
        )
    )
    lines.append("")
    if both_ways:
        lines.append(
            "A class holds both directions of a reaction. Barrier is for the direction the row "
            "names, and reverse barrier for the other."
        )
    unknown = sum(len(e.get("unknown_direction_barriers_eV", [])) for e in report["reactions"])
    if unknown:
        lines.append(
            f"{unknown} barrier{'' if unknown == 1 else 's'} of unknown direction, whose candidate "
            "files cannot be read, left out of the table and listed in --json."
        )
    verdicts = sum((Counter(entry["pathways"]) for entry in report["reactions"]), Counter())
    if any(verdicts[status] for status in _CONVERGED):
        line = (
            f"Saddle check: {verdicts['saddle_validated']} validated, "
            f"{verdicts['no_saddle']} no saddle, {verdicts['saddle_rejected']} rejected, "
            f"{verdicts['saddle_unchecked']} unchecked"
        )
        earlier = verdicts["ci_neb_converged"]
        if earlier:
            line += f"; {earlier} converged before the check, without a verdict"
        lines.append(line + ". Barriers come only from validated saddles.")
    if verdicts["stepwise"]:
        lines.append(
            f"{verdicts['stepwise']} event{'' if verdicts['stepwise'] == 1 else 's'} refined in"
            " steps: each step counts as an event of its own class."
        )
    if verdicts["time_limit"]:
        lines.append(
            f"{verdicts['time_limit']} refinement{'' if verdicts['time_limit'] == 1 else 's'}"
            " stopped at max_hours_per_barrier, with no barrier."
        )
    ended = [item for item in report.get("no_band", []) if item["status"] != "time_limit"]
    if ended:
        lines.append(
            f"Ended with no converged band: {sum(item['events'] for item in ended)} events."
        )
        lines += [f"  {i['events']} {i['status']}: {i['message'][:120]}" for i in ended[:5]]
        if len(ended) > 5:
            lines.append(f"  and {len(ended) - 5} other reasons, listed in --json as no_band.")
    radius = report["radius"]
    lines.append(
        f"A class is a bond change with the atoms within {radius} bond{'' if radius == 1 else 's'}"
        " of it; brackets give their formula."
    )
    lines.append("NVT barriers are potential energies; NPT barriers are enthalpies.")
    return "\n".join(lines)


__all__ = ["campaign_status", "format_status"]
