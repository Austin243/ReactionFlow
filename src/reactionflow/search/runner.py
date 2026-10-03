"""Adaptive kinetic Monte Carlo over EON process searches, committed after each search and step.

States, processes, searches, and steps are immutable records. `search-state.json` holds the
progress that changes: the current state, the clock, and each state's search counters.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io.jsonio import encode

from .._durable import ensure_directory, file_digest, publish, sync_directory
from ..detection import canonical_copy
from .akmc import AKMCConfig, confidence, in_window, kmc_step, rate
from .eon import validate_start
from .labels import bond_changes, reversed_changes
from .records import SearchStore


def _save_state(root: Path, state: dict[str, Any]) -> None:
    temporary = root / ".search-state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n")
    publish(temporary, root / "search-state.json")


def read_search_state(root: str | Path) -> dict[str, Any]:
    """Read progress without loading a calculator or changing the run."""

    state = json.loads((Path(root) / "search-state.json").read_text())
    if state.get("schema_version") != 1:
        raise ValueError("unsupported exploration state version")
    return state


def _bind_contract(root: Path, expected: dict[str, Any]) -> None:
    """Bind the model and settings before even the initial minimum can be published."""

    path = root / "search-contract.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("exploration structure, settings, model, or code changed")
        sync_directory(root)
        return
    if (root / "search-state.json").exists() or any(
        (root / kind).is_dir() and any((root / kind).iterdir())
        for kind in ("states", "processes", "attempts", "steps")
    ):
        raise ValueError("existing exploration records have no bound model contract")
    temporary = root / ".search-contract.json.tmp"
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n")
    publish(temporary, path)


def _state_id(store: SearchStore, backend: Any, atoms: Atoms, energy: float) -> str:
    """Return the known state these atoms are in, or record a new one."""

    states = store.ids("states")
    for identifier in states:
        known = store.read("states", identifier)
        if backend.same(known.structures["atoms"], known.data["energy_eV"], atoms, energy):
            return identifier
    identifier = f"state-{len(states):06d}"
    store.write("states", identifier, {"energy_eV": energy}, structures={"atoms": atoms})
    return identifier


def _processes(store: SearchStore, identifiers: list[str]) -> list[dict[str, Any]]:
    return [{**store.read("processes", pid).data, "id": pid} for pid in identifiers]


def _in_window(processes: list[dict[str, Any]], config: AKMCConfig) -> list[dict[str, Any]]:
    """Processes inside the thermal window above the lowest barrier."""

    if not processes:
        return []
    lowest = min(process["barrier_eV"] for process in processes)
    return [p for p in processes if in_window(p["barrier_eV"], lowest, config)]


def _repeat(store: SearchStore, backend: Any, known: list[dict], saddle: Atoms, energy: float):
    for process in known:
        stored = store.read("processes", process["id"])
        if backend.same(stored.structures["saddle"], process["saddle_energy_eV"], saddle, energy):
            return process
    return None


def _reverse(
    store: SearchStore, backend: Any, state: dict[str, Any], forward_id: str, forward
) -> str | None:
    """Give the product state the same saddle in reverse, unless it already knows it."""

    reverse_id = f"{forward_id}r"
    if reverse_id in store.ids("processes"):
        return reverse_id
    data, structures = forward.data, forward.structures
    product = state["states"].get(data["product"], {"processes": []})
    known = _processes(store, product["processes"])
    if _repeat(store, backend, known, structures["saddle"], data["saddle_energy_eV"]):
        return None
    reactant = store.read("states", data["state"])
    store.write(
        "processes",
        reverse_id,
        {
            "state": data["product"],
            "product": data["state"],
            "barrier_eV": data["saddle_energy_eV"] - data["product_energy_eV"],
            "saddle_energy_eV": data["saddle_energy_eV"],
            "product_energy_eV": reactant.data["energy_eV"],
            "prefactor_s": data["reverse_prefactor_s"],
            "reverse_prefactor_s": data["prefactor_s"],
            "bonds": reversed_changes(data["bonds"]),
            "source": f"reverse of {forward_id}",
        },
        structures={"saddle": structures["saddle"], "product": reactant.structures["atoms"]},
    )
    return reverse_id


def _attempt(
    store: SearchStore, state: dict[str, Any], active: dict[str, Any], backend: Any, config
) -> dict[str, Any]:
    origin_id = active["state"]
    forward_id = active["id"].replace("attempt-", "process-", 1)
    if forward_id in store.ids("processes"):
        # The search finished and its process was published before an interruption.
        forward = store.read("processes", forward_id)
        reverse = _reverse(store, backend, state, forward_id, forward)
        search = forward.data["search"]
        return {**search, "status": "new", "process": forward_id, "reverse": reverse}
    origin = store.read("states", origin_id)
    energy = origin.data["energy_eV"]
    data: dict[str, Any] = dict(active)
    result = backend.search(origin.structures["atoms"].copy(), seed=active["seed"])
    data.update(backend=result.metadata, message=result.message)
    if result.status != "good":
        return {**data, "status": "failed"}
    barrier = result.saddle_energy - energy
    data["barrier_eV"] = barrier
    known = _processes(store, state["states"][origin_id]["processes"])
    lowest = min([barrier] + [process["barrier_eV"] for process in known])
    if not in_window(barrier, lowest, config):
        return {**data, "status": "outside_window"}
    repeated = _repeat(store, backend, known, result.saddle, result.saddle_energy)
    if repeated is not None:
        relevant = repeated["id"] in {process["id"] for process in _in_window(known, config)}
        return {**data, "status": "repeat", "process": repeated["id"], "relevant": relevant}
    data["product"] = _state_id(store, backend, result.product, result.product_energy)
    if data["product"] == origin_id:
        # Equivalent atoms exchanged places: no state change, so no part of the kinetics.
        return {**data, "status": "same_state"}
    data["status"] = "new"
    store.write(
        "processes",
        forward_id,
        {
            "state": origin_id,
            "product": data["product"],
            "barrier_eV": barrier,
            "saddle_energy_eV": result.saddle_energy,
            "product_energy_eV": result.product_energy,
            "prefactor_s": result.prefactor,
            "reverse_prefactor_s": result.reverse_prefactor,
            "bonds": bond_changes(origin.structures["atoms"], result.product),
            "source": active["id"],
            "search": data,
        },
        structures={"saddle": result.saddle, "product": result.product},
    )
    reverse = _reverse(store, backend, state, forward_id, store.read("processes", forward_id))
    return {**data, "process": forward_id, "reverse": reverse}


def _new_counters() -> dict[str, Any]:
    return {"searches": 0, "repeats": 0, "processes": []}


def _commit_attempt(state: dict[str, Any], result: dict[str, Any]) -> None:
    counters = state["states"][result["state"]]
    counters["searches"] += 1
    if result["status"] == "new":
        counters["processes"].append(result["process"])
        counters["repeats"] = 0
        product = state["states"].setdefault(result["product"], _new_counters())
        if result["reverse"]:
            product["processes"].append(result["reverse"])
    elif result["status"] == "repeat" and result["relevant"]:
        counters["repeats"] += 1
    state["attempts"] += 1
    state["active"] = None


def _step(store: SearchStore, state: dict[str, Any], config: AKMCConfig) -> dict[str, Any]:
    identifier = f"step-{state['steps']:06d}"
    if identifier in store.ids("steps"):
        return store.read("steps", identifier).data
    current = state["current"]
    counters = state["states"][current]
    window = _in_window(_processes(store, counters["processes"]), config)
    rates = [rate(p["prefactor_s"], p["barrier_eV"], config.kT) for p in window]
    pick, dt = kmc_step(rates, [config.seed, 1, state["steps"]])
    data = {
        "from": current,
        "to": window[pick]["product"],
        "process": window[pick]["id"],
        "dt_s": dt,
        "time_s": state["time_s"] + dt,
        "total_rate_s": float(sum(rates)),
        "processes_in_window": len(window),
        "confidence": confidence(counters["repeats"]),
        "searches": counters["searches"],
    }
    store.write("steps", identifier, data)
    return data


def run_exploration(
    root: str | Path,
    atoms: Atoms,
    *,
    backend: Any,
    contract: dict[str, Any],
    config: AKMCConfig | None = None,
) -> dict[str, Any]:
    """Run or resume one AKMC trajectory. The caller owns the backend and its calculator.

    The backend relaxes, searches (`search(atoms, seed=...)` returns an EON process result), and
    compares structures. Resume reuses every published search and step exactly once; an
    interrupted search is retried with its saved seed.
    """

    config = config or AKMCConfig()
    root = Path(root).resolve()
    ensure_directory(root)
    with (root / ".search.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another process is already writing this exploration") from None
        atoms = canonical_copy(atoms)
        validate_start(atoms)
        encoded = json.dumps(json.loads(encode(atoms)), sort_keys=True, allow_nan=False)
        expected = {
            "initial_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "model_and_backend": contract,
            "akmc": asdict(config),
            "search_code": {
                name: file_digest(Path(__file__).with_name(name))
                for name in ("runner.py", "akmc.py", "labels.py", "records.py")
            },
        }
        # Normalize JSON-shaped contracts before comparing a freshly loaded checkpoint.
        expected = json.loads(encode(expected))
        _bind_contract(root, expected)
        store = SearchStore(root)
        if (root / "search-state.json").exists():
            state = read_search_state(root)
        else:
            relaxed, energy = backend.relax(atoms)
            initial = _state_id(store, backend, relaxed, energy)
            state = {
                "schema_version": 1,
                "current": initial,
                "time_s": 0.0,
                "steps": 0,
                "attempts": 0,
                "states": {initial: _new_counters()},
                "active": None,
                "stop_reason": None,
            }
            _save_state(root, state)
        while state["steps"] < config.steps:
            if state["active"] is None:
                counters = state["states"][state["current"]]
                if confidence(counters["repeats"]) >= config.confidence:
                    step = _step(store, state, config)
                    state.update(current=step["to"], time_s=step["time_s"])
                    state["steps"] += 1
                    _save_state(root, state)
                    continue
                index = state["attempts"]
                seed = int(np.random.SeedSequence([config.seed, 0, index]).generate_state(1)[0])
                state["active"] = {
                    "id": f"attempt-{index:06d}",
                    "state": state["current"],
                    "seed": seed,
                }
                _save_state(root, state)
            active = state["active"]
            if active["id"] in store.ids("attempts"):
                result = store.read("attempts", active["id"]).data
            else:
                result = _attempt(store, state, active, backend, config)
                store.write("attempts", active["id"], result)
            _commit_attempt(state, result)
            _save_state(root, state)
        state["stop_reason"] = "steps_completed"
        _save_state(root, state)
        return state
