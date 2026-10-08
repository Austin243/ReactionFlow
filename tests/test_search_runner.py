from __future__ import annotations

import fcntl

import numpy as np
import pytest
from ase import Atoms

from reactionflow.search import runner
from reactionflow.search.akmc import AKMCConfig
from reactionflow.search.eon import ProcessResult
from reactionflow.search.records import SearchStore
from reactionflow.search.runner import run_exploration

ENERGY = {0: 0.0, 1: -0.2, 2: 0.3}


def at(atoms: Atoms, x: float) -> Atoms:
    moved = atoms.copy()
    moved.positions[0, 0] = x
    return moved


class TwoStates:
    """A at x=0 and B at x=1 share one saddle at x=0.5 (barriers 0.5 and 0.7 eV)."""

    def __init__(self, script=None):
        self.calls = []
        self.script = script or {}

    def relax(self, atoms):
        return at(atoms, 0), ENERGY[0]

    def search(self, atoms, *, seed):
        x = round(atoms.positions[0, 0])
        self.calls.append((x, seed))
        saddle_x, product_x, saddle_energy = self.script.get(len(self.calls) - 1, (0.5, 1 - x, 0.5))
        return ProcessResult(
            "good",
            "",
            saddle=at(atoms, saddle_x),
            product=at(atoms, product_x),
            reactant_energy=ENERGY[x],
            saddle_energy=saddle_energy,
            product_energy=ENERGY[product_x],
            prefactor=1e13,
            reverse_prefactor=2e13,
            metadata={"seed": seed},
        )

    def same(self, first, first_energy, second, second_energy):
        return abs(first_energy - second_energy) < 1e-9 and np.allclose(
            first.positions, second.positions
        )


def run(root, backend, **kwargs):
    return run_exploration(
        root,
        Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4]),
        backend=backend,
        contract={"model": "fixture"},
        config=AKMCConfig(**{"steps": 3, "confidence": 0.5, **kwargs}),
    )


def records(root):
    store = SearchStore(root)
    return {
        (kind, name): store.read(kind, name).data
        for kind in ("states", "processes", "attempts", "steps")
        for name in store.ids(kind)
    }


def test_states_are_searched_to_confidence_then_stepped_by_rate(tmp_path):
    backend = TwoStates()
    state = run(tmp_path, backend)
    # A: one new process and two repeats; B: two repeats of the registered reverse process.
    assert [x for x, _ in backend.calls] == [0, 0, 0, 1, 1]
    assert state["steps"] == 3 and state["current"] == "state-000001"
    assert state["stop_reason"] == "steps_completed"
    store = SearchStore(tmp_path)
    steps = [store.read("steps", name).data for name in store.ids("steps")]
    assert [(step["from"], step["to"]) for step in steps] == [
        ("state-000000", "state-000001"),
        ("state-000001", "state-000000"),
        ("state-000000", "state-000001"),
    ]
    assert steps[0]["total_rate_s"] == pytest.approx(1e13 * np.exp(-0.5 / AKMCConfig().kT))
    assert 0 < steps[0]["time_s"] < steps[1]["time_s"] < steps[2]["time_s"] == state["time_s"]
    reverse = store.read("processes", "process-000000r").data
    assert reverse["barrier_eV"] == pytest.approx(0.7) and reverse["prefactor_s"] == 2e13
    assert reverse["bonds"] == {"formed": [], "broken": []}
    before = records(tmp_path)
    assert run(tmp_path, backend) == state
    assert len(backend.calls) == 5 and records(tmp_path) == before


@pytest.mark.parametrize("boundary", ["states", "processes", "attempts", "steps", "progress"])
def test_resume_after_each_publication_matches_an_uninterrupted_run(
    tmp_path, monkeypatch, boundary
):
    reference = TwoStates()
    expected = run(tmp_path / "reference", reference)
    backend = TwoStates()
    write, save = SearchStore.write, runner._save_state
    fired = False

    def interrupt_write(self, kind, *args, **kwargs):
        nonlocal fired
        result = write(self, kind, *args, **kwargs)
        if kind == boundary and not fired:
            fired = True
            raise OSError("interrupted publication")
        return result

    def interrupt_save(root, state):
        nonlocal fired
        if boundary == "progress" and state["steps"] == 1 and not fired:
            fired = True
            raise OSError("interrupted publication")
        return save(root, state)

    with monkeypatch.context() as patch:
        patch.setattr(SearchStore, "write", interrupt_write)
        patch.setattr(runner, "_save_state", interrupt_save)
        with pytest.raises(OSError, match="interrupted publication"):
            run(tmp_path / "run", backend)
    assert run(tmp_path / "run", backend) == expected
    assert backend.calls == reference.calls
    assert records(tmp_path / "run") == records(tmp_path / "reference")


def test_interrupted_search_retries_its_seed(tmp_path):
    backend = TwoStates()
    original = backend.search

    def interrupt(atoms, **kwargs):
        original(atoms, **kwargs)
        raise KeyboardInterrupt()

    backend.search = interrupt
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, backend)
    backend.search = original
    assert run(tmp_path, backend)["steps"] == 3
    assert backend.calls[0] == backend.calls[1]


def test_same_state_and_out_of_window_results_do_not_build_confidence(tmp_path):
    # Search 1 returns to A itself; search 2 reaches C over a 5 eV saddle, outside the window.
    backend = TwoStates(script={1: (0.2, 0, 0.1), 2: (-0.5, 2, 5.0)})
    state = run(tmp_path, backend, steps=1)
    store = SearchStore(tmp_path)
    statuses = [store.read("attempts", name).data["status"] for name in store.ids("attempts")]
    assert statuses == ["new", "same_state", "outside_window", "repeat", "repeat"]
    assert store.ids("states") == ("state-000000", "state-000001")
    assert state["current"] == "state-000001"


def test_a_search_rejected_after_its_saddle_keeps_the_saddle_minima_and_bond_changes(tmp_path):
    class Rejected(TwoStates):
        """The first search reaches a saddle whose minima are CH and C and H apart."""

        def search(self, atoms, *, seed):
            if self.calls:
                return super().search(atoms, seed=seed)
            self.calls.append((0, seed))
            saddle, apart = atoms.copy(), atoms.copy()
            saddle.positions[1, 0], apart.positions[1, 0] = 1.8, 3.0
            minima = ((atoms.copy(), 0.01), (apart, 0.4))
            message = "Saddle is not connected to initial state"
            return ProcessResult("failed", message, saddle=saddle, saddle_energy=0.8, minima=minima)

    start = Atoms("CH", positions=[[0, 0, 0], [1.09, 0, 0]], cell=[4, 4, 4])
    config = AKMCConfig(steps=1, confidence=0.5)
    state = run_exploration(
        tmp_path, start, backend=Rejected(), contract={"model": "fixture"}, config=config
    )
    assert state["steps"] == 1 and state["attempts"] == 4
    rejected = SearchStore(tmp_path).read("attempts", "attempt-000000")
    assert rejected.data["status"] == "failed" and rejected.data["barrier_eV"] == 0.8
    first, second = rejected.data["minima"]
    assert first == {"energy_eV": 0.01, "bonds": {"formed": [], "broken": []}}
    assert second["energy_eV"] == 0.4
    assert second["bonds"] == {"formed": [], "broken": [{"atom_ids": [0, 1], "elements": "C-H"}]}
    assert set(rejected.structures) == {"saddle", "minimum_1", "minimum_2"}
    assert rejected.structures["minimum_2"].positions[1, 0] == pytest.approx(3.0)


def test_changed_settings_and_second_writers_are_refused(tmp_path):
    run(tmp_path, TwoStates(), steps=1)
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, TwoStates(), steps=1, temperature_K=500.0)
    with (tmp_path / ".search.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another process"):
            run(tmp_path, TwoStates(), steps=1)
