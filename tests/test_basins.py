"""Chemical basins: states joined without a bond change are one state for the kinetics."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from ase import Atoms

from reactionflow.detection import BondDetectorConfig
from reactionflow.search import runner
from reactionflow.search.akmc import AKMCConfig, basin_escape, rate
from reactionflow.search.eon import ProcessResult
from reactionflow.search.labels import describe
from reactionflow.search.records import SearchStore
from reactionflow.search.runner import run_exploration

# H2 bonded at 0.70 A in two places (A at y=0, B at y=1) and dissociated (C).
ENERGY = {"A": 0.0, "B": 0.0, "C": -0.5}
INTERNAL, CHEMICAL = 0.05, 0.6
PREFACTOR = 1e13
H2 = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]], cell=[10, 10, 10])
# The same over a Cu atom that stays put. By the default distances both H are bonded to it at A,
# neither at B, and one at C. NO_CU_H leaves those contacts out of the bonds.
ON_CU = H2 + Atoms("Cu", positions=[[0, -1.6, 0]])
NO_CU_H = BondDetectorConfig(pair_thresholds={"Cu-H": (0.1, 0.2)})


def label(atoms: Atoms) -> str:
    if atoms.get_distance(0, 1) > 1.5:
        return "C"
    return "B" if atoms.positions[0, 1] > 0.5 else "A"


def place(atoms: Atoms, name: str) -> Atoms:
    moved = atoms.copy()
    y = 1.0 if name == "B" else 0.0
    separation = 2.5 if name == "C" else 0.70
    moved.positions[:2] = [[0.0, y, 0.0], [separation, y, 0.0]]
    return moved


class Conformers:
    """A <-> B is a 0.05 eV translation; B -> C breaks the H-H bond over 0.6 eV.

    Searches from B alternate between C and A. `chemistry=False` never finds C, so the A/B
    basin has no chemical exit.
    """

    def __init__(self, chemistry: bool = True):
        self.chemistry = chemistry
        self.calls: list[str] = []

    def relax(self, atoms):
        return place(atoms, "A"), ENERGY["A"]

    def search(self, atoms, *, seed):
        origin = label(atoms)
        made_from_b = sum(1 for name in self.calls if name == "B")
        self.calls.append(origin)
        if origin == "A":
            product, saddle_energy = "B", INTERNAL
        elif origin == "B":
            to_c = self.chemistry and made_from_b % 2 == 0
            product, saddle_energy = ("C", CHEMICAL) if to_c else ("A", INTERNAL)
        else:
            product, saddle_energy = "B", CHEMICAL
        # One saddle between each pair of states, whichever side the search starts from.
        saddle = place(atoms, product)
        saddle.positions = (saddle.positions + place(atoms, origin).positions) / 2 + [0, 0, 0.3]
        return ProcessResult(
            "good",
            "",
            saddle=saddle,
            product=place(atoms, product),
            reactant_energy=ENERGY[origin],
            saddle_energy=saddle_energy,
            product_energy=ENERGY[product],
            prefactor=PREFACTOR,
            reverse_prefactor=PREFACTOR,
            metadata={"seed": seed},
        )

    def same(self, first, first_energy, second, second_energy):
        return abs(first_energy - second_energy) < 1e-9 and np.allclose(
            first.positions, second.positions
        )


def run(root, backend, atoms=H2, detector=None, **kwargs):
    return run_exploration(
        root,
        atoms,
        backend=backend,
        contract={"model": "fixture"},
        config=AKMCConfig(**{"steps": 1, "confidence": 0.5, "temperature_K": 300.0, **kwargs}),
        detector=detector,
    )


def steps(root):
    store = SearchStore(root)
    return [store.read("steps", name).data for name in store.ids("steps")]


def records(root):
    store = SearchStore(root)
    return {
        (kind, name): store.read(kind, name).data
        for kind in ("states", "processes", "attempts", "steps")
        for name in store.ids(kind)
    }


def test_a_step_crosses_the_conformer_basin_to_its_chemical_exit(tmp_path):
    state = run(tmp_path, Conformers(), basins="bonds")
    kT = AKMCConfig(temperature_K=300.0).kT
    chemical = rate(PREFACTOR, CHEMICAL, kT)
    taken = steps(tmp_path)
    # Moving from A to the unexplored conformer B is not chemistry; the escape from the
    # explored A/B basin to C is, and with equal internal rates it takes 2/k on average.
    assert [(s["chemical"], s["from"], s["to"]) for s in taken] == [
        (False, "state-000000", "state-000001"),
        (True, "state-000001", "state-000002"),
    ]
    assert taken[1]["mean_escape_time_s"] == pytest.approx(2 / chemical, rel=1e-12)
    assert taken[1]["escape_probability"] == pytest.approx(1.0)
    assert taken[1]["basin_states"] == 2
    assert state["chemical_steps"] == 1 and state["steps"] == 2
    assert state["current"] == "state-000002"
    assert state["time_s"] == pytest.approx(taken[0]["dt_s"] + taken[1]["dt_s"])


def test_the_window_counts_only_chemical_exits_when_basins_are_on(tmp_path):
    # 0.6 eV lies more than 20 kT (0.52 eV at 300 K) above the 0.05 eV conformer exit.
    # Two plain steps reach B and search it.
    run(tmp_path / "none", Conformers(), steps=2)
    store = SearchStore(tmp_path / "none")
    statuses = {store.read("attempts", name).data["status"] for name in store.ids("attempts")}
    assert "outside_window" in statuses
    assert all(not s.get("chemical") for s in steps(tmp_path / "none"))
    run(tmp_path / "bonds", Conformers(), basins="bonds")
    store = SearchStore(tmp_path / "bonds")
    statuses = {store.read("attempts", name).data["status"] for name in store.ids("attempts")}
    assert "outside_window" not in statuses


def test_a_basin_without_a_chemical_exit_stops(tmp_path):
    state = run(tmp_path, Conformers(chemistry=False), basins="bonds")
    assert state["stop_reason"] == "no_chemical_exit"
    assert state["chemical_steps"] == 0
    assert [s["chemical"] for s in steps(tmp_path)] == [False]


def test_default_runs_keep_their_records(tmp_path):
    state = run(tmp_path, Conformers())
    assert "chemical_steps" not in state
    assert all("chemical" not in s for s in steps(tmp_path))


def test_detector_settings_decide_whether_a_metal_contact_is_a_bond(tmp_path):
    def taken(name, **settings):
        run(tmp_path / name, Conformers(), basins="bonds", atoms=ON_CU, **settings)
        moved = SearchStore(tmp_path / name).read("processes", "process-000000").data["bonds"]
        return describe(moved), [(s["chemical"], s["basin_states"]) for s in steps(tmp_path / name)]

    # By the default distances the move from A to B is chemistry, and the one step the run takes.
    assert taken("default") == ("broken Cu-H x2", [(True, 1)])
    # With the contacts left out it is a change of conformer, as without the Cu atom, and the
    # step that counts breaks the H-H bond.
    assert taken("set", detector=NO_CU_H) == ("no bond change", [(False, 1), (True, 2)])


def test_a_search_eon_rejects_as_not_connected_leaves_the_basin_by_its_bonds(tmp_path):
    class Settled(Conformers):
        """The first search from A breaks the H-H bond, but its start side settles into a
        conformer 0.1 eV below A, so EON finds neither minimum at A."""

        def search(self, atoms, *, seed):
            if self.calls:
                return super().search(atoms, seed=seed)
            self.calls.append("A")
            saddle = place(atoms, "C")
            saddle.positions[1, 0] = 1.2
            return ProcessResult(
                "failed",
                "Saddle is not connected to initial state",
                saddle=saddle,
                saddle_energy=CHEMICAL,
                metadata={"seed": seed, "termination_reason": 6},
                minima=((place(atoms, "B"), -0.1), (place(atoms, "C"), ENERGY["C"])),
            )

        def prefactors(self, first, saddle, second):
            assert (label(first), label(second)) == ("B", "C")
            return 2e12, 3e12

    run(tmp_path / "none", Settled())
    rejected = SearchStore(tmp_path / "none").read("attempts", "attempt-000000")
    assert rejected.data["status"] == "failed"
    run(tmp_path / "bonds", Settled(), basins="bonds")
    store = SearchStore(tmp_path / "bonds")
    attempt = store.read("attempts", "attempt-000000").data
    process = store.read("processes", attempt["process"])
    assert attempt["status"] == "new" and attempt["start_side_energy_eV"] == -0.1
    # Measured from the start-side conformer, which lies below A.
    assert process.data["barrier_eV"] == pytest.approx(CHEMICAL + 0.1)
    assert (process.data["prefactor_s"], process.data["reverse_prefactor_s"]) == (2e12, 3e12)
    assert process.data["bonds"] == {
        "formed": [],
        "broken": [{"atom_ids": [0, 1], "elements": "H-H"}],
    }
    assert label(process.structures["start_side"]) == "B"
    assert label(store.read("states", process.data["product"]).structures["atoms"]) == "C"

    # Over the Cu atom both minima change a Cu-H contact by the default distances, so neither
    # keeps the start's bonds and the search stays rejected.
    run(tmp_path / "cu", Settled(), basins="bonds", atoms=ON_CU)
    rejected = SearchStore(tmp_path / "cu").read("attempts", "attempt-000000").data
    assert rejected["status"] == "failed"
    sides = [describe(side["bonds"]) for side in rejected["minima"]]
    assert sides == ["broken Cu-H x2", "broken Cu-H, H-H"]
    # With the contacts left out, every record is that of the molecule alone.
    for basins in ("none", "bonds"):
        run(tmp_path / f"cu-{basins}", Settled(), basins=basins, atoms=ON_CU, detector=NO_CU_H)
        assert records(tmp_path / f"cu-{basins}") == records(tmp_path / basins)


def test_escape_stays_exact_across_twenty_orders_of_magnitude():
    fast, slow = 1e13, 1e-7
    rates = {"A": [("B", fast)], "B": [("A", fast), ("exit", slow)]}
    probabilities, mean = basin_escape(rates, "B")
    assert probabilities == {"exit": pytest.approx(1.0)}
    assert mean == pytest.approx(2 / slow, rel=1e-12)


def test_escape_matches_the_absorbing_chain_on_a_mixed_network():
    rng = np.random.default_rng(3)
    states, exits = ["s0", "s1", "s2"], ["x0", "x1"]
    rates = {s: [] for s in states}
    for s in states:
        for target in states + exits:
            if target != s and rng.random() < 0.8:
                rates[s].append((target, float(10 ** rng.uniform(-1, 2))))
    rates["s0"].append(("x0", 0.5))
    index = {s: i for i, s in enumerate(states)}
    total = np.array([sum(r for _, r in rates[s]) for s in states])
    generator = np.diag(total)
    to_exit = np.zeros((3, 2))
    for s in states:
        for target, r in rates[s]:
            if target in index:
                generator[index[s], index[target]] -= r
            else:
                to_exit[index[s], exits.index(target)] += r
    expected_time = np.linalg.solve(generator, np.ones(3))
    expected_exit = np.linalg.solve(generator, to_exit)
    for s in states:
        probabilities, mean = basin_escape(rates, s)
        assert mean == pytest.approx(expected_time[index[s]], rel=1e-10)
        for j, x in enumerate(exits):
            assert probabilities.get(x, 0.0) == pytest.approx(expected_exit[index[s], j], rel=1e-10)


@pytest.mark.parametrize("boundary", ["states", "processes", "attempts", "steps", "progress"])
def test_basin_runs_resume_after_each_publication(tmp_path, monkeypatch, boundary):
    reference = Conformers()
    expected = run(tmp_path / "reference", reference, basins="bonds")
    backend = Conformers()
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
            run(tmp_path / "run", backend, basins="bonds")
    assert run(tmp_path / "run", backend, basins="bonds") == expected
    assert records(tmp_path / "run") == records(tmp_path / "reference")


def test_basin_mode_is_part_of_the_contract(tmp_path):
    run(tmp_path, Conformers(), basins="bonds")
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, Conformers())


def test_detector_settings_are_part_of_the_contract(tmp_path):
    state = run(tmp_path / "set", Conformers(), atoms=ON_CU, detector=NO_CU_H)
    assert run(tmp_path / "set", Conformers(), atoms=ON_CU, detector=NO_CU_H) == state
    wider = BondDetectorConfig(pair_thresholds={"Cu-H": (0.1, 0.3)})
    for changed in (None, wider):
        with pytest.raises(ValueError, match="changed"):
            run(tmp_path / "set", Conformers(), atoms=ON_CU, detector=changed)
    run(tmp_path / "default", Conformers(), atoms=ON_CU)
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path / "default", Conformers(), atoms=ON_CU, detector=NO_CU_H)
    # At the defaults the contract is the one written before the settings existed.
    contract = json.loads((tmp_path / "default" / "search-contract.json").read_text())
    assert set(contract) == {"initial_sha256", "model_and_backend", "akmc", "search_code"}
    run(tmp_path / "default", Conformers(), atoms=ON_CU, detector=BondDetectorConfig())


def test_basins_setting_is_validated():
    with pytest.raises(ValueError, match=r"akmc\.basins"):
        AKMCConfig(basins="graph")
    assert AKMCConfig().basins == "none"
    assert math.isfinite(AKMCConfig(basins="bonds").kT)
