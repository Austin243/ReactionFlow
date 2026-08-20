from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ase.md.verlet import VelocityVerlet

from reactionflow import (
    BondDetectorConfig,
    ComponentState,
    ExactRestartSnapshot,
    PathwayConfig,
    ReactionRun,
    ReactionRunConfig,
    assign_atom_ids,
)
from reactionflow.segments import ResumeToken


class PairDoubleWell(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        vector = self.atoms.positions[1] - self.atoms.positions[0]
        distance = float(np.linalg.norm(vector))
        offset = distance - 1.1
        inner = offset * offset - 0.25
        derivative = 4 * offset * inner
        direction = vector / distance
        forces = np.zeros((2, 3))
        forces[0] = derivative * direction
        forces[1] = -derivative * direction
        self.results = {"energy": inner * inner, "forces": forces}


class LeaseCounter:
    def __init__(self) -> None:
        self.live = 0
        self.max_live = 0
        self.stages: list[str] = []

    @contextmanager
    def __call__(self, stage: str):
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        self.stages.append(stage)
        try:
            yield PairDoubleWell()
        finally:
            self.live -= 1


def run_config(
    *,
    observation_interval: int = 9,
    persistence_frames: int = 1,
) -> ReactionRunConfig:
    return ReactionRunConfig(
        observation_interval=observation_interval,
        detector=BondDetectorConfig(
            persistence_frames=persistence_frames,
            pair_thresholds={"H-H": (0.8, 1.2)},
        ),
        candidate_stability_frames=1,
        pathway=PathwayConfig(
            active_radius=2,
            relax_fmax=0.005,
            relax_steps=100,
            images=3,
            neb_fmax=0.005,
            neb_steps=200,
            ci_neb_steps=200,
        ),
    )


def pair(distance: float) -> Atoms:
    return Atoms("H2", positions=[[0, 0, 0], [distance, 0, 0]])


def test_run_ase_detects_checkpoints_refines_and_resumes(tmp_path, caplog) -> None:
    atoms = pair(0.6)
    atoms.set_momenta([[-0.5, 0, 0], [0.5, 0, 0]])
    leases = LeaseCounter()
    run = ReactionRun.create(tmp_path, config=run_config())
    caplog.set_level(logging.WARNING, logger="reactionflow.run")

    summary = run.run_ase(
        atoms,
        md_calculator_provider=leases,
        pathway_calculator_provider=leases,
        dynamics_factory=lambda frame: VelocityVerlet(frame, timestep=0.1, logfile=None),
        total_steps=18,
    )

    assert (summary.phase, summary.generation, summary.global_step) == ("completed", 1, 18)
    assert (summary.occurrences, summary.pathways) == (1, 1)
    (record,) = run.occurrences.records()
    assert record.is_representative
    result_dir = tmp_path / "pathways" / record.occurrence_id
    assert {path.name for path in result_dir.iterdir()} == {"result.json", "images.traj"}
    result = json.loads((result_dir / "result.json").read_text())
    assert result["status"] == "ci_neb_converged"
    assert result["barrier_eV"] == pytest.approx(0.0625, abs=2e-4)
    assert len(read(result_dir / "images.traj", ":")) == 3
    assert (tmp_path / "segments/0000/checkpoint/resume.json").is_file()
    assert (tmp_path / "segments/0000/trajectory.traj").is_file()
    assert (tmp_path / "segments/0001/trajectory.traj").is_file()
    assert leases.stages == ["md", "relax_reactant", "relax_product", "neb", "md"]
    assert leases.live == 0 and leases.max_live == 1
    assert any(
        "continuing generation 1 from a structural checkpoint" in message
        for message in caplog.messages
    )


def test_manual_reopen_suppresses_reverse_duplicate_and_serializes_leases(tmp_path, caplog) -> None:
    leases = LeaseCounter()
    run = ReactionRun.create(tmp_path, config=run_config(observation_interval=1))
    caplog.set_level(logging.WARNING, logger="reactionflow.run")
    first = run.start(pair(0.65))
    broken = first.atoms.copy()
    broken.positions[1, 0] = 1.55

    with leases("md") as calculator:
        broken.calc = calculator
        (representative,) = run.observe(broken, global_step=1, global_frame=1)
        broken.calc = None
    token = run.checkpoint(broken)
    assert token.path.is_file() and run.phase == "refining"

    reopened = ReactionRun.open(tmp_path)
    assert reopened.phase == "refining"
    (outcome,) = reopened.refine_pending(leases)
    assert outcome.converged
    (tmp_path / "segments/0001").mkdir()  # interrupted generation handoff
    second = reopened.resume_segment()
    caplog.clear()
    reopened = ReactionRun.open(tmp_path)  # running state, before resumed MD starts
    assert any(
        "continuing generation 1 from a structural checkpoint" in message
        for message in caplog.messages
    )
    assert reopened.current_segment is not None
    second = reopened.current_segment
    reverse = second.atoms.copy()
    reverse.positions[1, 0] = 0.65
    (duplicate,) = reopened.observe(reverse, global_step=2, global_frame=2)
    summary = reopened.complete()

    records = reopened.occurrences.records()
    assert [record.class_id for record in records] == [representative.class_id] * 2
    assert [record.is_representative for record in records] == [True, False]
    assert duplicate.occurrence_id != representative.occurrence_id
    assert reopened.pending_pathway_ids == ()
    assert summary.pathways == 1
    assert len([path for path in (tmp_path / "pathways").iterdir() if path.is_dir()]) == 1
    assert leases.stages == ["md", "relax_reactant", "relax_product", "neb"]
    assert leases.live == 0 and leases.max_live == 1
    assert not list(tmp_path.rglob("*.tmp"))


class ScriptedExactRuntime:
    def __init__(
        self,
        atoms: Atoms,
        *,
        nsteps: int = 0,
        rng_state=None,
        before_run=None,
    ) -> None:
        self._atoms = atoms
        self._nsteps = nsteps
        self._rng = np.random.default_rng(1234)
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state
        self._before_run = before_run

    @property
    def atoms(self) -> Atoms:
        return self._atoms

    @property
    def nsteps(self) -> int:
        return self._nsteps

    def run(self, steps: int) -> None:
        if self._before_run is not None:
            self._before_run()
        for _ in range(steps):
            self._rng.random()
            self._nsteps += 1
            if self._nsteps >= 1:
                self._atoms.positions[1, 0] = 1.55

    def snapshot(self) -> ExactRestartSnapshot:
        return ExactRestartSnapshot(
            atoms=self._atoms,
            dynamics=ComponentState(
                kind="test.scripted-dynamics",
                metadata={
                    "nsteps": self._nsteps,
                    "rng_state": self._rng.bit_generator.state,
                },
            ),
            calculator=ComponentState(kind="test.stateless-calculator"),
        )


class ScriptedRuntimeProvider:
    def __init__(self, leases: LeaseCounter, *, interrupt_after_calls: int | None = None):
        self.leases = leases
        self.interrupt_after_calls = interrupt_after_calls
        self.calls = 0

    def _before_run(self) -> None:
        self.calls += 1
        if self.interrupt_after_calls is not None and self.calls > self.interrupt_after_calls:
            raise KeyboardInterrupt("simulated scheduler interruption")

    @contextmanager
    def start(self, atoms: Atoms):
        with self.leases("md"):
            yield ScriptedExactRuntime(atoms, before_run=self._before_run)

    @contextmanager
    def restore(self, snapshot: ExactRestartSnapshot):
        metadata = snapshot.dynamics.metadata
        with self.leases("md"):
            yield ScriptedExactRuntime(
                snapshot.atoms,
                nsteps=int(metadata["nsteps"]),
                rng_state=metadata["rng_state"],
                before_run=self._before_run,
            )


def test_run_exact_restores_pending_monitor_then_refines_and_continues(tmp_path) -> None:
    initial = pair(0.6)
    leases = LeaseCounter()
    interrupted = ReactionRun.create(
        tmp_path,
        config=run_config(observation_interval=1, persistence_frames=2),
    )

    with pytest.raises(KeyboardInterrupt, match="scheduler interruption"):
        interrupted.run_exact(
            initial,
            runtime_provider=ScriptedRuntimeProvider(leases, interrupt_after_calls=1),
            pathway_calculator_provider=leases,
            total_steps=4,
        )

    interrupted_state = json.loads((tmp_path / "state.json").read_text())
    assert (interrupted_state["phase"], interrupted_state["global_step"]) == ("running", 1)
    assert interrupted_state["detector_state"]["pending"]
    assert interrupted_state["active_checkpoint"] is not None

    reopened = ReactionRun.open(tmp_path)
    summary = reopened.run_exact(
        runtime_provider=ScriptedRuntimeProvider(leases),
        pathway_calculator_provider=leases,
        total_steps=4,
    )

    assert (summary.phase, summary.generation, summary.global_step) == ("completed", 1, 4)
    assert (summary.occurrences, summary.pathways) == (1, 1)
    final_state = json.loads((tmp_path / "state.json").read_text())
    final_snapshot = ExactRestartSnapshot.read(
        tmp_path / final_state["active_checkpoint"] / "exact-restart"
    )
    control_atoms = assign_atom_ids(initial.copy())
    control = ScriptedExactRuntime(control_atoms)
    control.run(4)
    control_snapshot = control.snapshot()

    np.testing.assert_array_equal(final_snapshot.atoms.positions, control_snapshot.atoms.positions)
    assert final_snapshot.dynamics.metadata == control_snapshot.dynamics.metadata
    assert final_snapshot.calculator == control_snapshot.calculator
    token = ResumeToken.read(tmp_path / "segments/0000/checkpoint/resume.json")
    assert token.fidelity == "exact"
    result = json.loads(next((tmp_path / "pathways").glob("*/result.json")).read_text())
    assert result["status"] == "ci_neb_converged"
    assert leases.live == 0 and leases.max_live == 1
    assert leases.stages == [
        "md",
        "md",
        "relax_reactant",
        "relax_product",
        "neb",
        "md",
    ]
