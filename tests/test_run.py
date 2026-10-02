from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

from reactionflow import (
    BondDetectorConfig,
    ComponentState,
    ExactRestartSnapshot,
    PathwayConfig,
    PathwayOutcome,
    ReactionRun,
    ReactionRunConfig,
    assign_atom_ids,
)
from reactionflow.pathway import Vibrations
from reactionflow.run import refine_pathway
from reactionflow.status import _trajectory


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


class ScriptedExactRuntime:
    def __init__(
        self,
        atoms: Atoms,
        *,
        nsteps: int = 0,
        rng_state=None,
        before_run=None,
        script=None,
    ) -> None:
        self._atoms = atoms
        self._nsteps = nsteps
        self._rng = np.random.default_rng(1234)
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state
        self._before_run = before_run
        # Step at which the second atom jumps to each x position; the bond breaks at step 1.
        self._script = {1: 1.55} if script is None else script

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
            if self._nsteps in self._script:
                self._atoms.positions[1, 0] = self._script[self._nsteps]

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
    def __init__(
        self,
        leases: LeaseCounter,
        *,
        interrupt_after_calls: int | None = None,
        script=None,
    ):
        self.leases = leases
        self.interrupt_after_calls = interrupt_after_calls
        self.script = script
        self.calls = 0

    def _before_run(self) -> None:
        self.calls += 1
        if self.interrupt_after_calls is not None and self.calls > self.interrupt_after_calls:
            raise KeyboardInterrupt("simulated scheduler interruption")

    @contextmanager
    def start(self, atoms: Atoms):
        with self.leases("md"):
            yield ScriptedExactRuntime(atoms, before_run=self._before_run, script=self.script)

    @contextmanager
    def restore(self, snapshot: ExactRestartSnapshot):
        metadata = snapshot.dynamics.metadata
        with self.leases("md"):
            yield ScriptedExactRuntime(
                snapshot.atoms,
                nsteps=int(metadata["nsteps"]),
                rng_state=metadata["rng_state"],
                before_run=self._before_run,
                script=self.script,
            )


def interrupt_refinement(*_args, **_kwargs):
    raise KeyboardInterrupt("interrupted before refinement")


def test_run_detects_refines_and_resumes(tmp_path) -> None:
    leases = LeaseCounter()
    run = ReactionRun.create(tmp_path, config=run_config())

    summary = run.run(
        pair(0.6),
        runtime_provider=ScriptedRuntimeProvider(leases),
        pathway_calculator_provider=leases,
        total_steps=18,
    )

    assert (summary.phase, summary.generation, summary.global_step) == ("completed", 1, 18)
    assert (summary.occurrences, summary.pathways) == (1, 1)
    (record,) = run.occurrences.records()
    assert record.is_representative
    result_dir = tmp_path / "pathways" / record.occurrence_id
    assert {path.name for path in result_dir.iterdir()} == {"result.json", "images.traj"}
    result = json.loads((result_dir / "result.json").read_text())
    assert result["schema_version"] == 1
    assert result["status"] == "ci_neb_converged"
    assert result["barrier_eV"] == pytest.approx(0.0625, abs=2e-4)
    frequency = result["frequency_validation"]
    assert frequency["status"] == "one_imaginary_mode"
    assert frequency["scope"] == "active_atoms_fixed_environment"
    assert frequency["transition_state_index"] == 1
    assert frequency["active_atom_ids"] == [0, 1]
    assert frequency["imaginary_mode_indices"] == [0]
    assert frequency["frequencies_cm1"][0] < -frequency["imaginary_cutoff_cm1"]
    assert frequency["primary_mode_index"] == 0
    assert np.linalg.norm(frequency["primary_mode"]) == pytest.approx(1)
    assert len(read(result_dir / "images.traj", ":")) == 3
    assert (tmp_path / "segments/0000/trajectory.traj").is_file()
    assert (tmp_path / "segments/0001/trajectory.traj").is_file()
    assert leases.stages == ["md", "relax_reactant", "relax_product", "neb", "md"]
    assert leases.live == 0 and leases.max_live == 1


def test_frequency_failure_is_recorded_and_md_continues(tmp_path, monkeypatch) -> None:
    def fail_frequencies(_vibrations) -> None:
        raise RuntimeError("synthetic frequency failure")

    monkeypatch.setattr(Vibrations, "run", fail_frequencies)
    leases = LeaseCounter()
    run = ReactionRun.create(tmp_path, config=run_config())

    summary = run.run(
        pair(0.6),
        runtime_provider=ScriptedRuntimeProvider(leases),
        pathway_calculator_provider=leases,
        total_steps=18,
    )

    assert (summary.phase, summary.generation, summary.global_step) == ("completed", 1, 18)
    assert run.failure is None
    result_path = next((tmp_path / "pathways").glob("*/result.json"))
    result = json.loads(result_path.read_text())
    assert result["status"] == "ci_neb_converged"
    assert result["barrier_eV"] == pytest.approx(0.0625, abs=2e-4)
    assert result["frequency_validation"]["status"] == "failed"
    assert result["frequency_validation"]["message"] == "synthetic frequency failure"
    assert result["frequency_validation"]["frequencies_cm1"] == []
    assert (tmp_path / "segments/0001/trajectory.traj").is_file()
    assert leases.live == 0 and leases.max_live == 1


def test_reopen_refines_then_suppresses_reverse_duplicate_and_serializes_leases(
    tmp_path, monkeypatch
) -> None:
    leases = LeaseCounter()
    script = {1: 1.55, 2: 0.65}  # break, then re-form the same bond
    run = ReactionRun.create(tmp_path, config=run_config(observation_interval=1))
    with monkeypatch.context() as patch:
        patch.setattr("reactionflow.run.refine_pathway", interrupt_refinement)
        with pytest.raises(KeyboardInterrupt, match="before refinement"):
            run.run(
                pair(0.65),
                runtime_provider=ScriptedRuntimeProvider(leases, script=script),
                pathway_calculator_provider=leases,
                total_steps=2,
            )

    reopened = ReactionRun.open(tmp_path)
    assert reopened.phase == "refining"
    (representative,) = reopened.occurrences.records()
    (tmp_path / "segments/0001").mkdir()  # interrupted generation handoff
    summary = reopened.run(
        runtime_provider=ScriptedRuntimeProvider(leases, script=script),
        pathway_calculator_provider=leases,
        total_steps=2,
    )

    records = reopened.occurrences.records()
    assert [record.class_id for record in records] == [representative.class_id] * 2
    assert [record.is_representative for record in records] == [True, False]
    assert records[1].occurrence_id != representative.occurrence_id
    assert reopened.pending_pathway_ids == ()
    assert (summary.phase, summary.pathways) == ("completed", 1)
    assert len([path for path in (tmp_path / "pathways").iterdir() if path.is_dir()]) == 1
    assert leases.stages == ["md", "md", "relax_reactant", "relax_product", "neb", "md"]
    assert leases.live == 0 and leases.max_live == 1
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize(
    "failure_status",
    ["unresolved", "collapsed", "relaxation_failed", "neb_failed", "ci_neb_failed", "failed"],
)
def test_fresh_occurrence_retries_failed_class_until_converged(
    tmp_path, monkeypatch, failure_status
) -> None:
    attempts = []

    def fail_first(candidate, **kwargs):
        attempts.append(candidate)
        if len(attempts) == 1:
            return PathwayOutcome(status=failure_status, images=(candidate.reactant,))
        return refine_pathway(candidate, **kwargs)

    def fail_frequencies(_vibrations):
        raise RuntimeError("diagnostics remain separate from pathway convergence")

    monkeypatch.setattr("reactionflow.run.refine_pathway", fail_first)
    monkeypatch.setattr(Vibrations, "run", fail_frequencies)
    leases = LeaseCounter()

    def advance(run, atoms=None):
        return run.run(
            atoms,
            # Break the bond, re-form it, and break it again.
            runtime_provider=ScriptedRuntimeProvider(leases, script={1: 1.55, 2: 0.65, 3: 1.55}),
            pathway_calculator_provider=leases,
            total_steps=3,
        )

    publish_checkpoint = ReactionRun._checkpoint

    def interrupt_second_boundary(run, runtime):
        if run.global_frame == 2:
            raise KeyboardInterrupt("after registration, before its checkpoint")
        publish_checkpoint(run, runtime)

    run = ReactionRun.create(tmp_path, config=run_config(observation_interval=1))
    with monkeypatch.context() as patch:
        patch.setattr(ReactionRun, "_checkpoint", interrupt_second_boundary)
        with pytest.raises(KeyboardInterrupt, match="before its checkpoint"):
            advance(run, pair(0.65))
    first, retry = run.occurrences.records()
    assert first.is_representative
    assert not retry.is_representative and retry.class_id == first.class_id
    first_result = tmp_path / "pathways" / first.occurrence_id / "result.json"
    original_result = first_result.read_bytes()
    assert json.loads(original_result)["status"] == failure_status

    publish_outcome = ReactionRun._publish_outcome

    def interrupt_after_publication(run, occurrence_id, outcome):
        publish_outcome(run, occurrence_id, outcome)
        raise KeyboardInterrupt("after immutable outcome publication")

    run = ReactionRun.open(tmp_path)  # Registration survived; its queued attempt did not.
    assert run.pending_pathway_ids == ()
    with monkeypatch.context() as patch:
        patch.setattr(ReactionRun, "_publish_outcome", interrupt_after_publication)
        with pytest.raises(KeyboardInterrupt, match="outcome publication"):
            advance(run)

    run = ReactionRun.open(tmp_path)
    # The replayed observation queued the retry exactly once.
    assert run.pending_pathway_ids == (retry.occurrence_id,)
    summary = advance(run)
    assert len(attempts) == 2  # Reopen reused the published outcome.
    assert first_result.read_bytes() == original_result
    retried = json.loads((tmp_path / "pathways" / retry.occurrence_id / "result.json").read_text())
    assert retried["status"] == "ci_neb_converged"
    assert retried["frequency_validation"]["status"] == "failed"
    # The third occurrence belongs to a converged class and launches nothing.
    assert (summary.phase, summary.occurrences, summary.pathways) == ("completed", 3, 2)
    assert run.pending_pathway_ids == ()
    row = {}
    (reaction,) = _trajectory(tmp_path, row)
    assert row["pathways"] == {failure_status: 1, "ci_neb_converged": 1}
    assert reaction[1] == 3
    assert [result["status"] for result in reaction[2]] == [failure_status, "ci_neb_converged"]


@pytest.mark.parametrize("pressure", [None, 0.0])
def test_run_restores_pending_monitor_then_refines_and_continues(tmp_path, pressure) -> None:
    initial = pair(0.6)
    if pressure is not None:
        initial.set_cell([10, 10, 10])
        initial.pbc = True
    leases = LeaseCounter()
    interrupted = ReactionRun.create(
        tmp_path,
        config=run_config(observation_interval=1, persistence_frames=2),
    )

    with pytest.raises(KeyboardInterrupt, match="scheduler interruption"):
        interrupted.run(
            initial,
            runtime_provider=ScriptedRuntimeProvider(leases, interrupt_after_calls=1),
            pathway_calculator_provider=leases,
            total_steps=4,
            pressure_GPa=pressure,
        )

    interrupted_state = json.loads((tmp_path / "state.json").read_text())
    assert (interrupted_state["phase"], interrupted_state["global_step"]) == ("running", 1)
    assert interrupted_state["detector_state"]["pending"]
    assert interrupted_state["active_checkpoint"] is not None

    reopened = ReactionRun.open(tmp_path)
    summary = reopened.run(
        runtime_provider=ScriptedRuntimeProvider(leases),
        pathway_calculator_provider=leases,
        total_steps=4,
        pressure_GPa=pressure,
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
    result = json.loads(next((tmp_path / "pathways").glob("*/result.json")).read_text())
    if pressure is None:
        assert result["status"] == "ci_neb_converged"
        assert result["method"] == "neb" and result["barrier_quantity"] == "potential_energy"
    else:
        # This calculator has no stress. Record the SSNEB failure and resume
        # the exact MD state instead of silently falling back to fixed-cell NEB.
        assert result["status"] == "failed" and "stress" in result["message"]
        assert result["method"] == "ssneb" and result["pressure_GPa"] == pressure
    assert leases.live == 0 and leases.max_live == 1
    expected_stages = [
        "md",
        "md",
        "relax_reactant",
        "relax_product",
        "neb",
        "md",
    ]
    if pressure is not None:
        expected_stages = ["md", "md", "relax_reactant", "md"]
    assert leases.stages == expected_stages


def independent_run_config() -> ReactionRunConfig:
    return ReactionRunConfig(
        observation_interval=1,
        detector=BondDetectorConfig(
            persistence_frames=2,
            pair_thresholds={"H-H": (0.8, 1.2), "C-O": (0.8, 1.2)},
        ),
        candidate_stability_frames=2,
    )


def independent_pairs() -> Atoms:
    return Atoms("H2CO", positions=[[0, 0, 0], [0.6, 0, 0], [10, 0, 0], [10.6, 0, 0]])


def skipped_refinement(candidate, **_kwargs):
    return PathwayOutcome(status="failed", images=(candidate.reactant, candidate.product))


def interrupted_two_pair_run(tmp_path, monkeypatch, *, total_steps):
    """Stop in refinement of the first pair while the second pair's change is still pending."""

    original_run = ScriptedExactRuntime.run

    def run_both_pairs(runtime, steps):
        original_run(runtime, steps)
        if runtime.nsteps >= 3:
            runtime.atoms.positions[3, 0] = 11.55

    monkeypatch.setattr(ScriptedExactRuntime, "run", run_both_pairs)
    leases = LeaseCounter()
    initial = independent_pairs()
    run = ReactionRun.create(tmp_path, config=independent_run_config())
    with monkeypatch.context() as patch:
        patch.setattr("reactionflow.run.refine_pathway", interrupt_refinement)
        with pytest.raises(KeyboardInterrupt, match="before refinement"):
            run.run(
                initial,
                runtime_provider=ScriptedRuntimeProvider(leases),
                pathway_calculator_provider=leases,
                total_steps=total_steps,
            )
    assert run.phase == "refining" and run.global_frame == 3
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["detector_state"]["pending"]
    monkeypatch.setattr("reactionflow.run.refine_pathway", skipped_refinement)
    reopened = ReactionRun.open(tmp_path)
    summary = reopened.run(
        runtime_provider=ScriptedRuntimeProvider(leases),
        pathway_calculator_provider=leases,
        total_steps=total_steps,
    )
    return initial, reopened, summary


def test_reopen_after_local_confirmation_preserves_other_pending_region(tmp_path, monkeypatch):
    initial, reopened, summary = interrupted_two_pair_run(tmp_path, monkeypatch, total_steps=5)

    assert (summary.phase, summary.occurrences, summary.pathways) == ("completed", 2, 2)
    candidates = [
        reopened.occurrences.load(record.occurrence_id) for record in reopened.occurrences.records()
    ]
    assert [
        (item.atom_ids, item.reactant_frame, item.product_frame, item.observed_frame)
        for item in candidates
    ] == [((0, 1), 0, 1, 3), ((2, 3), 2, 3, 5)]
    assert all(item.resolved for item in candidates)
    final_state = json.loads((tmp_path / "state.json").read_text())
    final_snapshot = ExactRestartSnapshot.read(
        tmp_path / final_state["active_checkpoint"] / "exact-restart"
    )
    control = ScriptedExactRuntime(assign_atom_ids(initial.copy()))
    control.run(5)
    assert final_snapshot.dynamics.metadata == control.snapshot().dynamics.metadata
    np.testing.assert_array_equal(final_snapshot.atoms.positions, control.atoms.positions)


def test_completion_drains_other_pending_region_as_unresolved(tmp_path, monkeypatch):
    _, reopened, summary = interrupted_two_pair_run(tmp_path, monkeypatch, total_steps=3)

    assert (summary.phase, summary.occurrences, summary.pathways) == ("completed", 2, 1)
    (unresolved,) = [
        reopened.occurrences.load(record.occurrence_id)
        for record in reopened.occurrences.records()
        if not record.is_representative
    ]
    assert unresolved.atom_ids == (2, 3) and not unresolved.resolved
    assert (unresolved.reactant_frame, unresolved.product_frame) == (2, 3)
    leases = LeaseCounter()
    again = reopened.run(
        runtime_provider=ScriptedRuntimeProvider(leases),
        pathway_calculator_provider=leases,
        total_steps=3,
    )
    assert again == summary and leases.stages == []


def test_missing_tracker_checkpoint_does_not_reset_pending_history(tmp_path, monkeypatch):
    monkeypatch.setattr("reactionflow.run.refine_pathway", interrupt_refinement)
    leases = LeaseCounter()
    run = ReactionRun.create(tmp_path, config=run_config(observation_interval=1))
    with pytest.raises(KeyboardInterrupt, match="before refinement"):
        run.run(
            pair(0.6),
            runtime_provider=ScriptedRuntimeProvider(leases),
            pathway_calculator_provider=leases,
            total_steps=2,
        )
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["phase"] == "refining"
    shutil.rmtree(tmp_path / state["active_checkpoint"] / "tracker")
    with pytest.raises(FileNotFoundError):
        ReactionRun.open(tmp_path)
