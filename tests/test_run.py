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
    PathwayOutcome,
    ReactionRun,
    ReactionRunConfig,
    assign_atom_ids,
)
from reactionflow.pathway import Vibrations
from reactionflow.run import refine_pathway
from reactionflow.segments import ResumeToken
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


def test_open_accepts_legacy_run_state_without_frequency_controls(tmp_path) -> None:
    ReactionRun.create(tmp_path, config=run_config())
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text())
    assert state["schema_version"] == 2
    state["schema_version"] = 1
    state["config"]["pathway"].pop("frequency_delta")
    state["config"]["pathway"].pop("imaginary_frequency_cutoff_cm1")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    reopened = ReactionRun.open(tmp_path)

    assert reopened.config.pathway.frequency_delta == 0.01
    assert reopened.config.pathway.imaginary_frequency_cutoff_cm1 == 50.0


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
    result_path = result_dir / "result.json"
    result = json.loads(result_path.read_text())
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

    loaded = run._load_outcome(record.occurrence_id)
    assert loaded.frequency_validation is not None
    assert loaded.frequency_validation.frequencies_cm1 == tuple(frequency["frequencies_cm1"])
    assert loaded.frequency_validation.primary_mode == tuple(
        tuple(vector) for vector in frequency["primary_mode"]
    )

    result.pop("frequency_validation")
    for field in ("method", "pressure_GPa", "barrier_quantity", "enthalpies_eV", "volumes_A3"):
        result.pop(field)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    legacy = run._load_outcome(record.occurrence_id)
    assert legacy.frequency_validation is None
    assert legacy.method == "neb" and legacy.pressure_GPa is None
    assert legacy.barrier_quantity == "potential_energy"
    assert legacy.enthalpies == legacy.volumes == ()
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


def test_frequency_failure_is_recorded_and_md_continues(tmp_path, monkeypatch) -> None:
    def fail_frequencies(_vibrations) -> None:
        raise RuntimeError("synthetic frequency failure")

    monkeypatch.setattr(Vibrations, "run", fail_frequencies)
    atoms = pair(0.6)
    atoms.set_momenta([[-0.5, 0, 0], [0.5, 0, 0]])
    leases = LeaseCounter()
    run = ReactionRun.create(tmp_path, config=run_config())

    summary = run.run_ase(
        atoms,
        md_calculator_provider=leases,
        pathway_calculator_provider=leases,
        dynamics_factory=lambda frame: VelocityVerlet(frame, timestep=0.1, logfile=None),
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


@pytest.mark.parametrize("legacy_tracker", [False, True])
def test_manual_reopen_suppresses_reverse_duplicate_and_serializes_leases(
    tmp_path, caplog, legacy_tracker
) -> None:
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
    if legacy_tracker:
        import shutil

        shutil.rmtree(token.path.parent / "tracker")
        value = json.loads(token.path.read_text())
        value.pop("has_tracker")
        token.path.write_text(json.dumps(value))

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
    run = ReactionRun.create(tmp_path, config=run_config(observation_interval=1))
    atoms = run.start(pair(0.65)).atoms.copy()
    atoms.positions[1, 0] = 1.55
    (first,) = run.observe(atoms, global_step=1, global_frame=1)
    run.checkpoint(atoms)
    assert run.refine_pending(leases)[0].status == failure_status
    first_result = tmp_path / "pathways" / first.occurrence_id / "result.json"
    original_result = first_result.read_bytes()
    run.resume_segment()

    run = ReactionRun.open(tmp_path)
    atoms = run.current_segment.atoms.copy()
    atoms.positions[1, 0] = 0.65
    (retry,) = run.observe(atoms, global_step=2, global_frame=2)
    assert not retry.is_representative and retry.class_id == first.class_id
    assert run.pending_pathway_ids == (retry.occurrence_id,)
    candidate = run.occurrences.load(retry.occurrence_id)
    # Replaying the same registration and observing an equivalent candidate while pending
    # must not schedule the occurrence twice or queue another attempt for this class.
    run._register((candidate,), label="00000002")
    run._register((candidate,), label="equivalent-pending")
    assert run.pending_pathway_ids == (retry.occurrence_id,)
    run.checkpoint(atoms)

    def interrupt_state_write():
        raise KeyboardInterrupt("after immutable outcome publication")

    monkeypatch.setattr(run, "_write_state", interrupt_state_write)
    with pytest.raises(KeyboardInterrupt, match="outcome publication"):
        run.refine_pending(leases)
    run = ReactionRun.open(tmp_path)
    (outcome,) = run.refine_pending(leases)
    assert outcome.converged and outcome.frequency_validation.status == "failed"
    assert len(attempts) == 2  # Reopen reused the published outcome.
    assert first_result.read_bytes() == original_result
    assert run.refine_pending(leases) == ()

    atoms = run.resume_segment().atoms.copy()
    atoms.positions[1, 0] = 1.55
    run.observe(atoms, global_step=3, global_frame=3)
    assert run.pending_pathway_ids == ()
    assert run.complete().pathways == 2
    row = {}
    (reaction,) = _trajectory(tmp_path, row)
    assert row["pathways"] == {failure_status: 1, "ci_neb_converged": 1}
    assert reaction[2] == 4
    assert [result["status"] for result in reaction[3]] == [failure_status, "ci_neb_converged"]


def test_reopen_recovers_retry_registered_after_durable_boundary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "reactionflow.run.refine_pathway",
        lambda candidate, **kwargs: PathwayOutcome(
            status="neb_failed", images=(candidate.reactant,)
        ),
    )
    run = ReactionRun.create(tmp_path, config=run_config(observation_interval=1))
    atoms = run.start(pair(0.65)).atoms.copy()
    atoms.positions[1, 0] = 1.55
    (first,) = run.observe(atoms, global_step=1, global_frame=1)
    run._register((run.occurrences.load(first.occurrence_id),), label="equivalent-pending")
    run.checkpoint(atoms)
    run.refine_pending(LeaseCounter())
    run.resume_segment()
    run = ReactionRun.open(tmp_path)
    assert run.pending_pathway_ids == ()  # The old in-flight duplicate stays skipped.

    atoms = run.current_segment.atoms.copy()
    atoms.positions[1, 0] = 0.65
    (retry,) = run._observe(atoms, global_step=2, global_frame=2, persist=False)
    assert not retry.is_representative
    run = ReactionRun.open(tmp_path)  # Registration survived; pending state did not.
    assert run.pending_pathway_ids == (retry.occurrence_id,)
    run.observe(atoms, global_step=2, global_frame=2)
    assert run.pending_pathway_ids == (retry.occurrence_id,)
    run.checkpoint(atoms)
    run.refine_pending(LeaseCounter())
    run.resume_segment()
    assert ReactionRun.open(tmp_path).pending_pathway_ids == ()


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


@pytest.mark.parametrize("pressure", [None, 0.0])
def test_run_exact_restores_pending_monitor_then_refines_and_continues(tmp_path, pressure) -> None:
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
        interrupted.run_exact(
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
    summary = reopened.run_exact(
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
    token = ResumeToken.read(tmp_path / "segments/0000/checkpoint/resume.json")
    assert token.fidelity == "exact"
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
    from reactionflow import PathwayOutcome

    return PathwayOutcome(status="failed", images=(candidate.reactant, candidate.product))


def test_exact_reopen_after_local_confirmation_preserves_other_pending_region(
    tmp_path, monkeypatch
):
    original_run = ScriptedExactRuntime.run

    def run_both_pairs(runtime, steps):
        original_run(runtime, steps)
        if runtime.nsteps >= 3:
            runtime.atoms.positions[3, 0] = 11.55

    monkeypatch.setattr(ScriptedExactRuntime, "run", run_both_pairs)

    def interrupt_refinement(*_args, **_kwargs):
        raise KeyboardInterrupt("interrupted while another region is pending")

    monkeypatch.setattr("reactionflow.run.refine_pathway", interrupt_refinement)
    leases = LeaseCounter()
    initial = independent_pairs()
    run = ReactionRun.create(tmp_path, config=independent_run_config())
    with pytest.raises(KeyboardInterrupt, match="another region is pending"):
        run.run_exact(
            initial,
            runtime_provider=ScriptedRuntimeProvider(leases),
            pathway_calculator_provider=leases,
            total_steps=5,
        )
    assert run.phase == "refining" and run.global_frame == 3
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["detector_state"]["pending"]
    monkeypatch.setattr("reactionflow.run.refine_pathway", skipped_refinement)
    reopened = ReactionRun.open(tmp_path)
    summary = reopened.run_exact(
        runtime_provider=ScriptedRuntimeProvider(leases),
        pathway_calculator_provider=leases,
        total_steps=5,
    )
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


@pytest.mark.parametrize("finish_at_boundary", [False, True])
def test_structural_reopen_retains_other_region_or_drains_it_at_completion(
    tmp_path, monkeypatch, finish_at_boundary
):
    monkeypatch.setattr("reactionflow.run.refine_pathway", skipped_refinement)
    run = ReactionRun.create(tmp_path, config=independent_run_config())
    atoms = run.start(independent_pairs()).atoms
    for frame in (1, 2, 3):
        atoms.positions[1, 0] = 1.55
        if frame == 3:
            atoms.positions[3, 0] = 11.55
        run.observe(atoms, global_step=frame, global_frame=frame)
    run.checkpoint(atoms)
    reopened = ReactionRun.open(tmp_path)
    reopened.refine_pending(LeaseCounter())
    reopened = ReactionRun.open(tmp_path)
    if finish_at_boundary:
        summary = reopened.complete()
        assert summary.phase == "completed" and summary.occurrences == 2
        unresolved = [
            reopened.occurrences.load(record.occurrence_id)
            for record in reopened.occurrences.records()
            if not record.is_representative
        ]
        assert len(unresolved) == 1
        assert unresolved[0].atom_ids == (2, 3) and not unresolved[0].resolved
        assert (unresolved[0].reactant_frame, unresolved[0].product_frame) == (2, 3)
        assert reopened.complete() == summary
    else:
        reopened.resume_segment()
        # Exercise another interruption in the empty-generation handoff.
        reopened = ReactionRun.open(tmp_path)
        for frame in (4, 5):
            records = reopened.observe(atoms, global_step=frame, global_frame=frame)
            assert len(records) == (1 if frame == 5 else 0)
        second = reopened.occurrences.load(records[0].occurrence_id)
        assert second.atom_ids == (2, 3) and second.resolved
        assert (second.reactant_frame, second.product_frame, second.observed_frame) == (2, 3, 5)


def test_missing_new_tracker_checkpoint_does_not_reset_pending_history(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr("reactionflow.run.refine_pathway", skipped_refinement)
    run = ReactionRun.create(tmp_path, config=run_config(observation_interval=1))
    atoms = run.start(pair(0.6)).atoms
    atoms.positions[1, 0] = 1.55
    run.observe(atoms, global_step=1, global_frame=1)
    token = run.checkpoint(atoms)
    assert token.has_tracker
    shutil.rmtree(token.path.parent / "tracker")
    reopened = ReactionRun.open(tmp_path)
    reopened.refine_pending(LeaseCounter())
    with pytest.raises(FileNotFoundError):
        reopened.resume_segment()
