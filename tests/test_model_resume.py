from __future__ import annotations

import json
from contextlib import nullcontext
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from reactionflow import (
    BondDetectorConfig,
    ExactRestartSnapshot,
    PathwayOutcome,
    ReactionRun,
    ReactionRunConfig,
)
from reactionflow.adapters import _torch, uma
from reactionflow.adapters.ase import _ASERuntime
from reactionflow.campaign import TrajectorySpec


class Harmonic(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(self.scale * np.sum(self.atoms.positions**2) / 2),
            "forces": -self.scale * self.atoms.positions,
            "stress": np.zeros(6),
        }


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(uma, "require_package", lambda *args: None)
    monkeypatch.setattr(_torch, "_load_torch", lambda device: None)
    monkeypatch.setattr(_torch, "_preserve_rng", lambda torch: nullcontext())
    monkeypatch.setattr(_torch, "_torch_environment", lambda torch, device: {"device": device})
    monkeypatch.setattr(_torch, "version", lambda name: "test-version")
    monkeypatch.setattr(
        uma.UMAAdapter,
        "_new_calculator",
        lambda self: Harmonic(float(self.files["checkpoint"].read_text())),
    )
    path = tmp_path / "weights.pt"
    path.write_text("1.0")
    trajectory = TrajectorySpec(
        id="uma", total_steps=2, timestep_fs=0.1, temperature_K=300, pressure_GPa=None, seed=19
    )

    def create():
        return uma.create_adapter(
            trajectory=trajectory,
            options={"checkpoint": str(path), "task": "omol", "device": "cpu"},
        )

    return path, create


def pending_run(root, adapter, monkeypatch):
    """Stop a run in refinement at an exact checkpoint taken from a real ASE runtime."""

    run = ReactionRun.create(
        root,
        config=ReactionRunConfig(
            observation_interval=1,
            detector=BondDetectorConfig(persistence_frames=1, pair_thresholds={"H-H": (0.8, 1.2)}),
            candidate_stability_frames=1,
        ),
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.65, 0, 0]])
    atoms.info.update(charge=0, spin=1)
    advance = _ASERuntime.run

    def break_bond(runtime, steps):
        advance(runtime, steps)
        # Supply a bond-breaking observation while retaining an actual ASE restart state.
        runtime.atoms.positions[1] = runtime.atoms.positions[0] + [1.55, 0, 0]

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("interrupted before refinement")

    with monkeypatch.context() as patch:
        patch.setattr(_ASERuntime, "run", break_bond)
        patch.setattr("reactionflow.run.refine_pathway", interrupt)
        with pytest.raises(KeyboardInterrupt, match="before refinement"):
            run.run(
                atoms,
                runtime_provider=adapter,
                pathway_calculator_provider=adapter.calculator,
                total_steps=1,
            )
    assert run.phase == "refining"
    state = json.loads((root / "state.json").read_text())
    checkpoint = root / state["active_checkpoint"] / "exact-restart"
    return ExactRestartSnapshot.read(checkpoint), checkpoint


@pytest.mark.parametrize("total_steps", [1, 2])
def test_reopened_refinement_checks_model_before_publishing_and_failure_is_retryable(
    tmp_path, monkeypatch, model, total_steps
):
    weights, create = model
    root = tmp_path / "run"
    snapshot, checkpoint = pending_run(root, create(), monkeypatch)
    before_state = (root / "state.json").read_bytes()
    before_checkpoint = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    refinements = []
    advances = []
    original_run = _ASERuntime.run

    def advance(runtime, steps):
        advances.append((runtime.nsteps, steps))
        original_run(runtime, steps)

    def refine(candidate, *, calculator_provider, **kwargs):
        with calculator_provider("neb") as calculator:
            refinements.append(calculator.scale)
            calculator.get_potential_energy(candidate.product)
        return PathwayOutcome(
            status="ci_neb_converged", images=(candidate.reactant, candidate.product)
        )

    monkeypatch.setattr(_ASERuntime, "run", advance)
    monkeypatch.setattr("reactionflow.run.refine_pathway", refine)
    weights.write_text("2.0")
    reopened = ReactionRun.open(root)
    changed = create()
    with pytest.raises(ValueError, match="environment differs"):
        reopened.run(
            runtime_provider=changed,
            pathway_calculator_provider=changed.calculator,
            total_steps=total_steps,
        )

    assert reopened.phase == "refining" and reopened.failure is None
    assert refinements == advances == []
    assert not list((root / "pathways").glob("*/result.json"))
    assert (root / "state.json").read_bytes() == before_state
    assert {path.name: path.read_bytes() for path in checkpoint.iterdir()} == before_checkpoint

    weights.write_text("1.0")
    restored = create()
    result = ReactionRun.open(root).run(
        runtime_provider=restored,
        pathway_calculator_provider=restored.calculator,
        total_steps=total_steps,
    )
    assert (result.phase, result.global_step, result.pathways) == ("completed", total_steps, 1)
    assert refinements == [1.0]
    assert advances == ([] if total_steps == 1 else [(1, 1)])
    assert restored._contract == snapshot.calculator
