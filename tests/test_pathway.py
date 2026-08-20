from __future__ import annotations

from contextlib import contextmanager
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from reactionflow import (
    BondDetectorConfig,
    PathwayConfig,
    ReactionCandidate,
    atom_ids,
    refine_pathway,
)
from reactionflow.pathway import FIRE, NEB


class DoubleWell(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        index = atom_ids(self.atoms).index(1)
        x, y, z = self.atoms.positions[index]
        energy = (x * x - 1) ** 2 + y * y + z * z
        forces = np.zeros((len(self.atoms), 3))
        forces[index] = (-4 * x * (x * x - 1), -2 * y, -2 * z)
        self.results = {"energy": float(energy), "forces": forces}


class ConstantForce(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        forces = np.zeros((len(self.atoms), 3))
        forces[0, 0] = 1
        self.results = {"energy": 0.0, "forces": forces}


def pathway_candidate(*, resolved: bool = True, collapsed: bool = False) -> ReactionCandidate:
    reactant = Atoms(
        "He3",
        positions=[[-0.2, 5, 0], [-0.9, 0, 0], [5, 5, 5]],
        cell=[10, 10, 10],
        pbc=True,
    )
    reactant.set_array("atom_id", np.asarray([0, 1, 2]))
    scale = 1.01
    product = Atoms(
        "He3",
        positions=np.asarray(
            [
                [6, 6, 6],
                [-0.2, 5, 0],
                [-0.9 if collapsed else 1.1, 0, 0],
            ]
        )
        * scale,
        cell=[10 * scale] * 3,
        pbc=True,
        info={"atom_ids": [2, 0, 1]},
    )
    return ReactionCandidate(
        reactant=reactant,
        product=product,
        atom_ids=(0, 1),
        reactant_bonds=frozenset({(0, 1)}),
        product_bonds=frozenset(),
        reactant_frame=0,
        product_frame=1,
        observed_frame=2,
        resolved=resolved,
    )


def test_refinement_aligns_ids_freezes_spectators_and_finds_double_well_barrier(
    monkeypatch,
) -> None:
    stages: list[str] = []
    interpolation: dict[str, object] = {}
    climbing_stages: list[bool] = []
    live = 0
    max_live = 0

    original_interpolate = NEB.interpolate
    original_run = FIRE.run

    def record_interpolation(self, *args, **kwargs):
        interpolation.update(kwargs)
        return original_interpolate(self, *args, **kwargs)

    monkeypatch.setattr(NEB, "interpolate", record_interpolation)

    def record_run(self, *args, **kwargs):
        if isinstance(self.atoms, NEB):
            climbing_stages.append(bool(self.atoms.climb))
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(FIRE, "run", record_run)

    @contextmanager
    def provider(stage: str):
        nonlocal live, max_live
        live += 1
        max_live = max(max_live, live)
        stages.append(stage)
        try:
            yield DoubleWell()
        finally:
            live -= 1

    outcome = refine_pathway(
        pathway_candidate(),
        calculator_provider=provider,
        config=PathwayConfig(
            active_radius=0.2,
            relax_fmax=0.02,
            relax_steps=100,
            images=5,
            neb_fmax=0.03,
            neb_steps=250,
            ci_neb_steps=250,
        ),
        detector_config=BondDetectorConfig(pair_thresholds={"He-He": (5.07, 5.13)}),
    )

    assert outcome.converged, outcome.message
    assert outcome.barrier == pytest.approx(1.0, abs=0.02)
    assert len(outcome.energies) == len(outcome.images) == 5
    assert stages == ["relax_reactant", "relax_product", "neb"]
    assert interpolation == {"method": "idpp", "mic": True}
    assert climbing_stages == [False, True]
    assert live == 0 and max_live == 1
    assert all(atom_ids(image) == (0, 1, 2) and image.calc is None for image in outcome.images)
    assert all(np.array_equal(image.cell, outcome.images[0].cell) for image in outcome.images)
    assert outcome.images[0].positions[1, 0] == pytest.approx(-1, abs=0.02)
    assert outcome.images[-1].positions[1, 0] == pytest.approx(1, abs=0.02)
    np.testing.assert_allclose(
        [image.positions[2] for image in outcome.images],
        np.repeat([outcome.images[0].positions[2]], 5, axis=0),
    )


def test_refinement_returns_small_bounded_failure_outcomes() -> None:
    @contextmanager
    def unused_provider(_stage: str):
        raise AssertionError("calculator should not be acquired")
        yield DoubleWell()

    thresholds = BondDetectorConfig(pair_thresholds={"He-He": (5.07, 5.13)})
    unresolved = refine_pathway(
        pathway_candidate(resolved=False),
        calculator_provider=unused_provider,
        detector_config=thresholds,
    )
    large_cell_change = pathway_candidate()
    large_cell_change.product.set_cell([12, 12, 12], scale_atoms=True)
    cell_unresolved = refine_pathway(
        large_cell_change,
        calculator_provider=unused_provider,
        detector_config=thresholds,
    )

    @contextmanager
    def double_well(_stage: str):
        yield DoubleWell()

    collapsed = refine_pathway(
        pathway_candidate(collapsed=True),
        calculator_provider=double_well,
        config=PathwayConfig(active_radius=0.2, relax_fmax=0.02, relax_steps=100),
        detector_config=thresholds,
    )

    @contextmanager
    def constant_force(_stage: str):
        yield ConstantForce()

    relaxation_failed = refine_pathway(
        pathway_candidate(),
        calculator_provider=constant_force,
        config=PathwayConfig(active_radius=0.2, relax_fmax=1e-12, relax_steps=1),
        detector_config=thresholds,
    )

    @contextmanager
    def broken_provider(_stage: str):
        raise RuntimeError("calculator unavailable")
        yield DoubleWell()

    failed = refine_pathway(
        pathway_candidate(),
        calculator_provider=broken_provider,
        detector_config=thresholds,
    )

    assert unresolved.status == "unresolved" and unresolved.images == ()
    assert cell_unresolved.status == "unresolved" and "cell change" in cell_unresolved.message
    assert collapsed.status == "collapsed" and len(collapsed.images) == 2
    assert relaxation_failed.status == "relaxation_failed"
    assert failed.status == "failed" and "calculator unavailable" in failed.message
