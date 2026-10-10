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
from reactionflow.detection import classify_bonds
from reactionflow.pathway import FIRE, NEB, _check_connectivity, _classify_frequencies


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


class TiltedSpectator(Calculator):
    """Atom 1 crosses a double well; spectator 2 has its own double well that atom 1 tilts."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]
    coupling = 0.3

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        ids = atom_ids(self.atoms)
        mover, spectator = ids.index(1), ids.index(2)
        x, y, z = self.atoms.positions[mover] - (5, 15, 15)
        u, v, w = self.atoms.positions[spectator] - (15, 15, 15)
        energy = (x * x - 1) ** 2 + (u * u - 1) ** 2 + self.coupling * x * u
        forces = np.zeros((len(self.atoms), 3))
        forces[mover] = (-4 * x * (x * x - 1) - self.coupling * u, -2 * y, -2 * z)
        forces[spectator] = (-4 * u * (u * u - 1) - self.coupling * x, -2 * v, -2 * w)
        energy += y * y + z * z + v * v + w * w
        self.results = {"energy": float(energy), "forces": forces, "stress": np.zeros(6)}


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
    validation = outcome.frequency_validation
    assert validation is not None
    assert validation.status == "one_imaginary_mode"
    assert validation.transition_state_index == 1 + int(np.argmax(outcome.energies[1:-1]))
    assert validation.active_atom_ids == (0, 1)
    assert validation.active_max_force_eV_A is not None
    assert validation.active_max_force_eV_A < 0.03
    assert len(validation.frequencies_cm1) == 3 * len(validation.active_atom_ids)
    assert validation.imaginary_mode_indices == (0,)
    assert validation.frequencies_cm1[0] == pytest.approx(-521.3, abs=2)
    assert validation.primary_mode_index == 0
    assert outcome.connectivity is not None
    assert outcome.connectivity.status == "connects_endpoints"
    assert set(outcome.connectivity.sides) == {"reactant", "product"}
    np.testing.assert_allclose(validation.primary_mode[0], 0, atol=1e-10)
    np.testing.assert_allclose(np.abs(validation.primary_mode[1]), [1, 0, 0], atol=1e-10)
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


@pytest.mark.parametrize("pressure", [None, 0.0])
def test_product_starts_from_the_relaxed_reactant_so_spectators_settle_alike(pressure) -> None:
    # Atom 1 leaves atom 0. Spectator 2 sits on the ridge between its two wells, as it can in a
    # hot frame: bonded to atom 3 in one well and not in the other. Atom 1 tilts that ridge
    # oppositely in the two frames, so relaxed separately the endpoints settle 2 differently.
    reactant = Atoms(
        "He4",
        positions=[[2, 15, 15], [4.1, 15, 15], [15, 15, 15], [18, 15, 15]],
        cell=[30, 30, 30],
        pbc=True,
    )
    reactant.set_array("atom_id", np.arange(4))
    product = reactant.copy()
    product.positions[1, 0] = 6.1
    candidate = ReactionCandidate(
        reactant=reactant,
        product=product,
        atom_ids=(0, 1),
        reactant_bonds=frozenset({(0, 1)}),
        product_bonds=frozenset(),
        reactant_frame=0,
        product_frame=1,
        observed_frame=2,
        resolved=True,
    )

    @contextmanager
    def provider(_stage: str):
        yield TiltedSpectator()

    outcome = refine_pathway(
        candidate,
        calculator_provider=provider,
        config=PathwayConfig(
            active_radius=0.5,
            relax_fmax=0.01,
            relax_steps=500,
            images=5,
            neb_fmax=0.03,
            neb_steps=500,
            ci_neb_steps=500,
        ),
        detector_config=BondDetectorConfig(pair_thresholds={"He-He": (2.5, 3.0)}),
        pressure_GPa=pressure,
    )

    assert outcome.converged, outcome.message
    first, last = outcome.images[0], outcome.images[-1]
    assert first.get_distance(0, 1) < 2.5 < 3 < last.get_distance(0, 1)
    assert first.get_distance(2, 3) < 2.5 and last.get_distance(2, 3) < 2.5
    assert outcome.connectivity is not None
    assert outcome.connectivity.status == "connects_endpoints"


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


@pytest.mark.parametrize(
    ("raw", "expected_signed", "expected_indices", "expected_status"),
    [
        (
            np.asarray([100.0 + 0j, 49.0j]),
            (100.0, -49.0),
            (),
            "zero_imaginary_modes",
        ),
        (
            np.asarray([50.0j, 100.0 + 0j]),
            (-50.0, 100.0),
            (0,),
            "one_imaginary_mode",
        ),
        (
            np.asarray([60.0j, -75.0j]),
            (-60.0, -75.0),
            (0, 1),
            "multiple_imaginary_modes",
        ),
    ],
)
def test_frequency_classification_uses_a_significant_imaginary_cutoff(
    raw,
    expected_signed,
    expected_indices,
    expected_status,
) -> None:
    signed, indices, status = _classify_frequencies(raw, cutoff_cm1=50.0)

    assert signed == expected_signed
    assert indices == expected_indices
    assert status == expected_status


def test_frequency_failure_does_not_discard_a_converged_path(monkeypatch) -> None:
    def fail_frequency_setup(atoms):
        if atoms.calc is not None:
            raise RuntimeError("frequency setup unavailable")
        return atom_ids(atoms)

    monkeypatch.setattr("reactionflow.pathway.atom_ids", fail_frequency_setup)

    @contextmanager
    def provider(_stage: str):
        yield DoubleWell()

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

    assert outcome.status == "ci_neb_converged"
    assert outcome.converged
    assert len(outcome.images) == 5
    assert all(image.calc is None for image in outcome.images)
    validation = outcome.frequency_validation
    assert validation is not None
    assert validation.status == "failed"
    assert "frequency setup unavailable" in validation.message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frequency_delta", np.nan),
        ("frequency_delta", np.inf),
        ("imaginary_frequency_cutoff_cm1", np.nan),
        ("imaginary_frequency_cutoff_cm1", np.inf),
    ],
)
def test_frequency_config_rejects_non_finite_controls(field, value) -> None:
    with pytest.raises(ValueError):
        PathwayConfig(**{field: value})


class IndependentReactions(Calculator):
    """Two distant double wells allow an unrelated spectator bond to break."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        q = self.atoms.positions[[1, 3], 0] - 1.1
        forces = np.zeros((4, 3))
        forces[[1, 3], 0] = -4 * q * (q * q - 0.25)
        self.results = {"energy": float(np.sum((q * q - 0.25) ** 2)), "forces": forces}


@pytest.mark.parametrize(
    ("spectator_x", "expected_status", "expected_sides"),
    [
        (0.6, "connects_endpoints", ("reactant", "product")),
        (1.11, "does_not_connect", ("other", "other")),
    ],
)
def test_connectivity_checks_spectator_bonds(spectator_x, expected_status, expected_sides):
    reactant = Atoms("H4", positions=[[0, 0, 0], [0.6, 0, 0], [0, 10, 0], [0.6, 10, 0]])
    reactant.set_array("atom_id", np.arange(4))
    product = reactant.copy()
    product.positions[1, 0] = 1.6
    saddle = reactant.copy()
    saddle.positions[1, 0] = 1.1
    saddle.positions[3, 0] = spectator_x

    check = _check_connectivity(
        saddle,
        ((1.0, 0.0, 0.0),),
        (1,),
        IndependentReactions(),
        PathwayConfig(relax_fmax=1e-4),
        None,
        BondDetectorConfig(pair_thresholds={"H-H": (0.8, 1.2)}),
        endpoints=(reactant, product),
    )

    assert check.status == expected_status
    assert check.sides == expected_sides


class SpectatorGap(Calculator):
    """A spectator can descend to a stationary distance inside the hysteresis gap."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x, y = self.atoms.positions[[1, 3], 0]
        reaction = (x - 0.6) * (x - 1.6)
        spectator = (y - 1.0) * (y - 1.6)
        forces = np.zeros((4, 3))
        forces[1, 0] = -2 * reaction * (2 * x - 2.2)
        forces[3, 0] = -2 * spectator * (2 * y - 2.6)
        self.results = {"energy": float(reaction**2 + spectator**2), "forces": forces}


@pytest.mark.parametrize(
    ("endpoint_y", "saddle_y", "expected_status", "expected_sides"),
    [
        (1.6, 1.61, "connects_endpoints", ("reactant", "product")),
        (1.6, 1.01, "inconclusive", ("ambiguous", "ambiguous")),
        (1.0, 1.01, "inconclusive", ()),
    ],
)
def test_connectivity_rejects_spectator_hysteresis_gaps(
    endpoint_y, saddle_y, expected_status, expected_sides
):
    reactant = Atoms("H4", positions=[[0, 0, 0], [0.6, 0, 0], [0, 10, 0], [endpoint_y, 10, 0]])
    reactant.set_array("atom_id", np.arange(4))
    product = reactant.copy()
    product.positions[1, 0] = 1.6
    saddle = reactant.copy()
    saddle.positions[1, 0] = 1.1
    saddle.positions[3, 0] = saddle_y

    check = _check_connectivity(
        saddle,
        ((1.0, 0.0, 0.0),),
        (1,),
        SpectatorGap(),
        PathwayConfig(relax_fmax=1e-5),
        None,
        BondDetectorConfig(pair_thresholds={"H-H": (0.8, 1.2)}),
        endpoints=(reactant, product),
    )

    assert check.status == expected_status
    assert check.sides == expected_sides
    if endpoint_y == 1.0:
        assert "endpoint" in check.message and "hysteresis gap" in check.message


def test_bond_classification_uses_minimum_images_and_inclusive_formation_threshold():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[2, 10, 10], pbc=True)
    atoms.set_array("atom_id", np.array([10, 20]))
    detector_config = BondDetectorConfig(pair_thresholds={"H-H": (0.8, 1.6)})
    # The image at 1.2 A is in the gap, but the minimum image is bonded at the boundary.
    assert classify_bonds(atoms, detector_config) == ({(10, 20)}, set())
    atoms.positions[1, 0] = 0.9
    assert classify_bonds(atoms, detector_config) == (set(), {(10, 20)})
