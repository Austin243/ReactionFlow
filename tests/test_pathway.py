from __future__ import annotations

import json
from contextlib import contextmanager
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from reactionflow import (
    BondDetectorConfig,
    PathwayConfig,
    ReactionCandidate,
    atom_ids,
    refine_pathway,
)
from reactionflow.cli import main
from reactionflow.detection import classify_bonds
from reactionflow.pathway import FIRE, NEB, _check_connectivity, check_saddle


class DoubleWell(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]
    tilt = 0.0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        index = atom_ids(self.atoms).index(1)
        x, y, z = self.atoms.positions[index]
        energy = (x * x - 1) ** 2 + self.tilt * x + y * y + z * z
        forces = np.zeros((len(self.atoms), 3))
        forces[index] = (-4 * x * (x * x - 1) - self.tilt, -2 * y, -2 * z)
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


class Wells(Calculator):
    """Each atom in its own harmonic well; keeps every geometry it is asked about."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, centers) -> None:
        super().__init__()
        self.centers = np.asarray(centers, dtype=float)
        self.seen: list[np.ndarray] = []

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.seen.append(self.atoms.positions.copy())
        offset = self.atoms.positions - self.centers
        self.results = {
            "energy": float((offset * offset).sum()),
            "forces": -2 * offset,
            "stress": np.zeros(6),
        }


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
            relax_fmax=0.02,
            relax_steps=100,
            images=5,
            neb_fmax=0.03,
            neb_steps=250,
            ci_neb_steps=250,
        ),
        detector_config=BondDetectorConfig(pair_thresholds={"He-He": (5.07, 5.13)}),
    )

    assert outcome.status == "saddle_validated", outcome.message
    assert outcome.converged
    assert outcome.barrier == pytest.approx(1.0, abs=0.02)
    assert len(outcome.energies) == len(outcome.images) == 5
    validation = outcome.frequency_validation
    assert validation is not None
    assert validation.status == "one_imaginary_mode"
    assert validation.transition_state_index == 1 + int(np.argmax(outcome.energies[1:-1]))
    # Only atom 1 moves through the saddle, so the six atoms that move most are all three.
    assert validation.active_atom_ids == (0, 1, 2)
    assert validation.active_max_force_eV_A is not None
    assert validation.active_max_force_eV_A < 0.03
    assert len(validation.eigenvalues_eV_A2) == len(validation.frequencies_cm1) == 9
    # The well curves by -4 eV/Å² at its top, which is 521i cm⁻¹ for helium.
    assert validation.eigenvalues_eV_A2[0] == pytest.approx(-4, abs=0.05)
    assert validation.eigenvalues_eV_A2[1] > -validation.curvature_cutoff_eV_A2
    assert validation.frequencies_cm1[0] == pytest.approx(-521.3, abs=2)
    assert validation.tangent_overlap == pytest.approx(1)
    assert outcome.connectivity is not None
    assert outcome.connectivity.status == "connects_endpoints"
    assert set(outcome.connectivity.sides) == {"reactant", "product"}
    np.testing.assert_allclose(validation.primary_mode[0], 0, atol=1e-10)
    np.testing.assert_allclose(np.abs(validation.primary_mode[1]), [1, 0, 0], atol=1e-10)
    np.testing.assert_allclose(validation.primary_mode[2], 0, atol=1e-10)
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
@pytest.mark.parametrize("hop", [0.0, -1.0])
def test_product_moves_only_the_reacting_atoms_so_spectators_settle_alike(pressure, hop) -> None:
    # Atom 1 leaves atom 0. Spectator 2 sits on the ridge between its two wells, as it can in a
    # hot frame: bonded to atom 3 in one well and not in the other. Atom 1 tilts that ridge
    # oppositely in the two frames, so relaxed separately the endpoints settle 2 differently.
    # With a hop, 2 has also dropped into its unbonded well by the product frame. It is bonded to
    # neither reacting atom, so that move must not reach the product.
    reactant = Atoms(
        "He4",
        positions=[[2, 15, 15], [4.1, 15, 15], [15, 15, 15], [18, 15, 15]],
        cell=[30, 30, 30],
        pbc=True,
    )
    reactant.set_array("atom_id", np.arange(4))
    product = reactant.copy()
    product.positions[1, 0] = 6.1
    product.positions[2, 0] += hop
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

    assert outcome.status == "saddle_validated", outcome.message
    first, last = outcome.images[0], outcome.images[-1]
    assert first.get_distance(0, 1) < 2.5 < 3 < last.get_distance(0, 1)
    assert first.get_distance(2, 3) < 2.5 and last.get_distance(2, 3) < 2.5
    assert outcome.connectivity is not None
    assert outcome.connectivity.status == "connects_endpoints"


@pytest.mark.parametrize("pressure", [None, 0.0])
def test_product_start_places_the_reacting_atoms_where_the_product_frame_has_them(pressure) -> None:
    # Atom 1 leaves atom 0, taking its partner 2 along. Atom 2 is stretched in the reactant
    # frame, so relaxing the reactant swings it round to the far side of atom 1. The product
    # relaxation must start with 0, 1 and 2 where the product frame has them and spectator 3
    # where the relaxed reactant has it. Adding the move between the frames to 2's relaxed
    # position instead would set it 2.2 A from atom 3, inside bonding distance.
    reactant = Atoms(
        "He4",
        positions=[[2, 15, 15], [4.1, 15, 15], [4.1, 17.4, 15], [6.5, 10.5, 15]],
        cell=[30, 30, 30],
        pbc=True,
    )
    reactant.set_array("atom_id", np.arange(4))
    product = reactant.copy()
    product.positions[1:3, 0] = 6.5
    candidate = ReactionCandidate(
        reactant=reactant,
        product=product,
        atom_ids=(0, 1, 2),
        reactant_bonds=frozenset({(0, 1), (1, 2)}),
        product_bonds=frozenset({(1, 2)}),
        reactant_frame=0,
        product_frame=1,
        observed_frame=2,
        resolved=True,
    )
    relaxed = reactant.positions.copy()
    relaxed[2, 1] = 12.7
    wells = {"relax_reactant": Wells(relaxed), "relax_product": Wells(product.positions)}

    @contextmanager
    def provider(stage: str):
        if stage not in wells:
            raise RuntimeError("stop after the endpoints")
        yield wells[stage]

    outcome = refine_pathway(
        candidate,
        calculator_provider=provider,
        config=PathwayConfig(relax_fmax=0.01, relax_steps=500),
        detector_config=BondDetectorConfig(pair_thresholds={"He-He": (2.5, 3.0)}),
        pressure_GPa=pressure,
    )

    assert outcome.message == "stop after the endpoints"
    start = wells["relax_product"].seen[0]
    assert np.allclose(start[:3], product.positions[:3], atol=1e-6)
    assert np.allclose(start[3], relaxed[3], atol=1e-6)


def test_band_whose_highest_point_is_an_endpoint_reports_no_barrier(monkeypatch) -> None:
    # Atom 1's product well sits about 2 eV above its reactant well, with the barrier close to
    # it. The band is left as interpolated, so its one middle image stays below the product,
    # as when a band jumps across the reaction between two neighboring images.
    original_run = FIRE.run

    def leave_band_as_interpolated(self, *args, **kwargs):
        return True if isinstance(self.atoms, NEB) else original_run(self, *args, **kwargs)

    monkeypatch.setattr(FIRE, "run", leave_band_as_interpolated)
    reactant = Atoms("He2", positions=[[-3, 0, 0], [-1.1, 0, 0]], cell=[30, 30, 30], pbc=True)
    reactant.set_array("atom_id", np.arange(2))
    product = reactant.copy()
    product.positions[1, 0] = 0.84
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
    def uphill(_stage: str):
        calculator = DoubleWell()
        calculator.tilt = 1.0
        yield calculator

    outcome = refine_pathway(
        candidate,
        calculator_provider=uphill,
        config=PathwayConfig(relax_fmax=0.01, relax_steps=500, images=3),
        detector_config=BondDetectorConfig(pair_thresholds={"He-He": (2.5, 3.0)}),
    )

    assert outcome.status == "no_saddle"
    assert outcome.energies[1] < outcome.energies[-1] == max(outcome.energies)
    assert outcome.barrier is None
    assert outcome.frequency_validation is None and outcome.connectivity is None
    assert outcome.message == "an endpoint is the highest point, so the band holds no saddle"


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
        config=PathwayConfig(relax_fmax=0.02, relax_steps=100),
        detector_config=thresholds,
    )

    @contextmanager
    def constant_force(_stage: str):
        yield ConstantForce()

    relaxation_failed = refine_pathway(
        pathway_candidate(),
        calculator_provider=constant_force,
        config=PathwayConfig(relax_fmax=1e-12, relax_steps=1),
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


class InTurn(Calculator):
    """Atom 1 leaves atom 0, and atom 3 leaves atom 2, each across a double well along x.

    Each well has its minima at 2.25 and 3.25 A from the partner, either side of the 2.5 to 3.0 A
    gap. Atom 1's barrier is 0.25 eV and atom 3's 0.5 eV.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        ids = atom_ids(self.atoms)
        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))
        for mover, partner, height in ((1, 0, 0.25), (3, 2, 0.5)):
            first, second = ids.index(mover), ids.index(partner)
            x, y, z = self.atoms.positions[first] - self.atoms.positions[second]
            s = 2 * (x - 2.75)
            energy += height * (s * s - 1) ** 2 + y * y + z * z
            forces[first] = (-8 * height * s * (s * s - 1), -2 * y, -2 * z)
            forces[second] = -forces[first]
        self.results = {"energy": float(energy), "forces": forces}


IN_TURN_BONDS = BondDetectorConfig(pair_thresholds={"He-He": (2.5, 3.0)})


def in_turn(first: float, second: float) -> Atoms:
    """Atoms 1 and 3 at these distances from their partners 0 and 2, 10 A apart."""

    atoms = Atoms(
        "He4",
        positions=[[0, 0, 0], [first, 0, 0], [0, 10, 0], [second, 10, 0]],
        cell=[20, 20, 20],
        pbc=True,
    )
    atoms.set_array("atom_id", np.arange(4))
    return atoms


def staged_provider(stages: list[str]):
    @contextmanager
    def provider(stage: str):
        stages.append(stage)
        yield InTurn()

    return provider


STEP_CONFIG = {
    "relax_fmax": 0.01,
    "relax_steps": 500,
    "images": 5,
    "neb_fmax": 0.02,
    "neb_steps": 500,
    "ci_neb_steps": 500,
}


@pytest.mark.parametrize("reactant_frame", [0, 1])
def test_bonds_that_break_in_turn_are_refined_one_step_each(reactant_frame) -> None:
    # Atom 1 leaves in frame 1, where it is still inside the gap but relaxes out, then atom 3
    # leaves by the product frame. Taken from frame 1 with no lookback, the reactant relaxes into
    # the intermediate, so only the second step can be refined.
    frames = {0: in_turn(2.2, 2.3), 1: in_turn(2.8, 2.3), 2: in_turn(3.3, 2.4)}
    candidate = ReactionCandidate(
        reactant=frames[reactant_frame],
        product=in_turn(3.2, 3.3),
        atom_ids=(0, 1, 2, 3),
        reactant_bonds=frozenset({(0, 1), (2, 3)}),
        product_bonds=frozenset(),
        reactant_frame=reactant_frame,
        product_frame=3,
        observed_frame=5,
        resolved=True,
    )
    stages: list[str] = []

    outcome = refine_pathway(
        candidate,
        calculator_provider=staged_provider(stages),
        config=PathwayConfig(**STEP_CONFIG, reactant_lookback=0),
        detector_config=IN_TURN_BONDS,
        frames=frames,
    )

    assert outcome.status == "stepwise" and outcome.barrier is None
    assert outcome.frames == (reactant_frame, 3)
    first, second = outcome.steps
    assert (first.reactant_bonds, first.product_bonds) == ({(0, 1), (2, 3)}, {(2, 3)})
    assert (second.reactant_bonds, second.product_bonds) == ({(2, 3)}, set())
    assert second.outcome.status == "saddle_validated", second.outcome.message
    assert second.outcome.barrier == pytest.approx(0.5, abs=0.01)
    assert second.outcome.frames == (1, 3)
    if reactant_frame == 0:
        assert first.outcome.status == "saddle_validated", first.outcome.message
        assert first.outcome.barrier == pytest.approx(0.25, abs=0.01)
        assert first.outcome.frames == (0, 1)
        # The second step starts where the first one ended.
        np.testing.assert_array_equal(
            second.outcome.images[0].positions, first.outcome.images[-1].positions
        )
        expected = ["relax_reactant", "relax_frame", "relax_frame", *["relax_product", "neb"] * 2]
    else:
        assert first.outcome.status == "unresolved"
        assert first.outcome.message == "the reactant frame relaxes partway along the event"
        assert first.outcome.frames is None
        expected = ["relax_reactant", "relax_frame", "relax_product", "neb"]
    assert stages == expected
    assert [step["status"] for step in outcome.to_dict()["steps"]] == [
        first.outcome.status,
        "saddle_validated",
    ]


@pytest.mark.parametrize(
    ("lookback", "spectator", "status", "frames", "stages"),
    [
        # Frame 1 has atom 1 still bonded, so the reactant comes from there.
        (10, 2.3, "saddle_validated", (1, 3), ["relax_reactant", "relax_product", "neb"]),
        # The tracker's frame has atom 1 inside the gap and relaxes into the product.
        (0, 2.3, "collapsed", (2, 3), ["relax_reactant", "relax_product"]),
        # The spectator pair is broken in frame 1 but not in the tracker's frame.
        (10, 3.3, "collapsed", (2, 3), ["relax_reactant", "relax_reactant", "relax_product"]),
    ],
)
def test_reactant_comes_from_the_latest_frame_before_the_bond_began_to_break(
    lookback, spectator, status, frames, stages
) -> None:
    saved = {0: in_turn(2.2, 2.3), 1: in_turn(2.45, spectator)}
    candidate = ReactionCandidate(
        reactant=in_turn(2.8, 2.3),
        product=in_turn(3.3, 2.3),
        atom_ids=(0, 1),
        reactant_bonds=frozenset({(0, 1)}),
        product_bonds=frozenset(),
        reactant_frame=2,
        product_frame=3,
        observed_frame=5,
        resolved=True,
    )
    seen: list[str] = []

    outcome = refine_pathway(
        candidate,
        calculator_provider=staged_provider(seen),
        config=PathwayConfig(**STEP_CONFIG, reactant_lookback=lookback),
        detector_config=IN_TURN_BONDS,
        frames=saved,
    )

    # One bond changes, so the event is one band, without steps.
    assert outcome.status == status, outcome.message
    assert outcome.frames == frames and outcome.steps == ()
    assert seen == stages
    if status == "saddle_validated":
        assert outcome.barrier == pytest.approx(0.25, abs=0.01)


def test_step_settings_must_be_whole_and_boolean() -> None:
    with pytest.raises(ValueError, match="reactant_lookback"):
        PathwayConfig(reactant_lookback=-1)
    with pytest.raises(ValueError, match="split_steps"):
        PathwayConfig(split_steps=1)


class Saddle(Calculator):
    """Atom 1 crosses (curvature / 4)(x² - 1)² along x, which curves by -curvature at x = 0."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, curvature: float = 4.0, bend: float = 2.0) -> None:
        super().__init__()
        self.curvature = curvature
        self.bend = bend

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        index = atom_ids(self.atoms).index(1)
        x, y, z = self.atoms.positions[index]
        energy = self.curvature / 4 * (x * x - 1) ** 2 + self.bend / 2 * y * y + z * z
        forces = np.zeros((len(self.atoms), 3))
        forces[index] = (-self.curvature * x * (x * x - 1), -self.bend * y, -2 * z)
        self.results = {"energy": float(energy), "forces": forces}


# Atom 0 is bonded to atom 1 at x = -1 (5.06 A) and not at x = +1 (5.14 A).
SADDLE_BONDS = BondDetectorConfig(pair_thresholds={"He-He": (5.07, 5.13)})


def band(*xs: float, spectators: tuple = ()) -> list[Atoms]:
    """Atom 1 at each x; each spectator gives atom 2's position in the image with the same index."""

    images = []
    for index, x in enumerate(xs):
        positions = [[-0.2, 5, 0], [x, 0, 0], *spectators[index : index + 1]]
        atoms = Atoms(f"He{len(positions)}", positions=positions, cell=[10, 10, 10], pbc=True)
        atoms.set_array("atom_id", np.arange(len(atoms)))
        images.append(atoms)
    return images


@pytest.mark.parametrize(
    ("curvature", "bend", "status", "modes"),
    [
        (4.0, 2.0, "saddle_validated", "one_imaginary_mode"),
        # 52i cm⁻¹ for helium, beyond the old 50 cm⁻¹ rule, but flatter than -0.05 eV/Å².
        (0.04, 2.0, "no_saddle", "zero_imaginary_modes"),
        (4.0, -2.0, "saddle_rejected", "multiple_imaginary_modes"),
    ],
)
def test_saddle_gate_counts_curvatures_beyond_the_cutoff(curvature, bend, status, modes) -> None:
    outcome = check_saddle(band(-1, 0, 1), Saddle(curvature, bend), detector_config=SADDLE_BONDS)

    assert outcome.status == status, outcome.message
    validation = outcome.frequency_validation
    assert validation is not None and validation.status == modes
    if status == "saddle_validated":
        assert outcome.barrier == pytest.approx(curvature / 4)
        assert outcome.connectivity is not None
        assert outcome.connectivity.status == "connects_endpoints"
    else:
        assert outcome.barrier is None
    if status == "no_saddle":
        assert validation.frequencies_cm1[0] == pytest.approx(-52.1, abs=0.5)
        assert outcome.connectivity is None


def test_saddle_that_misses_an_end_of_the_band_is_rejected() -> None:
    # The band's product end also bonds atom 2 to atom 0, which the saddle never leads to.
    images = band(-1, 0, 1, spectators=([5, 5, 5], [5, 5, 5], [-0.2, 5, 5]))

    outcome = check_saddle(images, Saddle(), detector_config=SADDLE_BONDS)

    assert outcome.status == "saddle_rejected", outcome.message
    assert outcome.barrier is None
    assert outcome.connectivity is not None
    assert outcome.connectivity.status == "does_not_connect"
    assert sorted(outcome.connectivity.sides) == ["other", "reactant"]


def test_ends_with_the_same_bonds_are_told_apart_by_distance() -> None:
    # No pair is bonded anywhere, as in a conformer change, so each side reaches the nearer end.
    unbonded = BondDetectorConfig(pair_thresholds={"He-He": (1.0, 1.1)})

    outcome = check_saddle(band(-1, 0, 1), Saddle(), detector_config=unbonded)

    assert outcome.status == "saddle_validated", outcome.message
    assert outcome.connectivity is not None
    assert sorted(outcome.connectivity.sides) == ["product", "reactant"]


def test_check_leaves_the_band_unchanged() -> None:
    images = band(-1, 0.3, 1)
    before = [image.positions.copy() for image in images]

    outcome = check_saddle(images, Saddle(), detector_config=SADDLE_BONDS)

    assert outcome.frequency_validation is not None
    for image, positions in zip(images, before, strict=True):
        assert image.calc is None
        np.testing.assert_array_equal(image.positions, positions)


def test_cli_check_writes_the_verdict_beside_the_band(tmp_path, monkeypatch, capsys) -> None:
    # A check inside a Slurm job takes its trajectory from --index, not from the job's tasks.
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "3")
    images = band(-1, 0, 1)
    write(tmp_path / "path.extxyz", images)
    write(tmp_path / "start.extxyz", images[0])
    campaign = {
        "schema_version": 2,
        "structure": "start.extxyz",
        "output_root": "runs",
        "require_gpu": False,
        "adapter_profiles": {
            "saddle": {
                "factory": "reactionflow.adapters.ase:create_adapter",
                "options": {"calculator_factory": f"{__name__}:Saddle"},
            }
        },
        "reaction_run": {"detector": {"pair_thresholds": {"He-He": [5.07, 5.13]}}},
        "trajectories": [
            {
                "id": "only",
                "adapter_profile": "saddle",
                "total_steps": 1,
                "timestep_fs": 1.0,
                "temperature_K": 300.0,
                "pressure_GPa": None,
                "seed": 1,
            }
        ],
    }
    (tmp_path / "campaign.json").write_text(json.dumps(campaign))

    assert main(["check", str(tmp_path / "campaign.json"), str(tmp_path / "path.extxyz")]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "saddle_validated"
    assert printed["barrier_eV"] == pytest.approx(1.0)
    written = json.loads((tmp_path / "path.check.json").read_text())
    assert printed["check"] == str(tmp_path / "path.check.json")
    assert written["trajectory_id"] == "only" and written["status"] == "saddle_validated"
    assert written["frequency_validation"]["status"] == "one_imaginary_mode"
    assert written["connectivity"]["status"] == "connects_endpoints"
    assert "method" not in written


def test_hessian_failure_leaves_the_saddle_unchecked(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic Hessian failure")

    monkeypatch.setattr("reactionflow.pathway._hessian", fail)

    @contextmanager
    def provider(_stage: str):
        yield DoubleWell()

    outcome = refine_pathway(
        pathway_candidate(),
        calculator_provider=provider,
        config=PathwayConfig(
            relax_fmax=0.02,
            relax_steps=100,
            images=5,
            neb_fmax=0.03,
            neb_steps=250,
            ci_neb_steps=250,
        ),
        detector_config=BondDetectorConfig(pair_thresholds={"He-He": (5.07, 5.13)}),
    )

    assert outcome.status == "saddle_unchecked"
    assert outcome.converged and outcome.barrier is None
    assert len(outcome.images) == len(outcome.energies) == 5
    assert all(image.calc is None for image in outcome.images)
    validation = outcome.frequency_validation
    assert validation is not None
    assert validation.status == "failed"
    assert validation.message == "synthetic Hessian failure"
    assert "synthetic Hessian failure" in outcome.message
    assert outcome.connectivity is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frequency_delta", np.nan),
        ("frequency_delta", np.inf),
        ("curvature_cutoff_eV_A2", np.nan),
        ("curvature_cutoff_eV_A2", np.inf),
    ],
)
def test_frequency_config_rejects_non_finite_controls(field, value) -> None:
    with pytest.raises(ValueError):
        PathwayConfig(**{field: value})


def test_settings_of_the_retired_frequency_check_are_ignored() -> None:
    old = {"active_radius": 4.0, "imaginary_frequency_cutoff_cm1": 50.0, "relax_fmax": 0.02}

    assert PathwayConfig.from_dict(old) == PathwayConfig(relax_fmax=0.02)
    with pytest.raises(TypeError):
        PathwayConfig.from_dict({"active_radii": 4.0})


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
