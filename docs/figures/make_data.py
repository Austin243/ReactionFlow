"""Figure data produced by running ReactionFlow itself on small model systems.

Run it from the repository root with ReactionFlow installed from this checkout
(python -m pip install -e .):

    python docs/figures/make_data.py

Writes docs/figures/data/detection.json, pathway.json, ssneb.json, restart.json, and models.json.
The figure build (node docs/figures/build.mjs) reads only those files.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.data import covalent_radii

import reactionflow
from reactionflow import (
    BondChangeDetector,
    BondDetectorConfig,
    ExactRestartSnapshot,
    PathwayConfig,
    ReactionCandidate,
    ReactionTracker,
    assign_atom_ids,
    atom_ids,
    refine_pathway,
)
from reactionflow import pathway as pathway_module
from reactionflow.adapters.ase import ASECalculatorAdapter
from reactionflow.campaign import TrajectorySpec
from reactionflow.model_setup import model_catalog

DATA = Path(__file__).resolve().parent / "data"
ROOT = Path(__file__).resolve().parents[2]
# The data must describe this checkout, not some other installed copy of ReactionFlow.
if not Path(reactionflow.__file__).resolve().is_relative_to(ROOT / "src"):
    raise SystemExit(
        f"reactionflow is imported from {reactionflow.__file__}, not from {ROOT / 'src'}"
    )
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
COMMIT = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
).stdout.strip()


def dump(name: str, value: dict) -> None:
    DATA.mkdir(exist_ok=True)
    value = {"reactionflow_version": VERSION, "reactionflow_commit": COMMIT, **value}
    (DATA / name).write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    print("wrote", name)


# --- detection: the real detector and tracker on a prescribed C-N distance trace ---------------

# Pair distance per observation, in units of the summed covalent radii.
# fmt: off
RATIOS = [1.62, 1.55, 1.66, 1.48, 1.36, 1.12, 1.24, 1.41, 1.33, 1.21, 1.13, 1.07, 1.02, 0.98,
          1.04, 1.00, 1.09, 1.19, 1.06, 0.99, 1.03, 0.97]
# fmt: on


def detection() -> None:
    config = BondDetectorConfig()
    detector = BondChangeDetector(config)
    tracker = ReactionTracker(stability_frames=3)
    radius = float(covalent_radii[6] + covalent_radii[7])
    frames, events, candidates = [], [], []
    for frame, ratio in enumerate(RATIOS):
        atoms = assign_atom_ids(Atoms("CN", positions=[[0, 0, 0], [ratio * radius, 0, 0]]))
        events += [event.to_dict() for event in detector.process(atoms, frame=frame)]
        pending = detector.export_state()["pending"]
        emitted = tracker.process(
            atoms,
            frame=frame,
            stable_bonds=detector.stable_bonds,
            pending_bonds=detector.pending_bonds,
        )
        candidates += [
            {
                "reactant_frame": item.reactant_frame,
                "product_frame": item.product_frame,
                "observed_frame": item.observed_frame,
                "resolved": item.resolved,
            }
            for item in emitted
        ]
        regions = tracker._pending  # one stability window per connected region; one region here
        frames.append(
            {
                "frame": frame,
                "ratio": ratio,
                "distance_A": ratio * radius,
                "bonded": bool(detector.stable_bonds),
                "persistence_count": pending[0]["count"] if pending else 0,
                "stability_count": regions[0].count if regions else 0,
            }
        )
    form, breaking = config.threshold_for("C", "N")
    dump(
        "detection.json",
        {
            "pair": "C-N",
            "radius_sum_A": radius,
            "form_scale": config.form_scale,
            "break_scale": config.break_scale,
            "form_distance_A": form,
            "break_distance_A": breaking,
            "persistence_frames": config.persistence_frames,
            "stability_frames": tracker.stability_frames,
            "frames": frames,
            "events": events,
            "candidates": candidates,
        },
    )


# --- pathway: refine_pathway on a two-dimensional model potential -------------------------------

ORIGIN = np.array([6.0, 6.0, 6.0])
SURFACE = {"g": 3.0, "k": 6.0, "c": 0.8, "center": 0.1, "half_width": 1.1, "kz": 6.0}


def surface_energy(x: float, y: float, z: float = 0.0) -> tuple[float, np.ndarray]:
    """Curved valley from (-1, 0) to (1.2, 0) with its saddle near x = -0.1; energy and gradient."""

    g, k, c = SURFACE["g"], SURFACE["k"], SURFACE["c"]
    center, half, kz = SURFACE["center"], SURFACE["half_width"], SURFACE["kz"]
    floor = g * (x**4 / 4 - 0.1 * x**3 / 3 - 0.605 * x**2 - 0.12 * x + 0.2016666667)
    dfloor = g * (x + 1) * (x + 0.1) * (x - 1.2)
    valley = c * (1 - ((x - center) / half) ** 2)
    dvalley = -2 * c * (x - center) / half**2
    offset = y - valley
    energy = floor + 0.5 * k * offset**2 + 0.5 * kz * z**2
    return energy, np.array([dfloor - k * offset * dvalley, k * offset, kz * z])


class ModelSurface(Calculator):
    """The surface acts on atom 1; atom 0 only defines the bond that the reaction breaks."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        index = atom_ids(self.atoms).index(1)
        energy, gradient = surface_energy(*(self.atoms.positions[index] - ORIGIN))
        forces = np.zeros((len(self.atoms), 3))
        forces[index] = -gradient
        self.results = {"energy": float(energy), "forces": forces}


def pathway() -> None:
    anchor = np.array([-2.6, 0.0, 0.0])
    thresholds = (2.0, 2.4)
    snapshots = {"reactant": [-0.74, 0.30, 0.0], "product": [0.80, 0.62, 0.0]}

    def endpoint(position: list[float]) -> Atoms:
        atoms = Atoms(
            "He2", positions=[ORIGIN + anchor, ORIGIN + position], cell=[12] * 3, pbc=True
        )
        return assign_atom_ids(atoms)

    candidate = ReactionCandidate(
        reactant=endpoint(snapshots["reactant"]),
        product=endpoint(snapshots["product"]),
        atom_ids=(0, 1),
        reactant_bonds=frozenset({(0, 1)}),
        product_bonds=frozenset(),
        reactant_frame=0,
        product_frame=1,
        observed_frame=5,
        resolved=True,
    )

    # Record the moving atom at every optimizer step of each stage that refine_pathway runs:
    # relax reactant, relax product, NEB, climbing-image NEB, then the two connectivity sides.
    runs: list[dict] = []

    class RecordingFIRE(pathway_module.FIRE):
        def run(self, *args, **kwargs):
            target = self.atoms
            band = hasattr(target, "images")
            history: list = []

            def record() -> None:
                images = list(target.images) if band else [target]
                history.append(
                    [(image.positions[1] - ORIGIN)[:2].round(5).tolist() for image in images]
                )

            record()
            self.attach(record)
            converged = super().run(*args, **kwargs)
            record()
            runs.append(
                {
                    "kind": ("ci_neb" if target.climb else "neb") if band else "relax",
                    "steps": int(self.nsteps),
                    "history": history,
                }
            )
            return converged

    @contextmanager
    def provider(_stage: str):
        yield ModelSurface()

    original = pathway_module.FIRE
    pathway_module.FIRE = RecordingFIRE
    try:
        outcome = refine_pathway(
            candidate,
            calculator_provider=provider,
            config=PathwayConfig(),
            detector_config=BondDetectorConfig(pair_thresholds={"He-He": thresholds}),
        )
    finally:
        pathway_module.FIRE = original

    assert outcome.status == "ci_neb_converged", (outcome.status, outcome.message)
    assert [run["kind"] for run in runs] == ["relax", "relax", "neb", "ci_neb", "relax", "relax"]
    validation = outcome.frequency_validation
    connectivity = outcome.connectivity
    assert validation is not None and connectivity is not None
    neb_final = runs[2]["history"][-1]
    dump(
        "pathway.json",
        {
            "surface": SURFACE,
            "anchor": anchor[:2].tolist(),
            "thresholds_A": list(thresholds),
            "snapshots": {key: value[:2] for key, value in snapshots.items()},
            "config": {"images": PathwayConfig().images, "fmax": PathwayConfig().neb_fmax},
            "status": outcome.status,
            "barrier_eV": outcome.barrier,
            "energies_eV": list(outcome.energies),
            "images": [
                (image.positions[1] - ORIGIN)[:2].round(5).tolist() for image in outcome.images
            ],
            "neb_energies_eV": [float(surface_energy(x, y)[0]) for x, y in neb_final],
            "runs": runs,
            "frequency": {
                "status": validation.status,
                "saddle_index": validation.transition_state_index,
                "frequencies_cm1": list(validation.frequencies_cm1),
                "primary_mode": [list(vector) for vector in validation.primary_mode],
                "active_max_force_eV_A": validation.active_max_force_eV_A,
            },
            "connectivity": {
                "status": connectivity.status,
                "sides": list(connectivity.sides),
                "displacement_A": connectivity.displacement_A,
            },
        },
    )


# --- SSNEB: the volume-coupled double well from the test suite at three pressures ---------------


class VolumeDoubleWell(Calculator):
    """E = (q^2 - 1/4)^2 + K/2 (V - V0 - alpha q)^2, with fractional-coordinate q."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]
    stiffness = 0.02
    reference_volume = 216.0
    coupling = 12.0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        scaled = self.atoms.get_scaled_positions(wrap=False)
        q = 6 * (scaled[1, 0] - scaled[0, 0]) - 1.1
        volume_offset = self.atoms.get_volume() - self.reference_volume - self.coupling * q
        volume_derivative = self.stiffness * volume_offset
        derivative = 4 * q * (q * q - 0.25) - self.coupling * volume_derivative
        forces = np.zeros((len(self.atoms), 3))
        forces[0] = derivative * 6 * np.linalg.inv(self.atoms.cell)[:, 0]
        forces[1] = -forces[0]
        self.results = {
            "energy": (q * q - 0.25) ** 2 + 0.5 * self.stiffness * volume_offset**2,
            "forces": forces,
            "stress": np.diag([volume_derivative] * 3),
        }


def ssneb() -> None:
    reactant = Atoms("H2He", cell=[6, 6, 6], pbc=True)
    reactant.set_array("atom_id", np.arange(3))
    reactant.set_scaled_positions([[0.2, 0.2, 0.2], [0.3, 0.2, 0.2], [0.7, 0.7, 0.7]])
    product = reactant.copy()
    product.positions[0, 0] -= 0.5
    product.positions[1, 0] += 0.5
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
        yield VolumeDoubleWell()

    results = []
    for pressure_GPa in (0.0, 0.4, 0.8):
        outcome = refine_pathway(
            candidate,
            calculator_provider=provider,
            pressure_GPa=pressure_GPa,
            detector_config=BondDetectorConfig(pair_thresholds={"H-H": (0.8, 1.2)}),
            config=PathwayConfig(active_radius=0.2, relax_fmax=0.001, images=7, neb_fmax=0.002),
        )
        assert outcome.status == "ci_neb_converged", (outcome.status, outcome.message)
        results.append(
            {
                "pressure_GPa": pressure_GPa,
                "method": outcome.method,
                "barrier_quantity": outcome.barrier_quantity,
                "barrier_eV": outcome.barrier,
                "energies_eV": list(outcome.energies),
                "enthalpies_eV": list(outcome.enthalpies),
                "volumes_A3": list(outcome.volumes),
            }
        )
    dump("ssneb.json", {"results": results})


# --- exact restart: the generic ASE adapter on EMT copper, NPT ----------------------------------


def restart() -> None:
    total, split, chunk = 400, 150, 5
    trajectory = TrajectorySpec.from_dict(
        {
            "id": "emt-cu-npt",
            "total_steps": total,
            "timestep_fs": 2.0,
            "temperature_K": 600.0,
            "pressure_GPa": 1.0,
            "seed": 11,
            "conditions": {"hydrostatic": True},
        }
    )
    adapter = ASECalculatorAdapter(
        trajectory=trajectory,
        options={"calculator_factory": "ase.calculators.emt:EMT", "packages": ["ase"]},
    )

    def initial() -> Atoms:
        return assign_atom_ids(bulk("Cu", cubic=True).repeat(2))

    def advance(runtime, until: int, positions: list, volumes: list) -> None:
        while runtime.nsteps < until:
            runtime.run(chunk)
            positions.append(runtime.atoms.positions.copy())
            volumes.append(float(runtime.atoms.get_volume()))

    def new_series() -> tuple[list, list]:
        return [], []

    # Uninterrupted reference.
    reference_positions, reference_volumes = new_series()
    with adapter.start(initial()) as runtime:
        reference_positions.append(runtime.atoms.positions.copy())
        reference_volumes.append(float(runtime.atoms.get_volume()))
        advance(runtime, total, reference_positions, reference_volumes)

    # Stop at the split, write the exact checkpoint to disk, read it back, and continue.
    exact_positions, exact_volumes = new_series()
    with adapter.start(initial()) as runtime:
        exact_positions.append(runtime.atoms.positions.copy())
        exact_volumes.append(float(runtime.atoms.get_volume()))
        advance(runtime, split, exact_positions, exact_volumes)
        snapshot = runtime.snapshot()
        structure = runtime.atoms.copy()
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = ExactRestartSnapshot.read(snapshot.write(Path(temporary) / "checkpoint"))
    with adapter.restore(snapshot) as runtime:
        advance(runtime, total, exact_positions, exact_volumes)

    # Structural restart from the same step: atoms, momenta, and cell only. The integrator,
    # barostat, and random-number state start fresh.
    structural_positions = exact_positions[: split // chunk + 1]
    structural_volumes = exact_volumes[: split // chunk + 1]
    structure.calc = None
    with adapter.start(structure) as runtime:
        while runtime.nsteps < total - split:
            runtime.run(chunk)
            structural_positions.append(runtime.atoms.positions.copy())
            structural_volumes.append(float(runtime.atoms.get_volume()))

    def deviation(series: list) -> list[float]:
        return [
            float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))
            for a, b in zip(series, reference_positions, strict=True)
        ]

    exact_deviation = deviation(exact_positions)
    assert max(exact_deviation) == 0.0, max(exact_deviation)
    dump(
        "restart.json",
        {
            "system": "32-atom fcc Cu, ASE EMT, Langevin BAOAB NPT, 600 K, 1 GPa, 2 fs",
            "steps": list(range(0, total + 1, chunk)),
            "split_step": split,
            "atoms": len(initial()),
            "reference_volume_A3": reference_volumes,
            "exact_volume_A3": exact_volumes,
            "structural_volume_A3": structural_volumes,
            "exact_rms_deviation_A": exact_deviation,
            "structural_rms_deviation_A": deviation(structural_positions),
            "exact_max_abs_position_difference_A": float(
                max(
                    np.abs(a - b).max()
                    for a, b in zip(exact_positions, reference_positions, strict=True)
                )
            ),
        },
    )


def models() -> None:
    """The built-in model catalog, exactly as `reactionflow models` prints it."""

    dump("models.json", {"backends": model_catalog()})


if __name__ == "__main__":
    detection()
    pathway()
    ssneb()
    restart()
    models()
