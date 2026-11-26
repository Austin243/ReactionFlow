from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units
from ase.constraints import FixAtoms
from ase.io import read, write
from test_run import (
    LeaseCounter,
    PairDoubleWell,
    ScriptedRuntimeProvider,
    pair,
    run_config,
    run_pair,
)
from test_ssneb import VolumeDoubleWell

from reactionflow import BondDetectorConfig, PathwayConfig, ReactionRun, ReactionRunConfig
from reactionflow.cli import main
from reactionflow.status import _occurrences


def campaign(
    tmp_path: Path, calculator: str | None, reaction_run: dict, *, pressure=None, ids=("only",)
) -> str:
    """Write a campaign with these detection and pathway settings and return its path.

    Its one model is the named analytic calculator, or a module that cannot be imported.
    """

    write(tmp_path / "start.extxyz", pair(0.6))
    profile = (
        {"factory": "never_imported:create_adapter", "options": {}}
        if calculator is None
        else {
            "factory": "reactionflow.adapters.ase:create_adapter",
            "options": {"calculator_factory": calculator},
        }
    )
    value = {
        "schema_version": 2,
        "structure": "start.extxyz",
        "output_root": "runs",
        "require_gpu": False,
        "adapter_profiles": {"model": profile},
        "reaction_run": reaction_run,
        "trajectories": [
            {
                "id": name,
                "adapter_profile": "model",
                "total_steps": 1,
                "timestep_fs": 1.0,
                "temperature_K": 300.0,
                "pressure_GPa": pressure,
                "seed": 1,
            }
            for name in ids
        ],
    }
    (tmp_path / "campaign.json").write_text(json.dumps(value))
    return str(tmp_path / "campaign.json")


def ends(tmp_path: Path, reactant: Atoms, product: Atoms) -> list[str]:
    write(tmp_path / "reactant.extxyz", reactant)
    write(tmp_path / "product.extxyz", product)
    return [str(tmp_path / "reactant.extxyz"), str(tmp_path / "product.extxyz")]


def band(capsys, *arguments: object) -> dict:
    """Run ``reactionflow band`` and return the line it prints."""

    assert main(["band", *map(str, arguments)]) == 0
    return json.loads(capsys.readouterr().out)


def files(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_band_from_the_ends_of_a_run_refinement_reproduces_its_barrier_and_verdict(
    tmp_path, capsys
) -> None:
    (record,) = run_pair(tmp_path / "runs/pair").occurrences.records()
    made = tmp_path / "runs/pair/pathways" / record.occurrence_id
    expected = json.loads((made / "result.json").read_text())
    path = campaign(tmp_path, "test_run:PairDoubleWell", run_config().to_dict())
    first, last = f"{made}/images.traj@0", f"{made}/images.traj@-1"
    out = tmp_path / "again"

    assert main(["band", path, first, last, "--out", str(out)]) == 0

    captured = capsys.readouterr()
    assert "warning" not in captured.err  # each end names its frame
    printed = json.loads(captured.out)
    folder = out / "pathways/band"
    assert printed["band"] == str(folder)
    result = json.loads((folder / "result.json").read_text())
    assert printed["status"] == result["status"] == expected["status"] == "saddle_validated"
    assert result["barrier_eV"] == pytest.approx(expected["barrier_eV"], abs=1e-6)
    assert printed["barrier_eV"] == result["barrier_eV"]
    assert (result["method"], result["pressure_GPa"]) == ("neb", None)
    assert (printed["barrier_quantity"], printed["pressure_GPa"]) == ("potential_energy", None)
    for check in ("frequency_validation", "connectivity"):
        assert result[check]["status"] == expected[check]["status"]
    assert len(read(folder / "images.traj", ":")) == len(read(made / "images.traj", ":")) == 3
    # The folder holds what a trajectory's folder holds for one event, in the same files.
    assert result.keys() == expected.keys()
    assert {item.name for item in out.iterdir()} == {"candidates", "pathways", "reactions.sqlite3"}
    assert {item.name for item in folder.iterdir()} == {"result.json", "images.traj"}
    assert {item.name for item in (out / "candidates/band").iterdir()} == {
        "candidate.json",
        "reactant.traj",
        "product.traj",
    }
    (row,) = _occurrences(out)  # the registry status reads a trajectory's events from
    assert (row["occurrence_id"], row["class_id"]) == ("band", result["class_id"])

    assert main(["check", path, str(folder / "images.traj")]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["status"] == "saddle_validated"
    assert checked["barrier_eV"] == pytest.approx(result["barrier_eV"])


# The detection and pathway settings of the SSNEB test, for a run that reads the bond at each
# observation.
SOLID = ReactionRunConfig(
    observation_interval=9,
    detector=BondDetectorConfig(persistence_frames=1, pair_thresholds={"H-H": (0.8, 1.2)}),
    candidate_stability_frames=1,
    pathway=PathwayConfig(relax_fmax=0.001, images=5, neb_fmax=0.002),
)


@contextmanager
def volume_calculators(_stage: str):
    yield VolumeDoubleWell()


def refined_in_a_run(root: Path, pressure: float) -> Path:
    """Refine the step of the SSNEB test inside a run at a pressure, and return its pathway folder.

    H-H breaks across a double well that is coupled to the cell volume.
    """

    start = Atoms("H2He", cell=[6, 6, 6], pbc=True)
    start.set_scaled_positions([[0.2, 0.2, 0.2], [0.3, 0.2, 0.2], [0.7, 0.7, 0.7]])
    ReactionRun.create(root, config=SOLID).run(
        start,
        runtime_provider=ScriptedRuntimeProvider(LeaseCounter(), script={1: 2.8}),
        pathway_calculator_provider=volume_calculators,
        total_steps=18,
        pressure_GPa=pressure,
    )
    (folder,) = (root / "pathways").iterdir()
    return folder


def enthalpy_barrier(result: dict) -> float:
    return max(result["enthalpies_eV"]) - result["enthalpies_eV"][0]


@pytest.mark.parametrize(
    ("found_at", "trajectory_GPa", "options", "pressure"),
    [
        # With no --pressure the band is at the trajectory's pressure.
        (0.0, 0.005 / units.GPa, (), 0.005 / units.GPa),
        # --pressure replaces it, and zero is a pressure, not a fixed cell.
        (0.005 / units.GPa, 20.0, ("--pressure", "0"), 0.0),
    ],
)
def test_band_takes_a_step_to_another_pressure_and_gives_the_enthalpy_barrier_run_gives_there(
    tmp_path, capsys, found_at, trajectory_GPa, options, pressure
) -> None:
    # The ends come from a refinement at one pressure, and the band is compared with a refinement
    # inside a run at its own.
    found = refined_in_a_run(tmp_path / "runs/found", found_at)
    made = refined_in_a_run(tmp_path / "runs/made", pressure)
    expected = json.loads((made / "result.json").read_text())
    model = "test_ssneb:VolumeDoubleWell"
    path = campaign(tmp_path, model, SOLID.to_dict(), pressure=trajectory_GPa)
    first, last = f"{found}/images.traj@0", f"{found}/images.traj@-1"
    out = tmp_path / "again"

    printed = band(capsys, path, first, last, "--out", out, *options)

    folder = out / "pathways/band"
    result = json.loads((folder / "result.json").read_text())
    assert (result["method"], result["pressure_GPa"]) == ("ssneb", pressure)
    assert (printed["barrier_quantity"], printed["pressure_GPa"]) == ("enthalpy", pressure)
    # This saddle is mostly a change of cell, which the fixed-cell Hessian does not see, so the
    # gate records no barrier for either, and the band holds the enthalpy barrier.
    assert printed["status"] == result["status"] == expected["status"] == "no_saddle"
    assert expected["method"] == "ssneb" and np.ptp(expected["volumes_A3"]) > 10
    assert enthalpy_barrier(result) == pytest.approx(enthalpy_barrier(expected), abs=1e-4)
    # The ends relaxed their cells at the band's pressure. Moving the pair as a whole costs
    # nothing, so the springs leave the other two images loose along the path. Each of two
    # relaxed cells is within the SSNEB test's 0.03 Å³ of its minimum.
    highest = int(np.argmax(expected["enthalpies_eV"]))
    assert int(np.argmax(result["enthalpies_eV"])) == highest
    np.testing.assert_allclose(
        np.asarray(result["volumes_A3"])[[0, highest, -1]],
        np.asarray(expected["volumes_A3"])[[0, highest, -1]],
        atol=0.06,
    )

    images = str(folder / "images.traj")
    assert main(["check", path, images, "--pressure", repr(pressure)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == result["status"]
    checked = json.loads((folder / "images.check.json").read_text())
    np.testing.assert_allclose(checked["enthalpies_eV"], result["enthalpies_eV"])

    assert main(["export-electrace", str(folder), "--out", str(tmp_path / "dft")]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert (exported["frames"], exported["focus_atoms"]) == (5, [0, 1])
    assert (exported["pressure_GPa"], exported["status"]) == (pressure, "no_saddle")
    assert (tmp_path / "dft/04/POSCAR").is_file()


@pytest.mark.parametrize(
    ("thresholds", "distances", "status", "reason", "relaxed"),
    [
        # The product is past its break distance but short of the barrier top at 1.1 Å.
        (
            [0.7, 0.9],
            (0.6, 1.0),
            "collapsed",
            "relaxed endpoints occupy the same changed-bond basin",
            (0.6, 0.6),
        ),
        # The reactant's well lies between the two thresholds.
        (
            [0.5, 0.7],
            (0.45, 1.6),
            "unresolved",
            "a relaxed changed bond remains inside the hysteresis gap",
            (0.6, 1.6),
        ),
    ],
)
def test_ends_whose_bonds_change_on_relaxation_are_refused_with_the_reason(
    tmp_path, capsys, thresholds, distances, status, reason, relaxed
) -> None:
    settings = run_config().to_dict()
    settings["detector"]["pair_thresholds"] = {"H-H": thresholds}
    path = campaign(tmp_path, "test_run:PairDoubleWell", settings)
    out = tmp_path / "out"

    printed = band(capsys, path, *ends(tmp_path, *map(pair, distances)), "--out", out)

    assert (printed["status"], printed["message"], printed["barrier_eV"]) == (status, reason, None)
    result = json.loads((out / "pathways/band/result.json").read_text())
    assert (result["status"], result["barrier_eV"], result["energies_eV"]) == (status, None, [])
    # The two relaxed ends are kept, which shows where each went.
    images = read(out / "pathways/band/images.traj", ":")
    assert [image.get_distance(0, 1) for image in images] == pytest.approx(relaxed, abs=0.01)


def far_pairs(first: float, second: float) -> Atoms:
    return Atoms("H2CO", positions=[[0, 0, 0], [first, 0, 0], [10, 0, 0], [10 + second, 0, 0]])


def with_ids(atoms: Atoms) -> Atoms:
    atoms.set_array("atom_id", np.arange(len(atoms)))
    return atoms


@pytest.mark.parametrize(
    ("reactant", "product", "reason"),
    [
        (pair(0.6), pair(0.65), "same bonds"),
        # run refines changes this far apart as two events, each with its own band.
        (far_pairs(0.6, 0.6), far_pairs(1.6, 1.6), "2 separate regions"),
        (pair(0.6), Atoms("HHe", positions=[[0, 0, 0], [1.6, 0, 0]]), "same atoms"),
        # IDs in one file only would pair its atoms with the other's by guesswork.
        (with_ids(pair(0.6)), pair(1.6), "atom IDs"),
    ],
)
def test_structures_that_are_not_one_step_are_refused_before_the_model_loads(
    tmp_path, reactant, product, reason
) -> None:
    settings = run_config().to_dict()
    settings["detector"]["pair_thresholds"]["C-O"] = [0.8, 1.2]
    path = campaign(tmp_path, None, settings)

    with pytest.raises(ValueError, match=reason):
        main(["band", path, *ends(tmp_path, reactant, product), "--out", str(tmp_path / "out")])

    assert not (tmp_path / "out").exists()


def test_band_overwrites_no_band_and_stays_out_of_a_trajectory_folder(tmp_path, capsys) -> None:
    path = campaign(tmp_path, "test_run:PairDoubleWell", run_config().to_dict())
    structures = ends(tmp_path, pair(0.6), pair(1.6))
    out = tmp_path / "out"
    assert band(capsys, path, *structures, "--out", out)["status"] == "saddle_validated"
    before = files(out)

    # A second band in the folder, here at a pressure, would pass for the first or replace it.
    with pytest.raises(FileExistsError, match="already holds a band"):
        main(["band", path, *structures, "--out", str(out), "--pressure", "5"])
    assert files(out) == before

    # A trajectory's folder has one writer, its run, and status would count the band as its event.
    # The folder of a campaign trajectory that has not started is one too.
    root = tmp_path / "runs/pair"
    run_pair(root)
    before = files(root)
    for folder in (root, tmp_path / "runs/only"):
        with pytest.raises(ValueError, match="trajectory's folder"):
            main(["band", path, *structures, "--out", str(folder)])
    assert files(root) == before and not (tmp_path / "runs/only").exists()


class EndsWithTheJob(PairDoubleWell):
    """Stops like a job at its time limit while ``ended`` is set."""

    ended = False

    def calculate(self, *args, **kwargs):
        if EndsWithTheJob.ended:
            raise KeyboardInterrupt("the job ended")
        super().calculate(*args, **kwargs)


def test_interrupted_band_starts_over_in_its_folder_for_the_same_ends_only(
    tmp_path, capsys, monkeypatch
) -> None:
    path = campaign(tmp_path, f"{__name__}:EndsWithTheJob", run_config().to_dict())
    structures = ends(tmp_path, pair(0.6), pair(1.6))
    out = tmp_path / "out"
    monkeypatch.setattr(EndsWithTheJob, "ended", True)
    with pytest.raises(KeyboardInterrupt, match="the job ended"):
        main(["band", path, *structures, "--out", str(out)])
    assert not (out / "pathways/band").exists()
    monkeypatch.setattr(EndsWithTheJob, "ended", False)

    # The folder already holds the ends of its band, so another product is not refined into it.
    write(tmp_path / "other.extxyz", pair(1.7))
    with pytest.raises(ValueError, match="conflicting data"):
        main(["band", path, structures[0], str(tmp_path / "other.extxyz"), "--out", str(out)])

    printed = band(capsys, path, *structures, "--out", out)
    assert printed["status"] == "saddle_validated"
    assert printed["barrier_eV"] == pytest.approx(0.0625, abs=2e-4)


def test_band_says_when_a_file_holds_several_frames_or_constraints(tmp_path, capsys) -> None:
    path = campaign(tmp_path, "test_run:PairDoubleWell", run_config().to_dict())
    held = pair(0.6)
    held.set_constraint(FixAtoms([0, 1]))
    write(tmp_path / "reactant.traj", [pair(1.6), held])
    write(tmp_path / "product.traj", [pair(1.6), pair(0.6)])

    arguments = [path, str(tmp_path / "reactant.traj"), f"{tmp_path}/product.traj@0"]
    assert main(["band", *arguments, "--out", str(tmp_path / "out")]) == 0

    captured = capsys.readouterr()
    assert [line for line in captured.err.splitlines() if line.startswith("warning: ")] == [
        "warning: reactant.traj holds several frames; band takes the last one",
        "warning: reactant.traj has constraints, which refinement removes, so the barrier is "
        "for a system with every atom free",
    ]
    # With both atoms held no image could move. The barrier is that of the free pair.
    printed = json.loads(captured.out)
    assert printed["status"] == "saddle_validated"
    assert printed["barrier_eV"] == pytest.approx(0.0625, abs=2e-4)

    arguments[1] += "@:"
    with pytest.raises(ValueError, match="names several frames"):
        main(["band", *arguments, "--out", str(tmp_path / "sliced")])
    assert not (tmp_path / "sliced").exists()


def test_step_whose_reactant_relaxes_partway_is_published_in_steps(tmp_path, capsys) -> None:
    # H-C-O loses H and then O. C-H at 1.25 Å is within its formation distance here but past the
    # top of its well at 1.1 Å, so the reactant relaxes to H gone, and only C-O is left to refine.
    settings = run_config().to_dict()
    settings["detector"]["pair_thresholds"] = {
        "C-H": [1.3, 1.5],
        "C-O": [0.8, 1.2],
        "H-O": [0.1, 0.2],
    }
    path = campaign(tmp_path, "test_run:ChainDoubleWell", settings)

    def chain(hydrogen: float, oxygen: float) -> Atoms:
        positions = [[10 - hydrogen, 10, 10], [10, 10, 10], [10 + oxygen, 10, 10]]
        return Atoms("HCO", positions=positions, cell=[20, 20, 20], pbc=True)

    out = tmp_path / "out"
    printed = band(capsys, path, *ends(tmp_path, chain(1.25, 0.6), chain(1.6, 1.6)), "--out", out)

    assert (printed["status"], printed["barrier_eV"]) == ("stepwise", None)
    folder = out / "pathways/band"
    first, second = (json.loads((folder / f"step-{n}/result.json").read_text()) for n in (1, 2))
    assert first["status"] == "unresolved"
    assert first["message"] == "the reactant frame relaxes partway along the event"
    assert (second["reactant_bonds"], second["product_bonds"]) == ([[1, 2]], [])
    assert second["status"] == "saddle_validated"
    assert second["barrier_eV"] == pytest.approx(0.125, abs=4e-4)

    with pytest.raises(ValueError, match="was refined in steps"):
        main(["export-electrace", str(folder), "--out", str(tmp_path / "dft")])
    assert main(["export-electrace", str(folder / "step-2"), "--out", str(tmp_path / "dft")]) == 0
    # C, H, O in the POSCAR, with the C-O bond changing.
    assert json.loads(capsys.readouterr().out)["focus_atoms"] == [0, 2]


def test_band_names_its_model_and_pressure_as_check_does(tmp_path, capsys) -> None:
    path = campaign(tmp_path, None, run_config().to_dict(), ids=("first", "second"))
    structures = ends(tmp_path, pair(0.6), pair(1.6))
    arguments = ["band", path, *structures, "--out", str(tmp_path / "out")]

    with pytest.raises(SystemExit):
        main(arguments)
    assert "--index picks the trajectory when a campaign has several" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([*arguments, "--index", "1", "--pressure", "nan"])
    assert "--pressure must be a finite number" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
