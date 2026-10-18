from __future__ import annotations

import csv
import json

import pytest
from ase import Atoms
from ase.io import write

from reactionflow.cli import main


def rows(path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def hydrogen(*positions: float) -> Atoms:
    return Atoms(f"H{len(positions)}", positions=[[x, 0, 0] for x in positions])


def test_scan_reads_a_trajectory_file_and_counts_each_stage(tmp_path, capsys) -> None:
    # Two H2 molecules meet: the first approach turns back, the second forms a bond that holds.
    apart, joined = hydrogen(0, 0.7, 3, 3.7), hydrogen(0, 0.7, 1.45, 2.15)
    path = tmp_path / "md.extxyz"
    write(path, [apart, joined, apart, *[joined] * 6])
    settings = ["--pair", "H-H", "0.8", "1.2", "--persistence", "2", "--stability", "2"]

    assert main(["scan", str(path), *settings, "--every", "4", "--frame-ps", "0.5"]) == 0

    out = tmp_path / "md.extxyz-scan"
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "scan": str(out),
        "frames": 9,
        "first_frame": 0,
        "last_frame": 8,
        "reverted_crossings": 1,
        "bonds_formed": 1,
        "bonds_broken": 0,
        "candidates_resolved": 1,
        "candidates_unresolved": 0,
        "still_changing_at_end": 0,
        "net_changes_in_resolved": 1,
        "classes_in_resolved": 1,
    }
    assert rows(out / "bond_changes.csv") == [
        {
            "first_seen_frame": "3",
            "confirmed_frame": "4",
            "time_ps": "1.5",
            "event": "formed",
            "atom_a": "1",
            "atom_b": "2",
            "elements": "H-H",
            "distance_A": "0.75",
            "between_molecules": "True",
        }
    ]
    (candidate,) = rows(out / "candidates.csv")
    assert (candidate["changes"], candidate["resolved"], candidate["occurrence_id"]) == (
        "formed H1-H2",
        "True",
        "",
    )
    species = [
        (row["frame"], row["formula"], row["molecules"], row["count"])
        for row in rows(out / "species.csv")
    ]
    assert species == [("0", "H2", "1", "2"), ("4", "H4", "2", "1"), ("8", "H4", "2", "1")]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["settings"]["detector"]["pair_thresholds"] == {"H-H": [0.8, 1.2]}
    assert summary["settings"]["candidate_region_radius"] == 2


# Under whole components the break is still changing at the end, with the flicker.
@pytest.mark.parametrize(("radius", "counts"), [("2", (1, 0)), ("none", (0, 1))])
def test_scan_region_radius_decides_whether_distant_flicker_holds_back_a_break(
    tmp_path, capsys, radius, counts
) -> None:
    # A ten-atom chain: bond 0-1 breaks for good while bond 8-9, seven bonds away, flickers.
    chain = [0.7 * index for index in range(10)]
    frames = [hydrogen(*chain)]
    frames += [hydrogen(-1.5, *chain[1:9], chain[9] + 0.8 * (frame % 2)) for frame in range(1, 13)]
    path = tmp_path / "chain.extxyz"
    write(path, frames)

    settings = ["--pair", "H-H", "0.8", "1.2", "--persistence", "1", "--radius", radius]
    assert main(["scan", str(path), *settings]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert (printed["candidates_resolved"], printed["still_changing_at_end"]) == counts


def test_scan_rejects_options_that_do_not_fit_its_source(tmp_path) -> None:
    path = tmp_path / "md.extxyz"
    write(path, hydrogen(0, 0.7))
    for options in (["--index", "0"], ["--radius", "-1"], ["--pair", "H-H", "1.2", "0.8"]):
        with pytest.raises(SystemExit):
            main(["scan", str(path), *options])
