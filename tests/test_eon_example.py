"""Opt-in CLI run of the shipped AKMC example with real pyeonclient.

REACTIONFLOW_TEST_EON=1 requires the installed backend; missing dependencies fail.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from reactionflow.search.records import SearchStore


def test_real_eon_example_steps_between_the_two_wells_and_resumes(tmp_path):
    if os.environ.get("REACTIONFLOW_TEST_EON") != "1":
        pytest.skip("set REACTIONFLOW_TEST_EON=1 with reactionflow[eon] installed")

    example = Path(__file__).resolve().parents[1] / "examples" / "eon"
    isolated = tmp_path / "example"
    isolated.mkdir()
    for name in ("campaign.json", "start.extxyz", "toy.py"):
        shutil.copyfile(example / name, isolated / name)
    campaign = isolated / "campaign.json"
    settings = json.loads(campaign.read_text())
    output = tmp_path / "output"
    settings["output_root"] = str(output)
    campaign.write_text(json.dumps(settings))

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(isolated), environment.get("PYTHONPATH", "")))
    )
    for name in ("SLURM_NTASKS", "SLURM_PROCID"):
        environment.pop(name, None)

    def command(action, *arguments):
        completed = subprocess.run(
            [sys.executable, "-m", "reactionflow.cli", action, str(campaign), *arguments],
            cwd=isolated,
            env=environment,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
        return completed.stdout

    validated = json.loads(command("validate"))
    assert validated["mode"] == "eon" and validated["atoms"] == 2
    # EON's own log goes to eon.log, so the run prints only its JSON summary.
    json.loads(command("run"))
    assert (output / "eon.log").stat().st_size > 0
    first = json.loads(command("status", "--json"))
    assert first["status"] == "stopped" and first["stop_reason"] == "steps_completed"
    assert first["steps"] == 4 and first["states"] == 2
    rows = first["step_rows"]
    assert [(row["from"], row["to"]) for row in rows] == [
        ("state-000000", "state-000001"),
        ("state-000001", "state-000000"),
    ] * 2
    assert rows[0]["barrier_eV"] == pytest.approx(0.5461333, abs=2e-3)
    assert rows[1]["barrier_eV"] == pytest.approx(1.6128, abs=2e-3)
    assert all(row["bonds"] == "no bond change" for row in rows)
    # A 76 µs step after a step of ~10¹³ s does not change the clock in double precision.
    assert all(row["dt_s"] > 0 for row in rows)
    assert [row["time_s"] for row in rows] == sorted(row["time_s"] for row in rows)

    store = SearchStore(output)
    forward = store.read("processes", rows[0]["process"]).data
    frequency = math.sqrt(6.4 / 4.002602 * 9.648533e27) / (2 * math.pi)
    assert forward["prefactor_s"] == pytest.approx(frequency, rel=0.02)

    def saved():
        return {
            str(path.relative_to(output)): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file() and not path.name.startswith(".") and path.name != "eon.log"
        }

    before = saved()
    command("run")
    assert json.loads(command("status", "--json")) == first
    assert saved() == before
