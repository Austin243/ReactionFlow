from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from ase import Atoms
from ase.io import write

from reactionflow import cli, model_setup
from reactionflow.campaign import CampaignConfig

MACE = "reactionflow.adapters.mace:create_adapter"
UMA = "reactionflow.adapters.uma:create_adapter"
ANI = "reactionflow.adapters.ani1xnr:create_adapter"
CUSTOM = "custom_adapter:create_adapter"


@pytest.fixture(autouse=True)
def _no_installed_optional_packages(monkeypatch):
    def absent(_name):
        raise model_setup.PackageNotFoundError

    monkeypatch.setattr(model_setup, "version", absent)


def _campaign(tmp_path, profiles, assignments):
    write(tmp_path / "structure.extxyz", Atoms("He"))
    source = tmp_path / "campaign.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "structure": "structure.extxyz",
                "output_root": "runs",
                "require_gpu": False,
                "adapter_profiles": profiles,
                "trajectories": [
                    {
                        "id": f"trajectory-{index}",
                        "adapter_profile": name,
                        "total_steps": 1,
                        "timestep_fs": 0.5,
                        "temperature_K": 300,
                        "seed": index,
                    }
                    for index, name in enumerate(assignments)
                ],
            }
        ),
        encoding="utf-8",
    )
    return CampaignConfig.load(source)


def _mixed_campaign(tmp_path):
    return _campaign(
        tmp_path,
        {
            "mace": {"factory": MACE, "options": {"model": "small", "device": "cpu"}},
            "same-mace": {"factory": MACE, "options": {"device": "cpu", "model": "small"}},
            "large-mace": {"factory": MACE, "options": {"model": "large"}},
            "uma": {"factory": UMA, "options": {"model": "uma-s-1p2"}},
            "ani": {"factory": ANI},
            "custom": {"factory": CUSTOM},
            "unused": {"factory": MACE, "options": {"model": "unused"}},
        },
        ["mace", "same-mace", "uma", "large-mace", "ani", "custom"],
    )


def _fake_preparation(monkeypatch, tmp_path):
    prepared = []

    def import_module(module):
        assert module in {MACE.split(":")[0], UMA.split(":")[0]}

        def prepare(options, *, download):
            assert download is True
            prepared.append((module, options))
            return {"checkpoint": tmp_path / f"{options['model']}.pt"}

        return SimpleNamespace(prepare=prepare)

    monkeypatch.setattr(model_setup, "import_module", import_module)
    return prepared


def test_prepare_deduplicates_referenced_adapters_without_creating_runs(tmp_path, monkeypatch):
    campaign = _mixed_campaign(tmp_path)
    prepared = _fake_preparation(monkeypatch, tmp_path)

    reports = model_setup.prepare_campaign(campaign)

    assert prepared == [
        ("reactionflow.adapters.mace", {"model": "small", "device": "cpu"}),
        ("reactionflow.adapters.uma", {"model": "uma-s-1p2"}),
        ("reactionflow.adapters.mace", {"model": "large"}),
    ]
    assert [item["status"] for item in reports] == [
        "prepared",
        "prepared",
        "prepared",
        "external_setup",
        "external_setup",
    ]
    assert reports[0]["files"] == {"checkpoint": str(tmp_path / "small.pt")}
    assert [item["factory"] for item in reports[-2:]] == [ANI, CUSTOM]
    assert not campaign.output_root.exists()


def test_prepare_index_selects_only_requested_adapter(tmp_path, monkeypatch, capsys):
    campaign = _mixed_campaign(tmp_path)
    prepared = _fake_preparation(monkeypatch, tmp_path)

    assert cli.main(["prepare", str(campaign.source), "--index", "2"]) == 0

    assert prepared == [("reactionflow.adapters.uma", {"model": "uma-s-1p2"})]
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "campaign": str(campaign.source),
        "models": [
            {
                "factory": UMA,
                "status": "prepared",
                "files": {"checkpoint": str(tmp_path / "uma-s-1p2.pt")},
            }
        ],
    }
    assert not campaign.output_root.exists()


def test_prepare_invalid_index_fails_before_any_setup(tmp_path, monkeypatch):
    campaign = _mixed_campaign(tmp_path)
    prepared = _fake_preparation(monkeypatch, tmp_path)

    with pytest.raises(IndexError, match="outside"):
        model_setup.prepare_campaign(campaign, index=20)

    assert prepared == []
    assert not campaign.output_root.exists()


@pytest.mark.parametrize("factory", [CUSTOM, ANI])
def test_external_setup_is_skipped_and_download_is_explicitly_unsupported(
    tmp_path, monkeypatch, factory
):
    campaign = _campaign(tmp_path, {"external": {"factory": factory}}, ["external"])

    def unexpected_import(_module):
        raise AssertionError("external adapter code must not run during preparation")

    monkeypatch.setattr(model_setup, "import_module", unexpected_import)
    assert model_setup.prepare_campaign(campaign) == [
        {
            "factory": factory,
            "status": "external_setup",
            "message": "This adapter manages its own dependencies and model files.",
        }
    ]
    with pytest.raises(ValueError, match=r"built-in model adapters.*manages its own setup"):
        cli.run_selected_trajectory(campaign, index=0, download=True)
    assert not campaign.output_root.exists()


@pytest.mark.parametrize("index", [None, 0, 2])
def test_install_uses_selected_pins_then_prepares_in_fresh_interpreter(
    tmp_path, monkeypatch, index
):
    campaign = (
        _campaign(tmp_path, {"mace": {"factory": MACE}}, ["mace", "mace"])
        if index is None
        else _mixed_campaign(tmp_path)
    )
    commands = []

    def run(command, *, check):
        assert check is False
        commands.append(command)
        return SimpleNamespace(returncode=0)

    def unexpected_import(_module):
        raise AssertionError("do not load optional backends after pip in this interpreter")

    monkeypatch.setattr(model_setup.subprocess, "run", run)
    monkeypatch.setattr(model_setup, "import_module", unexpected_import)
    arguments = ["prepare", str(campaign.source), "--install"]
    selection = [] if index is None else ["--index", str(index)]

    assert cli.main(arguments + selection) == 0

    packages = ["fairchem-core==2.23.0" if index == 2 else "mace-torch==0.3.16"]
    assert commands == [
        [sys.executable, "-m", "pip", "install", *packages],
        [sys.executable, "-m", "reactionflow.cli", "prepare", str(campaign.source), *selection],
    ]
    assert not campaign.output_root.exists()


@pytest.mark.parametrize("returncodes", [(17,), (0, 23)])
def test_install_or_fresh_process_failure_is_propagated(tmp_path, monkeypatch, returncodes):
    campaign = _mixed_campaign(tmp_path)
    commands = []

    def run(command, *, check):
        commands.append(command)
        return SimpleNamespace(returncode=returncodes[len(commands) - 1])

    monkeypatch.setattr(model_setup.subprocess, "run", run)

    assert (
        cli.main(["prepare", str(campaign.source), "--install", "--index", "2"]) == returncodes[-1]
    )
    assert len(commands) == len(returncodes)
    assert not campaign.output_root.exists()


def test_install_external_adapter_does_not_install_packages(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, {"external": {"factory": CUSTOM}}, ["external"])
    commands = []

    def run(command, *, check):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(model_setup.subprocess, "run", run)

    assert model_setup.install_and_prepare(campaign) == 0
    assert commands == [[sys.executable, "-m", "reactionflow.cli", "prepare", str(campaign.source)]]


def test_install_rejects_mace_and_uma_before_pip(tmp_path, monkeypatch):
    campaign = _mixed_campaign(tmp_path)

    def unexpected_process(*_args, **_kwargs):
        raise AssertionError("incompatible setup must be rejected before changing the environment")

    monkeypatch.setattr(model_setup.subprocess, "run", unexpected_process)

    with pytest.raises(ValueError, match=r"e3nn==0.4.4.*e3nn>=0.5.*--index"):
        model_setup.install_and_prepare(campaign)


@pytest.mark.parametrize(
    ("index", "installed", "pinned"),
    [(2, "mace-torch", "0.3.16"), (2, "torchani", "2.8.4"), (0, "fairchem-core", "2.23.0")],
)
def test_install_preserves_existing_incompatible_backend(
    tmp_path, monkeypatch, index, installed, pinned
):
    campaign = _mixed_campaign(tmp_path)

    def version(name):
        if name == installed:
            return pinned
        raise model_setup.PackageNotFoundError(name)

    def unexpected_process(*_args, **_kwargs):
        raise AssertionError("installed model environments must not be replaced")

    monkeypatch.setattr(model_setup, "version", version)
    monkeypatch.setattr(model_setup.subprocess, "run", unexpected_process)

    with pytest.raises(ValueError, match="use a fresh Python environment") as error:
        model_setup.install_and_prepare(campaign, index=index)
    assert f"{installed}=={pinned}" in str(error.value)


@pytest.mark.parametrize("download", [False, True])
def test_run_prepares_before_adapter_loading_and_trajectory_creation(
    tmp_path, monkeypatch, download
):
    campaign = _campaign(tmp_path, {"mace": {"factory": MACE}}, ["mace"])
    events = []
    adapter = SimpleNamespace(calculator=object())
    summary = object()

    def prepare(spec):
        assert spec == campaign.adapter_for(0)
        assert not campaign.output_root.exists()
        events.append("prepare")

    def load(spec, trajectory):
        assert spec == campaign.adapter_for(0)
        assert trajectory == campaign.trajectory(0)
        events.append("load")
        return adapter

    def bind(root, contract):
        events.append("bind")

    def create(root, *, config):
        events.append("create")
        return SimpleNamespace(run=run)

    def run(atoms, **kwargs):
        assert kwargs["runtime_provider"] is adapter
        assert kwargs["pathway_calculator_provider"] is adapter.calculator
        events.append("run")
        return summary

    monkeypatch.setattr(cli, "prepare_adapter", prepare)
    monkeypatch.setattr(cli, "load_mlip_adapter", load)
    monkeypatch.setattr(cli, "_bind_trajectory_contract", bind)
    monkeypatch.setattr(cli.ReactionRun, "create", create)

    assert cli.run_selected_trajectory(campaign, index=0, download=download) is summary
    assert events == (["prepare"] if download else []) + ["load", "bind", "create", "run"]


def test_failed_run_download_stops_before_adapter_loading_or_state(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, {"uma": {"factory": UMA}}, ["uma"])

    def prepare(_spec):
        raise RuntimeError("model unavailable")

    def load(*_args):
        raise AssertionError("an unavailable model must not be constructed")

    monkeypatch.setattr(cli, "prepare_adapter", prepare)
    monkeypatch.setattr(cli, "load_mlip_adapter", load)

    with pytest.raises(RuntimeError, match="model unavailable"):
        cli.run_selected_trajectory(campaign, index=0, download=True)
    assert not campaign.output_root.exists()


def test_run_command_forwards_explicit_download_flag(tmp_path, monkeypatch, capsys):
    campaign = _mixed_campaign(tmp_path)
    calls = []

    def run(campaign, *, index, download):
        calls.append((campaign.source, index, download))
        return cli.RunSummary("completed", 1, 1, 1, 0, 0)

    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setattr(cli, "run_selected_trajectory", run)

    assert cli.main(["run", str(campaign.source), "--index", "2", "--download"]) == 0
    assert calls == [(campaign.source, 2, True)]
    assert json.loads(capsys.readouterr().out)["phase"] == "completed"


@pytest.mark.parametrize("command", ["validate", "plan", "status"])
def test_read_only_commands_do_not_prepare_models(tmp_path, monkeypatch, command):
    campaign = _mixed_campaign(tmp_path)

    def unexpected_import(_module):
        raise AssertionError("read-only commands must not load optional backends")

    monkeypatch.setattr(model_setup, "import_module", unexpected_import)

    assert cli.main([command, str(campaign.source)]) == 0
    assert not campaign.output_root.exists()


def test_new_run_preflight_failure_leaves_no_trajectory_contract(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, {"mace": {"factory": MACE}}, ["mace"])

    def preflight(atoms):
        assert atoms.get_chemical_symbols() == ["He"]
        raise ValueError("unsupported model inputs")

    adapter = SimpleNamespace(calculator=object(), preflight=preflight)
    monkeypatch.setattr(cli, "load_mlip_adapter", lambda *args: adapter)
    with pytest.raises(ValueError, match="unsupported model inputs"):
        cli.run_selected_trajectory(campaign, index=0)
    assert not campaign.output_root.exists()


def test_models_command_needs_neither_campaign_nor_optional_packages(capsys):
    assert cli.main(["models", "--backend", "chgnet"]) == 0
    result = json.loads(capsys.readouterr().out)["backends"]
    assert len(result) == 1 and result[0]["backend"] == "chgnet"
    assert {model["name"] for model in result[0]["models"]} == {"0.3.0", "r2scan"}


def test_models_rejects_unknown_backend(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["models", "--backend", "typo"])
    assert error.value.code == 2
    assert "unknown backend" in capsys.readouterr().err


def test_full_catalog_does_not_import_optional_libraries(capsys, monkeypatch):
    import builtins

    original = builtins.__import__
    optional = {
        "torch",
        "mace",
        "fairchem",
        "aimnet",
        "orb_models",
        "mattersim",
        "chgnet",
        "sevenn",
        "calorine",
        "_nepy",
    }

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in optional:
            pytest.fail(f"catalog imported optional library {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert cli.main(["models"]) == 0
    backends = json.loads(capsys.readouterr().out)["backends"]
    assert {item["backend"] for item in backends} == {
        "mace",
        "mace_field",
        "uma",
        "aimnet2",
        "orb",
        "mattersim",
        "chgnet",
        "sevennet",
        "nep",
    }


@pytest.mark.parametrize("backend", ["mattersim", "sevennet"])
def test_install_rejects_incompatible_e3nn_before_pip(tmp_path, monkeypatch, backend):
    campaign = _campaign(
        tmp_path,
        {
            "mace": {"factory": MACE},
            "other": {"factory": f"reactionflow.adapters.{backend}:create_adapter"},
        },
        ["mace", "other"],
    )
    monkeypatch.setattr(
        model_setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not install")
    )
    with pytest.raises(ValueError, match="e3nn"):
        model_setup.install_and_prepare(campaign)


@pytest.mark.parametrize(
    ("backend", "package", "pinned"),
    [("uma", "fairchem-core", "2.23.0"), ("mattersim", "mattersim", "1.2.5")],
)
@pytest.mark.parametrize("selection", ["both", "orb", "other"])
def test_installer_preserves_incompatible_neighbor_runtime(
    tmp_path, monkeypatch, backend, package, pinned, selection
):
    profiles = {
        "orb": {"factory": "reactionflow.adapters.orb:create_adapter"},
        "other": {"factory": f"reactionflow.adapters.{backend}:create_adapter"},
    }
    campaign = _campaign(
        tmp_path, profiles, ["orb", "other"] if selection == "both" else [selection]
    )

    def version(name):
        if selection == "orb" and name == package:
            return pinned
        if selection == "other" and name == "orb-models":
            return "0.7.0"
        raise model_setup.PackageNotFoundError

    monkeypatch.setattr(model_setup, "version", version)
    monkeypatch.setattr(
        model_setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not install")
    )
    with pytest.raises(ValueError, match="Python environment"):
        model_setup.install_and_prepare(campaign)


def test_polar_installs_compatible_electrostatics_only_when_selected(tmp_path, monkeypatch):
    campaign = _campaign(
        tmp_path,
        {"polar": {"factory": MACE, "options": {"family": "polar", "model": "polar-1-s"}}},
        ["polar"],
    )
    commands = []
    monkeypatch.setattr(
        model_setup.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0),
    )
    assert model_setup.install_and_prepare(campaign) == 0
    assert commands[0] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "mace-torch==0.3.16",
        "graph_longrange @ git+https://github.com/WillBaldwin0/graph_electrostatics.git@0e21d5546c482d08388a08eb4d948e833227ce47",
    ]


@pytest.mark.parametrize(
    ("backend", "packages"),
    [("sevennet", ["sevenn==0.13.0", "torch>=2.8,<3"]), ("nep", ["calorine==4.0"])],
)
def test_selected_setup_includes_backend_runtime_dependencies(
    tmp_path, monkeypatch, backend, packages
):
    campaign = _campaign(
        tmp_path,
        {"chosen": {"factory": f"reactionflow.adapters.{backend}:create_adapter"}},
        ["chosen"],
    )
    commands = []
    monkeypatch.setattr(
        model_setup.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0),
    )
    assert model_setup.install_and_prepare(campaign) == 0
    assert commands[0] == [sys.executable, "-m", "pip", "install", *packages]
    assert commands[1] == [
        sys.executable,
        "-m",
        "reactionflow.cli",
        "prepare",
        str(campaign.source),
    ]


@pytest.mark.parametrize(
    "other",
    [
        MACE,
        UMA,
        "reactionflow.adapters.mattersim:create_adapter",
        "reactionflow.adapters.sevennet:create_adapter",
    ],
)
def test_field_requires_compatible_environment_before_install(tmp_path, monkeypatch, other):
    field = "reactionflow.adapters.mace_field:create_adapter"
    campaign = _campaign(
        tmp_path, {"field": {"factory": field}, "other": {"factory": other}}, ["field", "other"]
    )
    monkeypatch.setattr(
        model_setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not install")
    )
    with pytest.raises(ValueError, match="separate"):
        model_setup.install_and_prepare(campaign)


@pytest.mark.parametrize("select_field", [False, True])
def test_installer_cannot_overwrite_other_mace_distribution(tmp_path, monkeypatch, select_field):
    field = "reactionflow.adapters.mace_field:create_adapter"
    factory = field if select_field else MACE
    campaign = _campaign(tmp_path, {"chosen": {"factory": factory}}, ["chosen"])

    def installed(name):
        if name == "mace-torch":
            return "0.3.16" if select_field else "0.3.15"
        raise model_setup.PackageNotFoundError

    info = {} if select_field else {"vcs_info": {"commit_id": model_setup._FIELD_COMMIT}}
    monkeypatch.setattr(model_setup, "version", installed)
    monkeypatch.setattr(
        model_setup,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda path: json.dumps(info)),
    )
    monkeypatch.setattr(
        model_setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not install")
    )
    with pytest.raises(ValueError, match="fresh Python environment"):
        model_setup.install_and_prepare(campaign)
