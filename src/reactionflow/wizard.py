"""Write a campaign file from answers to a few questions (`reactionflow init`).

A campaign is built from groups. Each group runs every combination of its starting structures,
models, temperatures, and pressures, a chosen number of times, so a campaign of hundreds of
trajectories still takes only a handful of answers.
"""

from __future__ import annotations

import glob
import itertools
import json
import math
import os
import re
import secrets
import shlex
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read

from ._durable import publish
from .campaign import CampaignConfig
from .model_setup import backend_conflict, model_catalog
from .run import ReactionRunConfig

# Backends whose catalog notes require a fully periodic cell.
_PERIODIC_ONLY = {"mattersim", "chgnet", "nep"}
# A catalog list field that a model requires one value from, and the option it sets.
_CHOICES = {"heads": "head", "tasks": "task", "modals": "modal"}
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class _Quit(Exception):
    """The session ended before the file was written."""


def _ask(question: str, default: str | None = None) -> str:
    """Return one answer. Enter takes the default; a None default means an answer is required."""

    shown = f" [{default}]" if default else ""
    while True:
        try:
            answer = input(f"{question}{shown}: ").strip()
        except EOFError:
            raise _Quit from None
        if answer or default is not None:
            return answer or default


def _ask_until_valid(question: str, parse: Callable[[str], Any], default: str | None = None):
    while True:
        try:
            return parse(_ask(question, default))
        except ValueError as error:
            print(f"  {error}")


def _yes(question: str, default: bool) -> bool:
    def parse(text: str) -> bool:
        if text.lower() not in ("y", "yes", "n", "no"):
            raise ValueError("answer y or n")
        return text.lower().startswith("y")

    return _ask_until_valid(question, parse, "y" if default else "n")


def _positive(kind: type = int, maximum: float = math.inf) -> Callable[[str], Any]:
    def parse(text: str):
        try:
            value = kind(text)
        except ValueError:
            raise ValueError(f"{text!r} is not a {'whole ' * (kind is int)}number") from None
        if not 0 < value <= maximum:
            raise ValueError(f"use a positive number up to {maximum:g}")
        return value

    return parse


def _numbers(text: str, *, minimum: float | None = None, allow_none: bool = False) -> list:
    """Values separated by spaces or commas; start:stop:step is an inclusive range."""

    values: list[float | None] = []
    for token in text.replace(",", " ").split():
        if allow_none and token.lower() in ("none", "nvt"):
            values.append(None)
            continue
        try:
            parts = [float(part) for part in token.split(":")]
        except ValueError:
            raise ValueError(f"{token!r} is not a number") from None
        if len(parts) == 1:
            values.extend(parts)
        elif len(parts) == 3 and parts[2] > 0 and parts[1] >= parts[0]:
            start, stop, step = parts
            count = math.floor((stop - start) / step + 1e-9) + 1
            values.extend(round(start + i * step, 6) for i in range(count))
        else:
            raise ValueError(f"{token!r}: write one value or start:stop:step")
    if not values:
        raise ValueError("enter at least one value")
    numbers = [value for value in values if value is not None]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("values must be finite")
    if minimum is not None and any(value < minimum for value in numbers):
        raise ValueError(f"values must be at least {minimum:g}")
    return list(dict.fromkeys(values))


def _count(number: int, word: str) -> str:
    plural = word[:-1] + "ies" if word.endswith("y") else word + "s"
    return f"{number} {word if number == 1 else plural}"


def _listed(values: list) -> str:
    return " ".join("none" if value is None else f"{value:g}" for value in values)


def _safe(text: str) -> str:
    return _UNSAFE.sub("-", text).strip("-._")


def _shown(path: Path) -> str:
    with suppress(ValueError):
        return str(path.relative_to(Path.cwd()))
    return str(path)


def _menu(
    title: str,
    labels: list[str],
    *,
    prompt: str = "Number",
    many: bool = False,
    default: str | None = None,
):
    print(title)
    for number, label in enumerate(labels, 1):
        print(f"  {number:>2}  {label}")

    def parse(text: str) -> list[int]:
        if not text and default == "":
            return []
        try:
            picks = [int(token) - 1 for token in text.replace(",", " ").split()]
        except ValueError:
            raise ValueError("enter the number of a choice") from None
        if not picks or not all(0 <= pick < len(labels) for pick in picks):
            raise ValueError(f"choose from 1 to {len(labels)}")
        if len(picks) > 1 and not many:
            raise ValueError("choose one")
        return picks

    picks = _ask_until_valid(prompt, parse, default)
    return picks if many else picks[0]


@dataclass(frozen=True)
class _Structure:
    path: Path
    atoms: Atoms = field(compare=False)

    @property
    def periodic(self) -> bool:
        return bool(self.atoms.pbc.all()) and self.atoms.cell.rank == 3


def _structures(text: str) -> list[_Structure]:
    found: dict[Path, _Structure] = {}
    for token in shlex.split(text):
        pattern = os.path.expanduser(token)
        names = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not names:
            raise ValueError(f"no file matches {token}")
        for name in names:
            path = Path(name).resolve()
            if not path.is_file():
                raise ValueError(f"{name} is not a file")
            try:
                atoms = read(path)
            except Exception as error:
                raise ValueError(f"cannot read {name}: {error}") from None
            found.setdefault(path, _Structure(path, atoms))
    if not found:
        raise ValueError("enter at least one structure file")
    for structure in found.values():
        atoms, cell = structure.atoms, "periodic" if structure.periodic else "not periodic"
        formula = atoms.get_chemical_formula()
        print(f"  {_shown(structure.path)}: {len(atoms)} atoms, {formula}, {cell}")
    return list(found.values())


@dataclass
class _Profile:
    name: str
    factory: str
    options: dict[str, Any]
    backend: str | None = None

    @property
    def key(self) -> str:
        return json.dumps([self.factory, self.options], sort_keys=True)


def _factory(text: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", text):
        raise ValueError("write it as module:callable, for example my_package.calculators:Model")
    return text


def _keywords(text: str) -> dict[str, Any]:
    keywords = {}
    for token in shlex.split(text):
        key, separator, value = token.partition("=")
        if not separator or not key:
            raise ValueError(f"{token!r}: write key=value")
        try:
            keywords[key] = json.loads(value)
        except ValueError:
            keywords[key] = value
    return keywords


def _absolute_paths(text: str) -> list[str]:
    paths = [os.path.expanduser(token) for token in shlex.split(text)]
    relative = [path for path in paths if not os.path.isabs(path)]
    if relative:
        raise ValueError(f"use absolute paths: {' '.join(relative)}")
    return paths


def _field(text: str) -> list[float]:
    try:
        values = [float(token) for token in text.replace(",", " ").split()]
    except ValueError:
        values = []
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("write three numbers: Ex Ey Ez")
    return values


def _builtin(backend: dict[str, Any]) -> _Profile:
    """Turn one catalog model, and any head, task, or modal it needs, into adapter options."""

    name = backend["backend"]
    models = backend["models"]
    if len(models) == 1:
        entry = models[0]
    else:
        labels = [
            " ".join(filter(None, (m["name"], m.get("family"), m.get("description"))))
            for m in models
        ]
        entry = models[_menu(f"{name} models", labels)]
    options: dict[str, Any] = {} if name == "ani1xnr" else {"model": entry["name"]}
    if "family" in entry:
        options["family"] = entry["family"]
    for choices, option in _CHOICES.items():
        values = entry.get(choices) or []
        if values:
            described = backend.get(choices) if isinstance(backend.get(choices), dict) else {}
            labels = [f"{value:<16} {described.get(value, '')}".rstrip() for value in values]
            options[option] = values[_menu(f"{option.capitalize()} for {entry['name']}", labels)]
    if name == "mace" and entry["name"] == "mh-0":
        options["head"] = _ask("Head name in the MACE-MH-0 checkpoint")
    if name == "mace_field":
        options["electric_field"] = _ask_until_valid("Electric field Ex Ey Ez in V/Å", _field)
    model = str(options.get("model", name))
    parts = [] if name.replace("_", "") in model.lower().replace("-", "") else [name]
    if options.get("family", "mp") != "mp" and options["family"] not in model:
        parts.append(options["family"])
    parts += [model, *(options[key] for key in ("head", "task", "modal") if key in options)]
    return _Profile(_safe("-".join(parts)), backend["factory"], options, name)


def _ase_calculator() -> _Profile:
    """A profile for the generic adapter around any ASE calculator."""

    factory = _ask_until_valid(
        "Calculator factory, module:callable (for example ase.calculators.emt:EMT)", _factory
    )
    options: dict[str, Any] = {"calculator_factory": factory}
    keywords = _ask_until_valid("Calculator arguments as key=value (Enter for none)", _keywords, "")
    files = _ask_until_valid(
        "Model files it loads, as absolute paths (Enter for none)", _absolute_paths, ""
    )
    if keywords:
        options["calculator_kwargs"] = keywords
    if files:
        options["model_files"] = files
    default = _safe(factory.rsplit(":", 1)[1].lower()) or "model"
    name = _ask_until_valid("Name for this model in the campaign", _profile_name, default)
    return _Profile(name, "reactionflow.adapters.ase:create_adapter", options)


def _profile_name(text: str) -> str:
    if _safe(text) != text:
        raise ValueError("use letters, digits, '.', '_' or '-', starting with a letter or digit")
    return text


def _needs_charge_and_spin(profile: _Profile) -> bool:
    options = profile.options
    return (
        profile.backend == "aimnet2"
        or (profile.backend == "uma" and options.get("task") == "omol")
        or (profile.backend == "orb" and options.get("model") == "orbmol-v2")
        or (profile.backend == "mace" and options.get("family") == "polar")
    )


def _models(catalog: list[dict], previous: list[_Profile] | None) -> list[_Profile]:
    labels = [
        f"{backend['backend']:<11} {len(backend['models'])} "
        f"model{'s' if len(backend['models']) > 1 else ''}"
        for backend in catalog
    ]
    labels.append("another model with an ASE calculator")
    hint = " (Enter: same as the previous group)" if previous else ""
    picks = _menu(
        f"Models; list several numbers to run each one{hint}",
        labels,
        prompt="Model numbers",
        many=True,
        default="" if previous else None,
    )
    if not picks:
        return list(previous or [])
    return [_builtin(catalog[pick]) if pick < len(catalog) else _ase_calculator() for pick in picks]


@dataclass
class _Group:
    structures: list[_Structure]
    profiles: list[_Profile]
    temperatures: list[float]
    pressures: list[float | None]
    hydrostatic: bool
    runs: int


def _group(number: int, catalog: list[dict], previous: _Group | None, used: list[_Profile]):
    print(f"\nGroup {number}")
    same = " (Enter: same as the previous group)" if previous else ""
    structures = _ask_until_valid(
        f"Starting structure files; wildcards work{same}",
        lambda text: _structures(text) if text or not previous else previous.structures,
        "" if previous else None,
    )
    while True:
        profiles = _models(catalog, previous.profiles if previous else None)
        conflict = backend_conflict(profile.factory for profile in used + profiles)
        flat = [s for s in structures if not s.periodic]
        periodic_only = [p.name for p in profiles if p.backend in _PERIODIC_ONLY]
        if conflict:
            print(f"  These models cannot share one Python environment: {conflict}.")
            print("  Choose others, or write a separate campaign for them.")
        elif flat and periodic_only:
            print(
                f"  {', '.join(periodic_only)} need a periodic cell, but "
                f"{', '.join(_shown(s.path) for s in flat)} is not periodic."
            )
        else:
            break
    for profile in profiles:
        missing = [s for s in structures if not {"charge", "spin"} <= set(s.atoms.info)]
        if _needs_charge_and_spin(profile) and missing:
            print(
                f"  Note: {profile.name} reads the total charge and spin multiplicity from the "
                f"structure, and {', '.join(_shown(s.path) for s in missing)} has none. In an "
                "extxyz file, add charge=0 spin=1 (your values) to the comment line."
            )
    temperatures = _ask_until_valid(
        "Temperatures in K, as a list or start:stop:step",
        lambda text: _numbers(text, minimum=0),
        _listed(previous.temperatures) if previous else "300",
    )
    while True:
        pressures = _ask_until_valid(
            "Pressures in GPa, or none for constant volume",
            lambda text: _numbers(text, allow_none=True),
            _listed(previous.pressures) if previous else "none",
        )
        flat = [s for s in structures if not s.periodic]
        if not flat or pressures == [None]:
            break
        print(
            f"  Constant pressure needs a periodic cell, but "
            f"{', '.join(_shown(s.path) for s in flat)} is not periodic."
        )
    hydrostatic = True
    if any(pressure is not None for pressure in pressures):
        hydrostatic = (
            _menu(
                "Under pressure, the cell",
                ["keeps its shape and scales uniformly", "may also change shape"],
                default="1" if not previous or previous.hydrostatic else "2",
            )
            == 0
        )
    combinations = len(structures) * len(profiles) * len(temperatures) * len(pressures)
    runs = _ask_until_valid(
        f"Runs of each of these {combinations} combinations",
        _positive(),
        str(previous.runs) if previous else "1",
    )
    factors = [
        (structures, "structure"),
        (profiles, "model"),
        (temperatures, "temperature"),
        (pressures, "pressure"),
        (range(runs), "run"),
    ]
    print(
        "  "
        + " x ".join(_count(len(items), word) for items, word in factors)
        + f" = {_count(combinations * runs, 'trajectory')}"
    )
    return _Group(structures, profiles, temperatures, pressures, hydrostatic, runs)


def _labels(paths: list[Path]) -> dict[Path, str]:
    """Short unique names for structure files, used in trajectory IDs."""

    labels = {path: _safe(path.stem) for path in paths}
    clashing = Counter(labels.values())
    for path in paths:
        if clashing[labels[path]] > 1 or not labels[path]:
            labels[path] = _safe(f"{path.parent.name}-{path.stem}")
    if all(labels.values()) and len(set(labels.values())) == len(labels):
        return labels
    return {path: f"s{number}" for number, path in enumerate(paths, 1)}


def _campaign(path: Path, groups: list[_Group], settings: dict[str, Any]) -> dict[str, Any]:
    profiles: dict[str, _Profile] = {}
    for group in groups:
        for profile in group.profiles:
            profiles.setdefault(profile.key, profile)
    names: dict[str, str] = {}
    for key, profile in profiles.items():
        name, number = profile.name, 2
        while name in names.values():
            name, number = f"{profile.name}-{number}", number + 1
        names[key] = name
    structures = list(dict.fromkeys(s.path for group in groups for s in group.structures))
    labels = _labels(structures)
    several_structures, several_models = len(structures) > 1, len(profiles) > 1

    def relative(structure: Path) -> str:
        return os.path.relpath(structure, path.parent)

    stems, trajectories = [], []
    seed = secrets.randbelow(2**32)
    for group in groups:
        combinations = itertools.product(
            group.structures, group.profiles, group.temperatures, group.pressures
        )
        for structure, profile, temperature, pressure in combinations:
            anisotropic = pressure is not None and not group.hydrostatic
            parts = [labels[structure.path]] if several_structures else []
            if several_models:
                parts.append(names[profile.key])
            whole = temperature == int(temperature)
            parts.append(f"{int(temperature):04d}K" if whole else f"{temperature:g}K")
            parts.append("nvt" if pressure is None else f"{pressure:g}GPa")
            if anisotropic:
                parts.append("aniso")
            for _ in range(group.runs):
                trajectory = {"adapter_profile": names[profile.key]}
                if several_structures:
                    trajectory["structure"] = relative(structure.path)
                trajectory |= {
                    "total_steps": settings["steps"],
                    "timestep_fs": settings["timestep"],
                    "temperature_K": temperature,
                    "pressure_GPa": pressure,
                    "seed": seed + len(trajectories),
                }
                if anisotropic:
                    trajectory["conditions"] = {"hydrostatic": False}
                stems.append(_safe("-".join(parts)))
                trajectories.append(trajectory)
    # Number the runs of each combination; a combination repeated in a later group continues
    # the count, so IDs stay unique.
    width = max(2, len(str(max(Counter(stems).values()))))
    counted: Counter[str] = Counter()
    for index, stem in enumerate(stems):
        counted[stem] += 1
        trajectories[index] = {"id": f"{stem}-{counted[stem]:0{width}d}", **trajectories[index]}

    adapter_profiles = {}
    for key, profile in profiles.items():
        options = dict(profile.options)
        if profile.backend and profile.backend != "nep":
            options["device"] = "cuda" if settings["gpu"] else "cpu"
        adapter_profiles[names[key]] = {"factory": profile.factory, "options": options}
    campaign: dict[str, Any] = {"schema_version": 2}
    if not several_structures:
        campaign["structure"] = relative(structures[0])
    return campaign | {
        "output_root": settings["output"],
        "require_gpu": settings["gpu"],
        "adapter_profiles": adapter_profiles,
        "reaction_run": {"observation_interval": settings["interval"]},
        "trajectories": trajectories,
    }


def _write(path: Path, campaign: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
    try:
        CampaignConfig.load(temporary)  # the same checks every other command applies
        publish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _session(path: Path) -> int:
    print(
        "ReactionFlow campaign setup. Enter takes the value in brackets; "
        "Ctrl-C stops without writing anything."
    )
    if path.exists() and not _yes(f"{_shown(path)} exists. Replace it?", False):
        print("Nothing was written.")
        return 1
    catalog = model_catalog()
    groups: list[_Group] = []
    while True:
        used = [profile for group in groups for profile in group.profiles]
        group = _group(len(groups) + 1, catalog, groups[-1] if groups else None, used)
        if not _yes("Keep this group?", True):
            continue
        groups.append(group)
        if not _yes("Add another group, for example a different model or structure?", False):
            break

    print()
    steps = _ask_until_valid("MD steps per trajectory", _positive(), "100000")
    timestep = _ask_until_valid("Time step in fs", _positive(float), "1")
    print(f"  {steps} steps of {timestep:g} fs = {steps * timestep / 1000:g} ps per trajectory")

    default_interval = str(ReactionRunConfig().observation_interval)
    every = _ask_until_valid(
        "Check bonds every how many MD steps", _positive(int, steps), default_interval
    )
    profiles = [profile for group in groups for profile in group.profiles]
    if all(profile.backend == "nep" for profile in profiles):
        gpu = False
        print("  NEP89 runs on CPUs, so the campaign will not ask for GPUs.")
    else:
        gpu = _yes("Run each trajectory on its own GPU?", True)
    output = _ask_until_valid(
        "Output directory, relative to the campaign file", _relative_directory, "runs"
    )
    settings = {
        "steps": steps,
        "timestep": timestep,
        "interval": every,
        "gpu": gpu,
        "output": output,
    }
    campaign = _campaign(path, groups, settings)

    trajectories = campaign["trajectories"]
    print(f"\n{_count(len(trajectories), 'trajectory')}")
    for key, label in (("adapter_profile", "per model"), ("structure", "per structure")):
        counts = Counter(trajectory[key] for trajectory in trajectories if key in trajectory)
        if len(counts) > 1:
            print(f"  {label}: " + ", ".join(f"{name} {count}" for name, count in counts.items()))
    last = f" ... {trajectories[-1]['id']}" if len(trajectories) > 1 else ""
    print(f"  IDs: {trajectories[0]['id']}{last}")
    if not _yes(f"Write {_shown(path)}?", True):
        print("Nothing was written.")
        return 1
    _write(path, campaign)
    shown = shlex.quote(_shown(path))
    print(
        f"Wrote {shown}. Next:\n"
        f"  reactionflow validate {shown}\n"
        f"  reactionflow prepare {shown} --install\n"
        f"  reactionflow run {shown} --index 0     runs the first trajectory here\n"
        f"  srun reactionflow run {shown}          under Slurm, with --ntasks={len(trajectories)}"
    )
    return 0


def _relative_directory(text: str) -> str:
    if os.path.isabs(text) or not text:
        raise ValueError("give a directory relative to the campaign file")
    return text


def create_campaign(path: Path) -> int:
    """Ask about the runs, then write a validated campaign file to `path`."""

    with suppress(ImportError):
        import readline  # noqa: F401  (line editing and history in input())
    try:
        return _session(Path(path).resolve())
    except (KeyboardInterrupt, _Quit):
        print("\nStopped. Nothing was written.")
        return 1


__all__ = ["create_campaign"]
