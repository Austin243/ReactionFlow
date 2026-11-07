"""Lay out one refined band as numbered VASP frame folders for ElecTrace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

from .detection import atom_ids


def _bonds(values: Any) -> frozenset[tuple[int, int]]:
    return frozenset((min(map(int, pair)), max(map(int, pair))) for pair in values)


def _changed_bonds(pathway: Path, result: dict[str, Any]) -> frozenset[tuple[int, int]]:
    """Bonds, as atom-ID pairs, that differ between the band's two ends."""

    if "step" in result:
        # A step of a stepwise event records the bonds of its own two ends.
        return _bonds(result["reactant_bonds"]) ^ _bonds(result["product_bonds"])
    if pathway.parent.name != "pathways":
        raise ValueError(f"{pathway} is not a folder under a trajectory's pathways/")
    bundle = pathway.parent.parent / "candidates" / result["occurrence_id"] / "candidate.json"
    candidate = json.loads(bundle.read_text(encoding="utf-8"))["candidate"]
    return _bonds(candidate["reactant_bonds"]) ^ _bonds(candidate["product_bonds"])


def _toml_string(value: str) -> str:
    return json.dumps(value)  # a JSON string with ASCII escapes is also a TOML basic string


def export_band(pathway: str | Path, out: str | Path) -> dict[str, Any]:
    """Write the band in ``pathway`` to ``out`` as ``00``, ``01``, ... folders of POSCARs.

    ``out`` also gets an ``electrace.toml`` for that path, with index atom identity, the band's
    pressure, and the atoms of its changed bonds as focus atoms, and a ``reactionflow.json``
    with the refinement record and the atom order. Energies stay in ``reactionflow.json``: they
    are the model's, and ElecTrace should read the energies of the calculations run on the frames.
    """

    pathway, out = Path(pathway), Path(out)
    result = json.loads((pathway / "result.json").read_text(encoding="utf-8"))
    if result.get("schema_version") != 1:
        raise ValueError(f"unsupported pathway result in {pathway}")
    images: list[Atoms] = read(pathway / "images.traj", index=":")  # type: ignore[assignment]
    if len(images) < 2:
        steps = sorted(path.name for path in pathway.glob("step-*") if path.is_dir())
        if steps:
            raise ValueError(
                f"{pathway} was refined in steps and has no band of its own; export one of "
                + ", ".join(str(pathway / step) for step in steps)
            )
        raise ValueError(f"{pathway} holds no band (status {result.get('status')!r})")

    ids = atom_ids(images[0])
    symbols = images[0].get_chemical_symbols()
    for number, image in enumerate(images):
        if atom_ids(image) != ids or image.get_chemical_symbols() != symbols:
            raise ValueError(f"image {number} of {pathway} does not keep the first image's atoms")
        if abs(np.linalg.det(image.cell.array)) < 1e-6:
            raise ValueError(f"image {number} of {pathway} has no cell, which a POSCAR needs")

    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"{out} already exists and is not an empty folder")
    out.mkdir(parents=True, exist_ok=True)

    # One stable element sort for every image, so each species is one POSCAR block and an atom
    # has the same index in every frame.
    order = sorted(range(len(symbols)), key=symbols.__getitem__)
    index_of = {ids[original]: new for new, original in enumerate(order)}
    width = max(2, len(str(len(images) - 1)))
    labels = [f"{number:0{width}d}" for number in range(len(images))]
    for label, image in zip(labels, images, strict=True):
        frame = image[order]
        frame.set_constraint()
        frame.calc = None
        (out / label).mkdir()
        write(out / label / "POSCAR", frame, format="vasp", direct=True, vasp5=True)

    changed = sorted(_changed_bonds(pathway, result))
    focus = sorted({index_of[atom_id] for bond in changed for atom_id in bond})
    pressure = result.get("pressure_GPa")
    name = f"ReactionFlow {result['occurrence_id']}"
    if "step" in result:
        name += f" step {result['step']}"
    config = [
        "# Written by reactionflow export-electrace. Frames hold only POSCARs: run the",
        "# calculations and descriptor analyses in them, then run electrace analyze.",
        "[project]",
        f"name = {_toml_string(name)}",
        'series_kind = "neb"',
        *(() if pressure is None else (f"pressure_gpa = {float(pressure)!r}",)),
        "",
        "[inputs]",
        'root = "."',
        "",
        "[matching]",
        'atom_identity = "index"',
        "",
        "[outputs]",
        "# Atoms of the bonds that differ between the band's two ends, as POSCAR indices.",
        f"focus_atoms = {focus}",
        "",
    ]
    (out / "electrace.toml").write_text("\n".join(config), encoding="utf-8")

    record = {
        "schema_version": 1,
        "source": str(pathway.resolve()),
        "frames": labels,
        "periodic": [bool(value) for value in images[0].pbc],
        "atom_ids": [ids[original] for original in order],
        "changed_bonds": {
            "atom_ids": [list(bond) for bond in changed],
            "indices": [[index_of[first], index_of[second]] for first, second in changed],
        },
        "result": result,
    }
    (out / "reactionflow.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validation = result.get("frequency_validation") or {}
    return {
        "export": str(out),
        "frames": len(labels),
        "focus_atoms": focus,
        "status": result.get("status"),
        "saddle_check": validation.get("status"),
        "barrier_eV": result.get("barrier_eV"),
        "barrier_quantity": result.get("barrier_quantity"),
        "pressure_GPa": pressure,
    }


__all__ = ["export_band"]
