"""Minimal durable storage for reaction-candidate occurrences."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from ase import Atoms
from ase.io import read, write

from .candidates import ReactionCandidate, same_reaction
from .detection import BondDetectorConfig, assign_atom_ids, atom_ids


@dataclass(frozen=True, slots=True)
class OccurrenceRecord:
    """Persistent assignment for one retained candidate occurrence."""

    occurrence_id: str
    class_id: str
    is_representative: bool
    directory: Path


def _canonical_bonds(values: frozenset[tuple[int, int]]) -> list[list[int]]:
    return [list(bond) for bond in sorted((min(bond), max(bond)) for bond in values)]


def _structure_digest(atoms: Atoms) -> str:
    ids = atom_ids(atoms)
    order = sorted(range(len(ids)), key=ids.__getitem__)
    data = {
        "atom_ids": [ids[index] for index in order],
        "symbols": [atoms[index].symbol for index in order],
        "positions": [atoms.positions[index].tolist() for index in order],
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
    }
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _candidate_data(candidate: ReactionCandidate) -> dict[str, object]:
    return {
        "atom_ids": sorted(candidate.atom_ids),
        "reactant_bonds": _canonical_bonds(candidate.reactant_bonds),
        "product_bonds": _canonical_bonds(candidate.product_bonds),
        "reactant_frame": candidate.reactant_frame,
        "product_frame": candidate.product_frame,
        "observed_frame": candidate.observed_frame,
        "resolved": candidate.resolved,
        "reactant_sha256": _structure_digest(candidate.reactant),
        "product_sha256": _structure_digest(candidate.product),
    }


def _write_endpoint(path: Path, atoms: Atoms) -> None:
    snapshot = atoms.copy()
    snapshot.info["atom_ids"] = list(atom_ids(snapshot))
    write(path, snapshot)


def _read_bundle(directory: Path) -> tuple[dict[str, object], ReactionCandidate]:
    metadata = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"unsupported candidate bundle in {directory}")
    data = metadata["candidate"]
    reactant = assign_atom_ids(read(directory / "reactant.traj"))
    product = assign_atom_ids(read(directory / "product.traj"))
    if data["reactant_sha256"] != _structure_digest(reactant) or data[
        "product_sha256"
    ] != _structure_digest(product):
        raise ValueError(f"candidate bundle endpoints conflict in {directory}")
    return metadata, ReactionCandidate(
        reactant=reactant,
        product=product,
        atom_ids=tuple(map(int, data["atom_ids"])),
        reactant_bonds=frozenset(tuple(map(int, bond)) for bond in data["reactant_bonds"]),
        product_bonds=frozenset(tuple(map(int, bond)) for bond in data["product_bonds"]),
        reactant_frame=int(data["reactant_frame"]),
        product_frame=int(data["product_frame"]),
        observed_frame=int(data["observed_frame"]),
        resolved=bool(data["resolved"]),
    )


class OccurrenceStore:
    """Single-writer SQLite registry backed by immutable candidate directories."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.database = self.root / "reactions.sqlite3"
        self.candidates = self.root / "candidates"
        self.candidates.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"unsupported occurrence database version {version}")
            db.execute(
                """CREATE TABLE IF NOT EXISTS occurrences (
                    sequence INTEGER PRIMARY KEY,
                    occurrence_id TEXT UNIQUE NOT NULL,
                    class_id TEXT NOT NULL,
                    representative INTEGER NOT NULL CHECK (representative IN (0, 1)),
                    bundle TEXT UNIQUE NOT NULL
                )"""
            )
            db.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS one_representative_per_class
                   ON occurrences(class_id) WHERE representative = 1"""
            )
            if version == 0:
                db.execute("PRAGMA user_version = 1")
        self._recover()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _record(self, row: sqlite3.Row) -> OccurrenceRecord:
        return OccurrenceRecord(
            occurrence_id=row["occurrence_id"],
            class_id=row["class_id"],
            is_representative=bool(row["representative"]),
            directory=self.root / row["bundle"],
        )

    def _insert(
        self,
        db: sqlite3.Connection,
        occurrence_id: str,
        bundle: str,
        candidate: ReactionCandidate,
    ) -> tuple[OccurrenceRecord, bool]:
        existing = db.execute(
            "SELECT * FROM occurrences WHERE occurrence_id = ?", (occurrence_id,)
        ).fetchone()
        if existing is not None:
            return self._record(existing), False

        class_id = ""
        seen_classes: set[str] = set()
        for row in db.execute("SELECT * FROM occurrences ORDER BY sequence"):
            if row["class_id"] in seen_classes:
                continue
            seen_classes.add(row["class_id"])
            stored = _read_bundle(self.root / row["bundle"])[1]
            if same_reaction(candidate, stored):
                class_id = row["class_id"]
                break
        class_id = class_id or f"reaction-{uuid4().hex}"
        has_representative = db.execute(
            "SELECT 1 FROM occurrences WHERE class_id = ? AND representative = 1", (class_id,)
        ).fetchone()
        representative = candidate.resolved and has_representative is None
        db.execute(
            """INSERT INTO occurrences
               (occurrence_id, class_id, representative, bundle)
               VALUES (?, ?, ?, ?)""",
            (occurrence_id, class_id, int(representative), bundle),
        )
        row = db.execute(
            "SELECT * FROM occurrences WHERE occurrence_id = ?", (occurrence_id,)
        ).fetchone()
        assert row is not None
        return self._record(row), True

    def _recover(self) -> None:
        bundles: list[tuple[str, str, ReactionCandidate]] = []
        for directory in sorted(self.candidates.iterdir()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            metadata, candidate = _read_bundle(directory)
            occurrence_id = metadata.get("occurrence_id")
            if occurrence_id != directory.name:
                raise ValueError(f"candidate bundle ID conflicts with {directory}")
            bundles.append((occurrence_id, str(Path("candidates") / occurrence_id), candidate))

        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT bundle FROM occurrences"):
                if not (self.root / row["bundle"]).is_dir():
                    raise FileNotFoundError(self.root / row["bundle"])
            for occurrence_id, bundle, candidate in bundles:
                self._insert(db, occurrence_id, bundle, candidate)
            db.commit()

    def _publish(
        self,
        occurrence_id: str,
        candidate: ReactionCandidate,
        metadata: dict[str, object],
    ) -> ReactionCandidate:
        final = self.candidates / occurrence_id
        if final.exists():
            stored_metadata, stored_candidate = _read_bundle(final)
            if stored_metadata != metadata:
                raise ValueError(f"occurrence ID {occurrence_id!r} has conflicting data")
            return stored_candidate

        temporary = self.candidates / f".{occurrence_id}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        try:
            _write_endpoint(temporary / "reactant.traj", candidate.reactant)
            _write_endpoint(temporary / "product.traj", candidate.product)
            (temporary / "candidate.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return _read_bundle(final)[1]

    def register(
        self,
        occurrence_id: str,
        candidate: ReactionCandidate,
        *,
        detector_config: BondDetectorConfig,
    ) -> tuple[OccurrenceRecord, bool]:
        """Retain an occurrence and return its assignment plus whether it was inserted."""

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", occurrence_id):
            raise ValueError("occurrence_id must be a filesystem-safe name")
        candidate_data = _candidate_data(candidate)
        metadata = {
            "schema_version": 1,
            "occurrence_id": occurrence_id,
            "candidate": candidate_data,
            "detector_config": detector_config.to_dict(),
        }
        bundle = str(Path("candidates") / occurrence_id)
        expected_directory = self.root / bundle
        with closing(self._connect()) as db:
            existing = db.execute(
                "SELECT * FROM occurrences WHERE occurrence_id = ?", (occurrence_id,)
            ).fetchone()
        if existing is not None:
            record = self._record(existing)
            if record.directory != expected_directory:
                raise ValueError(f"occurrence ID {occurrence_id!r} has an invalid bundle path")
            if not record.directory.is_dir():
                raise FileNotFoundError(record.directory)
            stored_metadata, _ = _read_bundle(record.directory)
            if stored_metadata != metadata:
                raise ValueError(f"occurrence ID {occurrence_id!r} has conflicting data")
            return record, False

        published = self._publish(occurrence_id, candidate, metadata)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            result = self._insert(db, occurrence_id, bundle, published)
            db.commit()
            return result

    def load(self, occurrence_id: str) -> ReactionCandidate:
        """Load one occurrence's immutable candidate endpoints and metadata."""

        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT * FROM occurrences WHERE occurrence_id = ?", (occurrence_id,)
            ).fetchone()
        if row is None:
            raise KeyError(occurrence_id)
        return _read_bundle(self.root / row["bundle"])[1]

    def load_detector_config(self, occurrence_id: str) -> BondDetectorConfig:
        """Load the detector settings recorded with an occurrence."""

        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT * FROM occurrences WHERE occurrence_id = ?", (occurrence_id,)
            ).fetchone()
        if row is None:
            raise KeyError(occurrence_id)
        metadata = json.loads(
            (self.root / row["bundle"] / "candidate.json").read_text(encoding="utf-8")
        )
        if metadata.get("schema_version") != 1:
            raise ValueError(f"unsupported candidate bundle for {occurrence_id!r}")
        return BondDetectorConfig.from_dict(metadata["detector_config"])

    def records(self) -> tuple[OccurrenceRecord, ...]:
        """Return every occurrence in registration order."""

        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM occurrences ORDER BY sequence").fetchall()
        return tuple(self._record(row) for row in rows)


__all__ = ["OccurrenceRecord", "OccurrenceStore"]
