"""Minimal durable storage for reaction-candidate occurrences."""

from __future__ import annotations

import json
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

from ._durable import ensure_directory, publish, sync_directory
from .candidates import ReactionCandidate, ReactionClasses
from .detection import BondDetectorConfig, atom_ids, canonical_copy, transport_copy


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
    symbols = atoms.get_chemical_symbols()
    order = sorted(range(len(ids)), key=ids.__getitem__)
    data = {
        "atom_ids": [ids[index] for index in order],
        "symbols": [symbols[index] for index in order],
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


def _read_bundle(directory: Path) -> tuple[dict[str, object], ReactionCandidate]:
    metadata = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"unsupported candidate bundle in {directory}")
    data = metadata["candidate"]
    reactant = canonical_copy(read(directory / "reactant.traj"))
    product = canonical_copy(read(directory / "product.traj"))
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
        self._classes: ReactionClasses | None = None
        ensure_directory(self.candidates)
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

    def _row(self, db: sqlite3.Connection, occurrence_id: str) -> sqlite3.Row | None:
        return db.execute(
            "SELECT * FROM occurrences WHERE occurrence_id = ?", (occurrence_id,)
        ).fetchone()

    def _class_id(self, db: sqlite3.Connection, candidate: ReactionCandidate) -> str:
        """Return the class of an equivalent stored reaction, or name a new class."""

        if self._classes is None:
            # One bundle per class is read once; later lookups compare graphs in memory.
            self._classes = ReactionClasses()
            for row in db.execute(
                "SELECT class_id, bundle, MIN(sequence) FROM occurrences GROUP BY class_id"
            ):
                self._classes.add(row["class_id"], _read_bundle(self.root / row["bundle"])[1])
        class_id = self._classes.find(candidate)
        if class_id is None:
            class_id = f"reaction-{uuid4().hex}"
            self._classes.add(class_id, candidate)
        return class_id

    def _insert(
        self,
        db: sqlite3.Connection,
        occurrence_id: str,
        candidate: ReactionCandidate,
    ) -> tuple[OccurrenceRecord, bool]:
        existing = self._row(db, occurrence_id)
        if existing is not None:
            return self._record(existing), False
        class_id = self._class_id(db, candidate)
        has_representative = db.execute(
            "SELECT 1 FROM occurrences WHERE class_id = ? AND representative = 1", (class_id,)
        ).fetchone()
        representative = candidate.resolved and has_representative is None
        db.execute(
            """INSERT INTO occurrences
               (occurrence_id, class_id, representative, bundle)
               VALUES (?, ?, ?, ?)""",
            (occurrence_id, class_id, int(representative), f"candidates/{occurrence_id}"),
        )
        row = self._row(db, occurrence_id)
        assert row is not None
        return self._record(row), True

    def _recover(self) -> None:
        # An interrupted publication may have renamed a bundle without syncing its parent.
        sync_directory(self.candidates)
        bundles = {
            path.name
            for path in self.candidates.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            registered = {row["occurrence_id"] for row in db.execute("SELECT * FROM occurrences")}
            for occurrence_id in sorted(registered - bundles):
                raise FileNotFoundError(self.candidates / occurrence_id)
            # A complete bundle left before its row was committed is registered now.
            for occurrence_id in sorted(bundles - registered):
                metadata, candidate = _read_bundle(self.candidates / occurrence_id)
                if metadata.get("occurrence_id") != occurrence_id:
                    raise ValueError(f"candidate bundle ID conflicts with {occurrence_id}")
                self._insert(db, occurrence_id, candidate)
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
            sync_directory(self.candidates)
            return stored_candidate

        temporary = self.candidates / f".{occurrence_id}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        try:
            write(temporary / "reactant.traj", transport_copy(candidate.reactant))
            write(temporary / "product.traj", transport_copy(candidate.product))
            (temporary / "candidate.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            publish(temporary, final)
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
        metadata = {
            "schema_version": 1,
            "occurrence_id": occurrence_id,
            "candidate": _candidate_data(candidate),
            "detector_config": detector_config.to_dict(),
        }
        with closing(self._connect()) as db:
            existing = self._row(db, occurrence_id)
        if existing is not None:
            record = self._record(existing)
            if not record.directory.is_dir():
                raise FileNotFoundError(record.directory)
            if _read_bundle(record.directory)[0] != metadata:
                raise ValueError(f"occurrence ID {occurrence_id!r} has conflicting data")
            return record, False

        published = self._publish(occurrence_id, candidate, metadata)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            result = self._insert(db, occurrence_id, published)
            db.commit()
            return result

    def record(self, occurrence_id: str) -> OccurrenceRecord:
        """Return one occurrence's registry assignment."""

        with closing(self._connect()) as db:
            row = self._row(db, occurrence_id)
        if row is None:
            raise KeyError(occurrence_id)
        return self._record(row)

    def load(self, occurrence_id: str) -> ReactionCandidate:
        """Load one occurrence's immutable candidate endpoints and metadata."""

        return _read_bundle(self.record(occurrence_id).directory)[1]

    def load_detector_config(self, occurrence_id: str) -> BondDetectorConfig:
        """Load the detector settings recorded with an occurrence."""

        metadata = _read_bundle(self.record(occurrence_id).directory)[0]
        return BondDetectorConfig.from_dict(metadata["detector_config"])

    def records(self) -> tuple[OccurrenceRecord, ...]:
        """Return every occurrence in registration order."""

        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM occurrences ORDER BY sequence").fetchall()
        return tuple(self._record(row) for row in rows)


__all__ = ["OccurrenceRecord", "OccurrenceStore"]
