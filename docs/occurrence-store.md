# Occurrence storage

`OccurrenceStore` retains every reaction candidate and groups exact topology matches into one
class. It is a small, single-writer persistence layer rather than a workflow database.

```python
from reactionflow.store import OccurrenceStore

store = OccurrenceStore("run")

for index, candidate in enumerate(candidates):
    record, inserted = store.register(
        f"segment-0003-{index:04d}",
        candidate,
        detector_config=detector.config,
    )
    if inserted and record.is_representative:
        print("new resolved reaction class", record.class_id)
```

The caller supplies a stable, filesystem-safe occurrence ID. Repeating the same ID and data is an
idempotent retry with `inserted=False`; using the ID for different endpoint geometry, topology,
frame metadata, or detector settings is an error.

## Stored data

The run root contains one version-1 SQLite table and one immutable directory per occurrence:

```text
run/
  reactions.sqlite3
  candidates/<occurrence-id>/
    candidate.json
    reactant.traj
    product.traj
```

The JSON record contains the region IDs, reactant/product bonds, frame fields, resolved flag,
endpoint integrity digests, and detector configuration. ASE trajectory files carry calculator-free
endpoint structures and transport stable atom IDs through `atoms.info["atom_ids"]`; loading
restores the canonical `atoms.arrays["atom_id"]` representation.

Equivalent occurrences use [`same_reaction()`](reaction-identity.md) and share an opaque class ID.
Every occurrence still has its own row and bundle. An unresolved terminal candidate is retained
but is not a representative; the first later resolved match becomes the class representative.

Directories are written beside their final destination and renamed only after all three files are
complete. If interruption leaves a complete directory before its SQLite row is committed, opening
the store registers that orphan before accepting more work. A database row whose directory is
missing is reported as an error.

## Current limits

This version supports one writer on an ordinary POSIX filesystem. It does not provide concurrent
worker writes, pathway status, retry policy, representative replacement, query indexes, or a
general migration framework. Structure digests used for idempotency cover stable IDs, elements,
positions, cell, and periodicity; exact reaction class identity remains topology-only.
