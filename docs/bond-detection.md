# Geometric bond-change detection

ReactionFlow detects persistent changes in geometric connectivity from ordered ASE `Atoms` frames.
This layer has no scheduler, trajectory watcher, reaction grouping, or pathway-search dependency.

## Example

```python
from ase import Atoms

from reactionflow import BondChangeDetector, BondDetectorConfig, assign_atom_ids

atoms = assign_atom_ids(Atoms("CO", positions=[[0, 0, 0], [2.0, 0, 0]]))
detector = BondChangeDetector(
    BondDetectorConfig(
        persistence_frames=2,
        pair_thresholds={"C-O": (1.5, 1.8)},
    )
)

assert detector.process(atoms, frame=0) == []  # eventless baseline
atoms.positions[1, 0] = 1.4
assert detector.process(atoms, frame=1) == []
atoms.positions[1, 0] = 1.3
event = detector.process(atoms, frame=2)[0]

assert event.event_type == "formed"
assert event.atom_ids == (0, 1)
```

The runnable version is [`examples/detect_bonds.py`](../examples/detect_bonds.py).

## Semantics

- Distances use ASE's angstrom convention.
- The first frame establishes the baseline and emits no events.
- Formation uses `distance <= form_distance`; breakage uses
  `distance >= break_distance`.
- A crossing must appear in `persistence_frames` consecutive processed frames. A frame in the
  hysteresis gap resets an unconfirmed crossing.
- Frame numbers must increase, but may have gaps. Persistence counts processed frames, not MD
  integration steps or elapsed time.

Default thresholds scale the sum of ASE covalent radii. Pair overrides are order-independent:

```python
BondDetectorConfig(
    form_scale=1.15,
    break_scale=1.30,
    pair_thresholds={"C-O": (1.5, 1.8)},
)
```

Elements only select threshold data. Hydrogen and non-hydrogen systems use the same state machine.

## Atom identity and periodic systems

Each frame must carry one unique stable integer ID per atom in `atoms.arrays["atom_id"]`. The helper
`assign_atom_ids()` assigns `0..N-1` when starting a new trajectory. ReactionFlow also reads
`atoms.info["atom_ids"]` for compatibility with existing ASE trajectory files. The same ID must
continue to identify the same element, even if atom order changes.

ASE neighbor lists provide minimum-image distances for nonperiodic, partial-periodic, orthogonal,
and triclinic cells. A previously stable bond remains monitored after it leaves the neighbor-list
cutoff so breakage is still detected.

## Checkpointing

Detector state is ordinary versioned JSON data:

```python
import json

payload = json.loads(json.dumps(detector.export_state()))
restored = BondChangeDetector.from_state(payload, config=detector.config)
```

The checkpoint stores configuration, atom identities, stable bonds, pending crossings, and the last
frame. Version 1 is intentionally small; stricter validation and migrations will be added when a
durable run store exists.

## Scientific limits

This detector reports persistent geometric connectivity changes. It does not establish electronic
bond order, identify a reaction mechanism, locate or validate a transition state, or estimate
kinetics. Thresholds and saved-frame cadence require system-specific calibration.
