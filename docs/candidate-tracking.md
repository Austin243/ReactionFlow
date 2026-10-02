# Reaction candidate tracking

`ReactionTracker` groups stable bond topologies into connected reactant/product candidates. It is
an in-memory layer between bond detection and later reaction identity or pathway refinement.

```python
from reactionflow import BondChangeDetector, ReactionTracker

detector = BondChangeDetector()
tracker = ReactionTracker(stability_frames=3)

for frame, atoms in enumerate(frames):
    detector.process(atoms, frame=frame)
    candidates = tracker.process(
        atoms,
        frame=frame,
        stable_bonds=detector.stable_bonds,
        pending_bonds=detector.pending_bonds,
    )
    for candidate in candidates:
        print(candidate.reactant_frame, candidate.product_frame)

unresolved = tracker.finish()
```

Each frame must carry the same stable integer atom IDs used by the detector. Frames may reorder
their arrays because bonds and candidate regions are keyed by those IDs rather than array indices.
Start the detector and tracker together so the tracker sees a baseline before any pending change.

## Tracking semantics

- The first stable topology establishes the accepted reactant basin.
- Each connected changed region has its own stability window. Connectivity uses the union of
  accepted, stable, and provisional bonds, so interacting changes stay together.
- `pending_bonds` freezes each region's last pre-crossing frame. Only provisional detector changes
  in that region block its confirmation; unrelated bond flicker does not delay a stable reaction.
- A product topology must repeat for `stability_frames` consecutive confirmed observations in its
  region. A local topology change or a merge/split restarts that window. Each atom retains one
  pre-crossing origin, with whole-frame snapshots shared across atoms. Merging retains these
  origins so either region can recover its own baseline after a later split.
- Disconnected regions emit independently in stable atom-ID order, and each accepted change emits
  once. The earliest retained snapshot matching the region's accepted topology becomes its
  reactant. If no snapshot matches that baseline, overlapping history is retained as unresolved
  using a truthful changed pair of snapshots, rather than combining incompatible geometry and
  topology. Provisional bonds remain part of the topology recorded with each snapshot.
- `finish()` returns incomplete product topologies with `resolved=False` and drains them once.

A candidate contains full, calculator-free copies of the reactant and product structures. Its
`atom_ids`, `reactant_bonds`, and `product_bonds` describe one connected changed region. The frame
fields record the retained reactant, first observation of the accepted product proposal, and the
observation that confirmed or drained it.

The stability window counts processed frames, not MD steps or physical time. It is uniform across
elements: hydrogen, metals, solvents, and other atoms receive no special transition policy.

Tracker checkpoints use schema version 2 to retain every independent window, endpoint, origin, and
count. The reader also accepts version 1 and splits its global window into connected regions,
retaining the original reactant snapshot; it cannot reconstruct alternative snapshots discarded
by the older tracker.
ReactionRun includes the tracker in reaction checkpoints, so structural and exact continuation both
retain other unfinished regions when a confirmed reaction pauses MD for refinement.

## Current limits

This layer identifies geometric topology-change occurrences. The separate
[reaction topology identity](reaction-identity.md) layer compares candidates, and the
[occurrence store](occurrence-store.md) retains them. These layers do not locate a transition state
or establish kinetics. Endpoints remain actual whole-system frames; simultaneous chemistry outside
a candidate region can still make its pathway refinement unresolved. The tracker does not synthesize
mixed endpoint geometries to remove that activity.
