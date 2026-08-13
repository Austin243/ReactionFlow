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
- While the detector is waiting for bond-change persistence, `pending_bonds` freezes the last
  pre-crossing frame and carries the provisional topology.
- A different complete topology must repeat for `stability_frames` processed observations.
- A new topology during that window restarts product stability without replacing the reactant.
- Disconnected components touched by simultaneous changes become separate candidates, all returned
  in stable atom-ID order.
- `finish()` returns incomplete product topologies with `resolved=False` and drains them once.

A candidate contains full, calculator-free copies of the reactant and product structures. Its
`atom_ids`, `reactant_bonds`, and `product_bonds` describe one connected changed region. The frame
fields record the retained reactant, first observation of the accepted product proposal, and the
observation that confirmed or drained it.

The stability window counts processed frames, not MD steps or physical time. It is uniform across
elements: hydrogen, metals, solvents, and other atoms receive no special transition policy.

## Current limits

This layer identifies geometric topology-change occurrences. It does not decide whether two
candidates are the same reaction class, write a registry, publish files, locate a transition state,
or establish kinetics. Those capabilities are separate roadmap increments.
