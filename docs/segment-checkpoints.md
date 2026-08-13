# Segment checkpoints

`SegmentStore` publishes calculator-free structural checkpoints and gives every resumed segment a
new trajectory path. It does not run ASE dynamics or decide when a simulation should stop; callers
checkpoint only at a boundary they consider safe.

```python
from reactionflow.segments import ResumeToken, SegmentStore

store = SegmentStore("run-01")
segment = store.start(atoms)

# Attach a calculator and write MD frames to segment.trajectory_path.
# At a safe boundary, publish the current structure and counters:
token = store.checkpoint(
    segment,
    running_atoms,
    global_step=10_000,
    global_frame=1_000,
)
```

After a process restart, read the token and claim the next generation:

```python
store = SegmentStore("run-01")
token = ResumeToken.read("run-01/segments/0000/checkpoint/resume.json")
segment = store.resume(token)

assert segment.generation == 1
assert segment.trajectory_path.name == "trajectory.traj"
```

The first generation writes under `segments/0000/`; a resumed generation writes under
`segments/0001/`, then `0002/`, and so on. Completed checkpoints and nonempty generations are never
reopened or overwritten. `ReactionRun` may reclaim only an empty generation directory left by an
interrupted resume handoff before its trajectory began.

## Publication and fidelity

The checkpoint snapshot and versioned `resume.json` are staged together, then the complete bundle
is renamed into place. A visible resume token therefore has a visible, readable snapshot beside
it. The token records the source generation and global step/frame counters.

Resume restores positions, momenta, cell, periodicity, stable atom IDs, and counters. The returned
atoms have no calculator attached. Arbitrary calculator, optimizer, thermostat, integrator, and
random-number state is not retained, so this is a **structural resume**, not bitwise-exact
continuation. Driver-specific exact restart and cooperative stop policy are intentionally outside
this layer.
