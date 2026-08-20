# Exact runtime restart state

`ExactRestartSnapshot` is the portable checkpoint boundary for runtime state that ASE trajectory
files do not contain. A snapshot contains three required parts:

- calculator-free ASE atoms, including positions, momenta, cell, periodicity, arrays, and metadata;
- one versioned dynamics component; and
- one versioned calculator component.

Both components must explicitly declare that their state is exact. ReactionFlow rejects an
inexact component rather than silently describing a structural restart as exact.

```python
from reactionflow import ComponentState, ExactRestartSnapshot
from reactionflow.ase_npt import snapshot_langevin_baoab

snapshot = ExactRestartSnapshot(
    atoms=atoms,
    dynamics=snapshot_langevin_baoab(dynamics),
    calculator=calculator_adapter.snapshot(calculator),
)
snapshot.write("restart-000001")
```

The directory contains a versioned JSON manifest, calculator-free `atoms.traj`, and a compressed
NumPy array archive. Every ASE atom array is stored explicitly because ASE trajectory files do not
retain arbitrary custom arrays. SHA-256 digests detect partial or damaged artifacts, object arrays
and pickle loading are forbidden, and publication uses an atomic rename. Existing checkpoints are
never overwritten.

## Langevin BAOAB NPT

`snapshot_langevin_baoab()` records the dedicated NumPy generator state, integration counter,
acceleration, cell momentum, cell force, barostat drag, and every constructor value required to
recreate ASE's `LangevinBAOAB`. `restore_langevin_baoab()` reconstructs the integrator around a
caller-restored calculator and resumes at the next numerical step.

Exact continuation requires the same ASE and NumPy versions, so the codec records and verifies
both. Observer callbacks and output handles are intentionally recreated by the runner because they
do not influence the propagated state.

Calculator state is not guessed. A future MLIP adapter must either serialize all mutable state or
declare that deterministic reconstruction from pinned model/configuration data is exact. An
adapter that cannot satisfy that contract cannot be used in strict exact-restart mode.

## Exact ReactionRun execution

`ReactionRun.run_exact()` accepts an `ExactRuntimeProvider`. The provider has two context-managed
operations: `start(atoms)` creates a fresh runtime and `restore(snapshot)` reconstructs one. The
acquired runtime exposes its live atoms, exact step counter, `run(steps)`, and `snapshot()`.

At every observation boundary, `ReactionRun` atomically publishes the dynamics/calculator
snapshot together with the detector and reaction-tracker state. Reopening a run therefore retains
an in-progress persistence window instead of forgetting a provisional bond event. When a stable
reaction candidate is emitted, the exact checkpoint is bound to the segment resume token and the
MD provider is released before endpoint and pathway calculators are acquired. The same runtime
state is restored after refinement.

```python
run = ReactionRun.create("trajectory-000", config=config)
summary = run.run_exact(
    atoms,
    runtime_provider=runtime_provider,
    pathway_calculator_provider=pathway_calculators,
    total_steps=1_000_000,
)

# The same call resumes from the last complete observation checkpoint.
run = ReactionRun.open("trajectory-000")
summary = run.run_exact(
    runtime_provider=runtime_provider,
    pathway_calculator_provider=pathway_calculators,
    total_steps=1_000_000,
)
```

The runner refuses a provider whose step counter or atomic state disagrees with the durable
boundary. It never silently falls back from exact to structural recovery. The ANI-1xnr provider is
delivered separately so its model identity and calculator-state contract can be tested directly.
