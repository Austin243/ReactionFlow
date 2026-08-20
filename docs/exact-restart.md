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
NumPy array archive. SHA-256 digests detect partial or damaged artifacts, object arrays and pickle
loading are forbidden, and publication uses an atomic rename. Existing checkpoints are never
overwritten.

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

This layer supplies the artifact and integrator codec. Integration with `ReactionRun`, rolling
checkpoints, and the ANI-1xnr adapter are delivered separately so each capability remains small and
independently testable.
