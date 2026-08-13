# Standalone ASE runs

`ReactionRun` connects the detector, candidate tracker, occurrence store, pathway refiner, and
segment checkpoints in one scheduler-independent run directory. `run_ase()` executes those same
operations synchronously.

```python
from contextlib import contextmanager

from ase.md.verlet import VelocityVerlet
from reactionflow import ReactionRun, ReactionRunConfig


@contextmanager
def md_calculators(stage):
    calculator = make_mlip(stage)
    try:
        yield calculator
    finally:
        release_mlip(calculator)


@contextmanager
def pathway_calculators(stage):
    calculator = make_mlip(stage)
    try:
        yield calculator
    finally:
        release_mlip(calculator)


run = ReactionRun.create(
    "run-01",
    config=ReactionRunConfig(observation_interval=100),
)
summary = run.run_ase(
    atoms,
    md_calculator_provider=md_calculators,
    pathway_calculator_provider=pathway_calculators,
    dynamics_factory=lambda frame: VelocityVerlet(frame, timestep=0.5),
    total_steps=100_000,
)
```

Dynamics run in chunks ending at observation boundaries. Every candidate from a boundary is
registered. A newly resolved representative stops the current segment, publishes a structural
checkpoint, and releases the MD calculator before representatives are refined sequentially.
Equivalent later occurrences remain in the registry but do not launch another pathway.

Each calculator provider is a context manager receiving `md`, `relax_reactant`, `relax_product`,
or `neb`. The synchronous executor never overlaps leases. This supports serial one-GPU use, but
release of device memory still depends on the selected calculator implementation.

## Manual driving and recovery

The same run can be driven by a future scheduler adapter without calling `run_ase()`:

```python
segment = run.start(atoms)
records = run.observe(current_atoms, global_step=100, global_frame=1)

if run.phase == "checkpoint_pending":
    token = run.checkpoint(current_atoms)
    outcomes = run.refine_pending(pathway_calculators)
    segment = run.resume_segment()
```

After a completed checkpoint, a new process can reopen the run and continue refinement/resume:

```python
run = ReactionRun.open("run-01")
run.refine_pending(pathway_calculators)
segment = run.resume_segment()
```

`state.json` atomically records the phase, generation, global counters, pending representative
occurrence IDs, detector continuity state, configuration, and orchestration failures. Candidate
bundles, checkpoint bundles, and pathway directories are separately atomic. Each pathway directory
contains versioned `result.json` and calculator-free `images.traj`. If interruption occurs while a
completed checkpoint is claiming its still-empty next generation, reopening safely completes that
handoff; a generation whose trajectory has begun remains immutable.

Recovery is structural: positions, momenta, cell, periodicity, stable IDs, and counters survive a
checkpoint. Integrator, thermostat, random-number, and calculator state do not. An active segment
interrupted before a checkpoint cannot be resumed by this version.
