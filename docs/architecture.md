# Architecture

## Boundary

ReactionFlow owns reaction detection, reaction identity, scientific pathway refinement, durable
state, and the policy that connects those operations. It does not own cluster scheduling.

```text
standalone ASE runner -----+
external trajectory tailer +----> ReactionFlow core and run store
MatEnsemble adapter -------+
```

The dependency direction is one-way: adapters depend on ReactionFlow. ReactionFlow does not import
MatEnsemble, Flux, Slurm, MPI, a dashboard, or a particular calculator implementation.

The MatEnsemble adapter will initially live in MatEnsemble because Flux resources and dynamic chore
admission are MatEnsemble concepts. A third adapter repository is not justified at this stage.

## Generality

Core algorithms must not contain element-specific reaction control flow. In particular, no atom is
treated specially because it is hydrogen, a metal, a solvent atom, or part of a particular named
functional group.

The detector may use element-dependent covalent radii and user-supplied pair thresholds as input
data. Transition aggregation must instead use generic concepts such as persistence, stable
topologies, connected changed regions, and configurable temporal windows. More specialized
chemistry policies may be added later as optional user-provided policies, never as hidden defaults.

Stable atom IDs are required from the initial frame onward and are validated at every boundary.
The canonical ASE representation is the integer array `atoms.arrays["atom_id"]`. A standalone run
assigns IDs once if they are absent; an external producer must provide them. Compatibility readers
may accept `atoms.info["atom_ids"]`, but normalize it before writing and reject conflicting
representations. Array reordering must not change reaction identity.

The first-alpha reaction graph labels nodes by element and edges as unchanged, formed, or broken
within the connected region touched by a change. Geometry, bond order, charge, spin,
stereochemistry, and periodic image shifts are not identity labels in this milestone. Those limits
are recorded with the identity algorithm and configuration so later schema versions can add richer
identity without silently reclassifying old records.

## Planned modules

- `models.py`: versioned configuration, candidate, outcome, and status records.
- `detection.py`: bond-change detection and generic topology-transition tracking.
- `store.py`: atomic artifact publication, the run layout, and the single-writer SQLite registry.
- `pathway.py`: endpoint preparation, relaxation, interpolation, and ASE CI-NEB.
- `segments.py`: stop, checkpoint, immutable generation, and resume contracts.
- `coordinator.py`: a small scheduler-neutral state machine.
- `local.py`: synchronous ASE execution, including sequential one-GPU operation.
- `trajectory.py`: optional append-only ASE trajectory monitoring and replay.

This is intentionally not decomposed into plugin frameworks, an ORM, a general event bus, or a new
workflow scheduler.

## Coordination model

The coordinator has normal and terminal/recovery phases:

```text
running -> stopping -> refining -> running
                                -> completed
any phase ---------------------> failed
any active phase --------------> interrupted -> recovered phase
```

It consumes domain facts—candidate observed, segment stopped or failed, pathway completed or
failed—and produces a small set of commands: request a safe stop, run a pathway, start the next
segment, or finish. The local executor runs those commands synchronously. The MatEnsemble adapter
translates the same commands into chores and service cohorts.

All complete candidate occurrences are registered. One stop request may cover multiple new
reaction classes from the same segment, and production resumes only after the selected refinements
are complete. Duplicate occurrences are retained but do not automatically launch another pathway.

## Artifact ownership

The package owns a scheduler-independent run directory:

```text
run/
  manifest.json
  state.json
  reactions.sqlite3
  events/bond-events.jsonl
  segments/0000/
    trajectory.traj
    checkpoint.traj
    resume.json
    detector-checkpoint.json
  candidates/<candidate-id>/
    candidate.json
    candidate-status.json
    reactant.traj
    product.traj
    detector-config.json
  pathways/<candidate-id>/
    result.json
    relaxed-reactant.traj
    relaxed-product.traj
    neb.traj
  adapters/<adapter-name>/
    jobs.json
```

Every durable JSON record has a schema version. SQLite uses `PRAGMA user_version`. Artifact paths in
records are relative to the run root. Scientific state is not stored only in pickle files. State
files, checkpoints, and candidate directories are published atomically when partial visibility
could corrupt a restart.

`state.json` is authoritative for the coordinator phase and recovery cursor. SQLite is
authoritative for occurrence, class, and refinement lifecycle. Candidate bundles and pathway
results are immutable scientific facts once atomically published. Each adapter exclusively owns
its subdirectory; the core neither interprets nor rewrites scheduler metadata.

SQLite has one coordinator writer. Workers communicate by publishing complete files. The first
implementation targets ordinary POSIX filesystems; correctness and performance on a particular
shared HPC filesystem must be validated before relying on its SQLite locking behavior. A run may
keep its registry node-local and export immutable artifacts when shared locking is unsuitable.

## Calculator lifecycle

Public APIs accept context-managed calculator providers or serializable import-path specifications,
not persistent calculator instances. A provider receives a context identifying MD, endpoint
relaxation, or a NEB image and must explicitly release its calculator lease. The standalone
one-GPU executor asserts that at most one lease is live, releases one phase before constructing the
next, and does not run NEB images concurrently by default. A subprocess-based provider remains an
option for model stacks whose GPU allocator cannot release memory reliably in-process.

GPU and electronic-structure packages remain user-selected dependencies.

## Resume fidelity

ReactionFlow distinguishes:

- **structural resume**: atoms, momenta, cell, periodicity, stable IDs, and counters are restored;
- **exact resume**: structural state plus integrator, thermostat, random-number, and calculator
  state are restored by an explicitly supported driver codec; and
- **unsupported resume**: required state cannot be reconstructed safely.

The package must never silently describe a structural resume as bitwise exact.

## Scientific scope

The first milestone refines a constrained potential-energy path with endpoint relaxation and
climbing-image NEB. A successful status is `neb_converged`.

It does not establish a validated transition state. Vibrational analysis, confirmation of exactly
one imaginary mode, intrinsic reaction coordinates, zero-point corrections, entropy, free-energy
barriers, tunneling, uncertainty, alternative pathways, and kinetics are future capabilities.
