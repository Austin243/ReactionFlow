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
may accept `atoms.info["atom_ids"]`. Candidate bundles mirror IDs for ASE trajectory transport;
segment-wide publication is added with the segment layer. Array reordering must not change
reaction identity.

The first-alpha reaction graph labels nodes by element and edges as unchanged, formed, or broken
within the connected region touched by a change. Geometry, bond order, charge, spin,
stereochemistry, and periodic image shifts are not identity labels in this milestone. Those limits
are recorded with the identity algorithm and configuration so later schema versions can add richer
identity without silently reclassifying old records.

## Planned modules

- `models.py`: future versioned configuration, outcome, and status records.
- `detection.py`: persistent geometric bond-change detection.
- `candidates.py`: generic topology-transition tracking, candidate records, and exact identity.
- `store.py`: atomic artifact publication, the run layout, and the single-writer SQLite registry.
- `pathway.py`: fixed-cell endpoint preparation, relaxation, IDPP interpolation, and serial ASE
  CI-NEB with context-managed calculator leases.
- `segments.py`: structural checkpoints, immutable generations, and resume tokens.
- `run.py`: the small scheduler-neutral state machine and synchronous ASE executor.
- `trajectory.py`: optional append-only ASE trajectory monitoring and replay.

This is intentionally not decomposed into plugin frameworks, an ORM, a general event bus, or a new
workflow scheduler.

## Coordination model

`ReactionRun` has normal and terminal/recovery phases:

```text
new -> running -> checkpoint_pending -> refining -> resume_ready -> running
                                                      |             -> completed
any phase --------------------------------------------> failed
```

Its `start`, `observe`, `checkpoint`, `refine_pending`, `resume_segment`, and `complete` methods are
scheduler-neutral. `run_ase()` invokes them synchronously; a MatEnsemble adapter can invoke the
same methods from chores later.

All candidate occurrences, including unresolved terminal changes, are registered. One stop request
may cover multiple new reaction classes from the same segment, and production resumes only after
the selected refinements are complete. Duplicate occurrences are retained but do not automatically
launch another pathway; only a resolved occurrence may represent a class for refinement.

## Artifact ownership

The package owns a scheduler-independent run directory:

```text
run/
  state.json
  reactions.sqlite3
  segments/0000/
    trajectory.traj
    checkpoint/
      atoms.traj
      resume.json
  candidates/<occurrence-id>/
    candidate.json  # includes detector configuration
    reactant.traj
    product.traj
  pathways/<occurrence-id>/
    result.json
    images.traj
```

Every durable JSON record has a schema version. SQLite uses `PRAGMA user_version`. Artifact paths in
records are relative to the run root. Scientific state is not stored only in pickle files. State
files, checkpoints, and candidate directories are published atomically when partial visibility
could corrupt a restart.

`state.json` is authoritative for the `ReactionRun` phase and recovery cursor. SQLite is
authoritative for occurrence and class assignment. Candidate bundles and pathway results are
immutable scientific facts once atomically published. A future adapter exclusively owns any
scheduler metadata it adds; the core does not interpret it.

SQLite has one `ReactionRun` writer. The first implementation targets ordinary POSIX filesystems;
correctness and performance on a particular shared HPC filesystem must be validated before relying
on its SQLite locking behavior. A run may keep its registry node-local and export immutable
artifacts when shared locking is unsuitable.

## Calculator lifecycle

Public execution APIs accept context-managed calculator providers, not persistent calculator
instances. A provider receives a stage identifying MD, endpoint relaxation, or NEB and must release
its calculator lease. The standalone executor sequences leases one at a time and does not run NEB
images concurrently.

GPU and electronic-structure packages remain user-selected dependencies.

The pathway primitive uses `relax_reactant`, `relax_product`, or `neb`; `ReactionRun` additionally
uses `md` and owns durable outcome publication. Serializable provider specifications remain
deferred until an adapter has a concrete need.

## Resume fidelity

ReactionFlow distinguishes:

- **structural resume**: atoms, momenta, cell, periodicity, stable IDs, and counters are restored;
- **exact resume**: a future driver integration may restore structural state plus integrator,
  thermostat, random-number, and calculator state through an explicitly supported codec; and
- **unsupported resume**: required state cannot be reconstructed safely.

The package must never silently describe a structural resume as bitwise exact.

## Scientific scope

The first milestone refines a constrained potential-energy path with endpoint relaxation and
climbing-image NEB. A successful status is `neb_converged`.

It does not establish a validated transition state. Vibrational analysis, confirmation of exactly
one imaginary mode, intrinsic reaction coordinates, zero-point corrections, entropy, free-energy
barriers, tunneling, uncertainty, alternative pathways, and kinetics are future capabilities.
