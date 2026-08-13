# Current-capability parity plan

The `v0.1.0a1` milestone extracts and generalizes the current private MatEnsemble functionality. It
is intentionally incremental: every pull request should leave the repository installable and its
new behavior directly testable.

## Bootstrap commit

The bootstrap establishes packaging, CI, dependency boundaries, scope, provenance, and this plan.
It contains no extracted scientific implementation.

## Cross-repository pull request sequence

`RF-*` pull requests belong to ReactionFlow. `ME-*` pull requests belong to MatEnsemble and begin
only after the required ReactionFlow release is available.

### RF-1: portable bond detection

Add stable atom-ID handling, canonical bond types, pair thresholds, hysteresis, periodic
minimum-image distances, persistence, event records, and checkpointable detector state.

Acceptance:

- formation and breakage require configured persistence;
- the hysteresis gap does not flicker stable topology;
- periodic-boundary and atom-reordering tests pass;
- detector state round-trips through versioned JSON; and
- the implementation has no scheduler or model-stack imports.

### RF-2a: generic candidate tracking

Add topology-transition tracking, stable product windows, connected changed regions, endpoint
snapshots, and generic unresolved output at stream termination.

Transition aggregation is generic. It must not special-case hydrogen handoffs or any other
element. Product stability applies uniformly according to configuration.

Acceptance:

- disconnected changes produce distinct candidates;
- the retained reactant precedes a detector persistence crossing;
- product-topology changes restart stability without replacing that reactant;
- hydrogen and non-hydrogen streams produce the same candidate boundaries and control behavior;
  and
- unfinished product topologies become unresolved candidates at stream termination.

### RF-2b: exact reaction identity

Add element-labeled reaction graphs, unchanged/formed/broken edge labels, atom-renumbering
invariance, forward/reverse equivalence, and exact graph isomorphism. This increment contains no
registry or file I/O.

Acceptance:

- atom renumbering, array order, geometry, and direction do not change topology identity; and
- element labels, connectivity, and non-reversed change labels must match exactly.

### RF-2c: occurrence registry and candidate artifacts

Add the single-writer SQLite registry and atomic candidate bundles. Every occurrence is retained,
graph-equivalent duplicates share a class, and all candidates from one frame are registered.

Acceptance:

- every resolved or unresolved occurrence has its own immutable endpoint bundle and registry row;
- exact topology matches share a class, while only the first resolved occurrence represents it;
- stable IDs and detector settings survive bundle publication and reopening; and
- same-ID retries are idempotent and complete bundles left before a database commit are recovered.

### RF-3: standalone pathway refinement

Add a context-managed calculator-provider seam, endpoint alignment, active-region selection,
bounded variable-cell mapping, fixed-cell endpoint relaxation, relaxed-topology gates, IDPP
interpolation, and serial ASE CI-NEB. Artifact publication remains with the later runner.

Acceptance:

- the analytic double-well test reproduces its approximately 1 eV barrier with one calculator lease
  live at a time;
- calculator-free endpoint or NEB snapshots remain inspectable for every bounded outcome;
- collapsed, unresolved, relaxation-failed, NEB-failed, and `neb_converged` outcomes are distinct;
- stable IDs, product reordering, periodic cell mapping, and frozen spectators are exercised; and
- documentation does not call CI-NEB convergence transition-state validation.

### RF-4: segment, checkpoint, and resume protocol

Add immutable segment generations and atomically published structural checkpoints with resume
tokens. The caller chooses a safe checkpoint boundary.

Acceptance:

- structural restart preserves positions, momenta, cell, periodicity, stable IDs, and counters;
- checkpoint data is atomically complete before its token becomes visible;
- every generation has a distinct trajectory path; and
- completed checkpoint and generation directories are never overwritten.

This increment does not claim exact restart. Driver-specific state codecs remain deferred until a
real driver requires one.

### RF-5: standalone ASE engine

Add the small persistent coordinator, cooperative safe-boundary stop policy, an in-process frame
observer, and a synchronous ASE segment runner. The one-GPU policy runs MD and pathway phases
sequentially and releases calculators between them.

Acceptance:

- a deterministic CPU/EMT workflow performs MD, detects a new reaction, checkpoints, refines a real
  pathway, and resumes;
- a later graph-equivalent occurrence is retained without a second refinement;
- interruption and `ReactionRun.open(...).resume(...)` recover from durable state; and
- a fake one-GPU provider proves at most one calculator lease is live across every phase;
- runnable CPU and serial-GPU-oriented examples document calculator setup, resume, artifacts, and
  a generic batch launch; and
- producer, detector, pathway, and resume failures have distinct statuses.

### RF-6: external trajectory monitor

Add an optional append-only ASE trajectory tailer with bounded polling, detector checkpoints,
event-log replay, heartbeat and summary artifacts, truncation detection, and terminal draining.

Acceptance:

- replay after interruption does not produce conflicting event sequences;
- complete candidate directories become visible atomically;
- terminal draining registers every complete occurrence, including unresolved pending changes; and
- documentation claims truncation detection only, not detection of arbitrary same-length history
  replacement.

### RF-7: standalone alpha release

Validate wheel and source-distribution installation in clean environments, publish a private
GitHub prerelease and checksums for `v0.1.0a1`, and document installation from that release on an
offline or HPC system. MatEnsemble work pins this release rather than an arbitrary branch.

### ME-1: MatEnsemble runtime extension

In MatEnsemble, add a narrow supported interface for polling controllers, task status and results,
dynamic task admission, and atomic producer/service-cohort admission. This PR contains no
reaction-specific science.

Acceptance:

- a fake controller uses no underscore-prefixed manager or pipeline storage;
- dynamic Python work receives normal environment and dependency handling;
- service cohorts receive combined resource validation; and
- existing scheduling behavior and tests remain intact.

### ME-2: MatEnsemble adapter and Flux parity

In MatEnsemble, translate ReactionFlow coordinator commands into production, monitoring, pathway,
and resumed chores. Keep temporary compatibility facades only where current scripts need them.

Acceptance:

- a deterministic Flux smoke workflow executes detection, cooperative stop, real CI-NEB, and one
  resumed segment;
- all same-segment occurrences are registered before resume;
- one stop covers all new classes selected from that segment;
- scheduler, producer, monitor, and pathway failures remain distinguishable; and
- the adapter contains all MatEnsemble-specific resources and job identifiers.

### RF-8: integration fixes and second alpha

Apply fixes exposed by the first real MatEnsemble integration and one-GPU smoke runs, then publish
`v0.1.0a2` with the same clean-install and checksum process.

### ME-3: extraction cleanup

After MatEnsemble CI and at least one representative Flux campaign pass against a pinned ReactionFlow
alpha, deprecate and then remove the duplicated scientific implementation. Keep rollback-compatible
facades for one transition release and update examples and compatibility documentation.

## Correctness changes included in parity

The extraction will not deliberately preserve these known defects:

- candidate ingestion stopping after the first new class;
- selecting a reactant endpoint after the threshold crossing has already begun;
- dropping ordinary pending changes at segment termination;
- inconsistent atom-ID lookup between detection and pathway preparation;
- exposing a resume token before its checkpoint is safely published;
- reporting producer or monitor failure as normal completion; and
- naming CI-NEB convergence `validated`.

## Milestone exit criteria

Parity is complete when both standalone ASE and MatEnsemble/Flux workflows exercise the same core
engine through real end-to-end tests, all current supported scientific artifacts are produced, the
standalone one-GPU path has been smoke-tested, and MatEnsemble no longer contains a second copy of
the reaction-science implementation.

## Deferred work

The first alpha does not include vibrational validation, IRC, free energies, distributed NEB,
multiple pathway guesses, automatic replacement of failed reaction representatives, multi-GPU
scheduling, a dashboard, a general plugin system, or reaction-network exploration.
