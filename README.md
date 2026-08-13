# ReactionFlow

ReactionFlow is an early-stage Python package for detecting reaction events in atomistic
trajectories and automating reaction-path refinement around those events.

The project is being extracted from transition-path automation originally developed inside
MatEnsemble. ReactionFlow will be usable in two modes:

1. directly from an ASE molecular-dynamics process, including a serial one-GPU workflow; and
2. through a thin MatEnsemble adapter that supplies scheduling and resources without becoming a
   dependency of the ReactionFlow core.

The project is under active development. Persistent geometric bond-change detection is the first
implemented layer; the remaining extraction is tracked in
[docs/parity-plan.md](docs/parity-plan.md).

## Design principles

- **General-purpose chemistry.** Detection and transition aggregation operate on generic atoms,
  bonds, and topology changes. ReactionFlow will not hard-code hydrogen handoffs or any other
  element-specific mechanism. Element and pair behavior may be supplied as data or configuration.
- **ASE-first, scheduler-independent core.** The core depends on ASE, NumPy, and the
  Python standard library—not MatEnsemble, Flux, a particular ML potential, or an HPC system.
- **Lean orchestration.** A small persistent state machine coordinates detection, checkpointing,
  refinement, and resume. ReactionFlow is not a general workflow engine.
- **Auditable artifacts.** Scientific records use versioned JSON, JSONL, SQLite, and ASE trajectory
  files with atomic publication where state consistency matters.
- **Precise scientific claims.** A converged climbing-image NEB path is not, by itself, a
  vibrationally validated transition state or a free-energy barrier.

See [docs/architecture.md](docs/architecture.md) for the planned package boundary and artifact
ownership and [docs/public-contract.md](docs/public-contract.md) for the provisional API.

## Available now

The first implemented layer is persistent geometric bond-change detection. See
[Geometric bond-change detection](docs/bond-detection.md).

## Initial extraction scope

The first alpha will reproduce the useful capabilities of the private MatEnsemble implementation:

- persistent bond-change detection under periodic boundary conditions;
- stable reaction candidates and graph-based occurrence deduplication;
- endpoint preparation and relaxation;
- ASE IDPP interpolation and climbing-image NEB;
- cooperative MD stop, checkpoint, and resume;
- synchronous standalone execution and a live trajectory-watching mode; and
- optional MatEnsemble integration through a narrow adapter.

Frequency calculations, imaginary-mode validation, IRC calculations, free energies, distributed
NEB, and reaction-network exploration are intentionally deferred.

## Development

ReactionFlow currently requires Python 3.12 or newer.

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
python -m build
```

## License and provenance

ReactionFlow is distributed under the BSD 3-Clause License. Planned extracted portions originate
from the BSD-licensed MatEnsemble codebase; provenance is recorded in [NOTICE](NOTICE) and
[docs/provenance.md](docs/provenance.md).
