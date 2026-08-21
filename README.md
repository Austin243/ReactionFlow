# ReactionFlow

ReactionFlow is an early-stage Python package for detecting reaction events in atomistic
trajectories and automating reaction-path refinement around those events.

The project was extracted from transition-path automation originally developed inside
MatEnsemble, but it now runs independently. Its core is a normal ASE process and has no
MatEnsemble, Flux, scheduler, container, or MLIP dependency.

The project is under active development. Geometric bond detection, reaction-candidate tracking,
exact topology identity, durable occurrence storage, serial NEB/CI-NEB refinement, exact rolling
checkpoints, and pause/refine/resume now form a synchronous standalone ASE workflow; remaining work
is tracked in [docs/parity-plan.md](docs/parity-plan.md).

## Design principles

- **General-purpose chemistry.** Detection and transition aggregation operate on generic atoms,
  bonds, and topology changes. ReactionFlow will not hard-code hydrogen handoffs or any other
  element-specific mechanism. Element and pair behavior may be supplied as data or configuration.
- **ASE-first, scheduler-independent core.** The core depends on ASE, NetworkX, NumPy, and the
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

Available layers:

- [Geometric bond-change detection](docs/bond-detection.md)
- [Reaction candidate tracking](docs/candidate-tracking.md)
- [Reaction topology identity](docs/reaction-identity.md)
- [Occurrence storage](docs/occurrence-store.md)
- [Pathway refinement](docs/pathway-refinement.md)
- [Segment checkpoints](docs/segment-checkpoints.md)
- [Exact runtime restart state](docs/exact-restart.md)
- [Standalone ASE runs](docs/standalone-ase.md)
- [Independent multi-GPU campaigns](docs/campaigns.md)

## Perlmutter quick start: four high-pressure ACN trajectories

The included example requests one Perlmutter GPU node and runs four independent ANI-1xnr
hydrostatic-NPT trajectories of the same relaxed 192-atom beta-acetonitrile structure at 20 GPa.
The trajectories use 100, 300, 500, and 700 K, seeds 11, 22, 33, and 44, and 1,000 steps at
1 fs per step.

Clone the repository into a filesystem visible from compute nodes, then run the checked-in setup
script on a Perlmutter login node:

```bash
cd "$PSCRATCH"
git clone https://github.com/Austin243/ReactionFlow.git
cd ReactionFlow
./scripts/setup-perlmutter-ani1xnr.sh
```

The setup layers pinned ASE, NumPy, TorchANI, ReactionFlow, and pinned/checksummed ANI-1xnr
weights over NERSC's maintained `pytorch/2.11.0` module. It does not build a container. Submit the
four-trajectory example with your GPU project:

```bash
sbatch -A <GPU_PROJECT> -q regular \
  examples/perlmutter/acn_20gpa_ani1xnr/submit.sbatch
```

If the *project's entire GPU allocation* is exhausted, NERSC's overrun QOS is free and
low-priority; it also requires a minimum time:

```bash
sbatch -A <GPU_PROJECT> -q overrun --time-min=00:10:00 \
  examples/perlmutter/acn_20gpa_ani1xnr/submit.sbatch
```

Each Slurm task sees one GPU, propagates one trajectory there, and runs its live bond monitor
in-process on CPU cores on the same node. A confirmed bond change pauses only that trajectory,
relaxes the endpoints, runs NEB and then CI-NEB, restores its exact checkpoint, and continues.
Outputs are under `outputs/acn_20gpa_ani1xnr/<trajectory-id>/`; re-submitting resumes incomplete
work and leaves completed trajectories unchanged.

The scheduler path has no 4- or 32-GPU ceiling. A campaign contains one configuration object per
trajectory, and the launch requests exactly one Slurm task and GPU per object. For example, a
32-trajectory campaign uses `--nodes=8 --ntasks=32` on Perlmutter. See the
[example guide](examples/perlmutter/acn_20gpa_ani1xnr/README.md) and
[campaign guide](docs/campaigns.md).

The commands and CPU-side tests are checked in; the ANI example itself still needs its first
Perlmutter GPU run. The setup follows NERSC's current guidance for
[PyTorch modules](https://docs.nersc.gov/machinelearning/pytorch/#using-nersc-pytorch-modules),
[one GPU per task](https://docs.nersc.gov/systems/perlmutter/running-jobs/#1-node-4-tasks-4-gpus-1-gpu-visible-to-each-task),
and [overrun jobs](https://docs.nersc.gov/jobs/examples/#projects-that-have-exhausted-their-allocation).

## Initial extraction scope

The first alpha will reproduce the useful capabilities of the private MatEnsemble implementation:

- persistent bond-change detection under periodic boundary conditions;
- stable reaction candidates and graph-based occurrence deduplication;
- endpoint preparation and relaxation;
- ASE IDPP interpolation and climbing-image NEB;
- cooperative MD stop, checkpoint, and resume;
- synchronous standalone execution and a live trajectory-watching mode; and
- arbitrary-size one-worker-per-GPU campaigns through a narrow user-selected MLIP adapter.

Frequency calculations, imaginary-mode validation, IRC calculations, free energies, distributed
NEB, and reaction-network exploration are intentionally deferred.

## Development

ReactionFlow currently requires Python 3.12 or newer and ASE 3.28 or newer.

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
