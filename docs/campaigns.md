# Independent trajectory campaigns

A campaign is a JSON file containing one starting structure, one MLIP adapter factory, and any
number of independently parameterized trajectories. ReactionFlow does not create a central
scheduler or a monitor service. Each process owns exactly one trajectory, propagates MD on its one
visible GPU, and runs that trajectory's bond monitor on CPU cores local to the same node at each
observation boundary.

There is no ReactionFlow campaign-size ceiling. On Perlmutter, four workers fit on each four-GPU
node. A 4-trajectory campaign uses one node, 32 trajectories use eight nodes, and larger campaigns
use the same mapping until they reach the allocation limits imposed by the site or queue.

## Campaign file

```json
{
  "schema_version": 1,
  "structure": "structure.extxyz",
  "output_root": "runs",
  "require_gpu": true,
  "adapter": {
    "factory": "my_mlip.reactionflow:create_adapter",
    "options": {"model": "my-model"}
  },
  "reaction_run": {
    "observation_interval": 10,
    "detector": {"persistence_frames": 3},
    "candidate_stability_frames": 3
  },
  "trajectories": [
    {
      "id": "T100-P20-seed11",
      "total_steps": 100000,
      "timestep_fs": 1.0,
      "temperature_K": 100.0,
      "pressure_GPa": 20.0,
      "seed": 11,
      "conditions": {"hydrostatic": true}
    }
  ]
}
```

Paths are relative to the campaign file. Trajectory IDs are unique output-directory names. The
standard temperature, pressure, time-step, seed, and step-count fields give adapters a common
baseline; `conditions` carries additional JSON parameters without putting MLIP- or
integrator-specific settings into ReactionFlow core.

Validate and size a campaign without importing its MLIP:

```bash
reactionflow validate campaign.json
reactionflow plan campaign.json --gpus-per-node 4
```

## MLIP adapter

### Use an ASE calculator directly

For a deterministic, stateless MLIP that already exposes an ASE `Calculator`, use the built-in
generic adapter. No ReactionFlow-specific Python class is required:

```json
{
  "adapter": {
    "factory": "reactionflow.adapters.ase:create_adapter",
    "options": {
      "calculator_factory": "my_mlip.calculator:create_calculator",
      "calculator_kwargs": {
        "checkpoint": "/absolute/path/to/model.ckpt",
        "device": "cuda"
      },
      "model_files": ["/absolute/path/to/model.ckpt"],
      "packages": ["my-mlip-package", "torch"]
    }
  }
}
```

`calculator_factory` can name a calculator class or a function; ReactionFlow calls it with
`calculator_kwargs` and requires it to return an ASE `Calculator`. Install that MLIP in the same
Python environment used to run ReactionFlow. On Perlmutter, create the model-neutral checkout
environment and then add the calculator package:

```bash
./scripts/setup-perlmutter.sh
module load pytorch/2.11.0
export PYTHONUSERBASE="$PWD/.perlmutter-python"
python -m pip install --user my-mlip-package
```

Use absolute model paths in portable batch configurations. Every path in `model_files` is required
at runtime and SHA-256 hashed into each exact checkpoint. ReactionFlow also records the configured
kwargs, Python/ASE/NumPy versions, calculator source hash, its installed distribution version, and
the versions named in `packages`. Changing any of those inputs makes exact resume fail clearly.

This adapter supplies the same exactly restartable Langevin BAOAB NVT/NPT runtime used by the
built-in ANI example. It is intentionally limited to deterministic calculators whose inference
state is fully described by their constructor arguments, files, and package versions. A model with
mutable calculator state or its own RNG should use the small custom adapter interface below so that
state can be captured explicitly.

### Write a custom adapter

`adapter.factory` is an explicit `module:callable` reference. ReactionFlow calls it once in each
worker:

```python
def create_adapter(*, trajectory, options):
    return MyAdapter(trajectory=trajectory, **options)
```

The returned object has only three responsibilities:

- `start(atoms)` context-manages a fresh exact MD runtime;
- `restore(snapshot)` context-manages that runtime from an `ExactRestartSnapshot`; and
- `calculator(stage)` context-manages a calculator for `relax_reactant`, `relax_product`, or
  `neb`.

The MD runtime contract is `atoms`, `nsteps`, `run(steps)`, and `snapshot()`. This narrow boundary
lets a user package an ANI, MACE, NequIP, Allegro, CHGNet, or other ASE-compatible MLIP without
adding that stack to ReactionFlow. Strict execution rejects an adapter that cannot supply exact
dynamics and calculator state.

ReactionFlow includes one optional reference implementation:
`reactionflow.adapters.ani1xnr:create_adapter`. It lazily imports the pinned ANI-1xnr stack, so
normal ReactionFlow installation and import remain independent of Torch. The complete four-GPU
example is in [`examples/perlmutter/acn_20gpa_ani1xnr`](../examples/perlmutter/acn_20gpa_ani1xnr/README.md).

## Perlmutter mapping

[`examples/perlmutter/run-campaign.sbatch`](../examples/perlmutter/run-campaign.sbatch) requests
four Slurm tasks per GPU node, one GPU per task, and 32 logical CPU cores per task. `srun` launches
the same command in every worker. `SLURM_PROCID` selects one campaign entry, while Slurm restricts
that process to one `CUDA_VISIBLE_DEVICES` entry. These resource flags follow NERSC's
[Perlmutter GPU job](https://docs.nersc.gov/systems/perlmutter/running-jobs/) and
[process/GPU affinity](https://docs.nersc.gov/jobs/affinity/) guidance.

Submit four trajectories on one node:

```bash
sbatch -A <project> --nodes=1 --ntasks=4 \
  examples/perlmutter/run-campaign.sbatch campaign.json
```

Submit 32 trajectories on eight nodes:

```bash
sbatch -A <project> --nodes=8 --ntasks=32 \
  examples/perlmutter/run-campaign.sbatch campaign.json
```

For any other size, request the campaign's task count and enough four-GPU nodes. Partial final
nodes are valid, for example `--nodes=33 --ntasks=130`. The CLI deliberately fails before MD if
`SLURM_NTASKS` differs from the number of trajectories, preventing duplicated or omitted work.
NERSC users eligible for scavenger scheduling can add `-q overrun`; queue eligibility and maximum
job size remain site policy rather than ReactionFlow settings. Current NERSC policy also requires
`--time-min` for overrun jobs.

Every trajectory writes to `output_root/<trajectory-id>`. Resubmitting the same campaign exactly
resumes incomplete workers from their last complete observation checkpoint and treats completed
workers as idempotent no-ops. A per-trajectory contract binds the structure digest, trajectory
conditions, reaction settings, adapter factory, and adapter options to that directory; changing
those scientific inputs is rejected instead of being mislabeled as an exact resume.
