# Independent trajectory campaigns

A campaign is a JSON file containing one starting structure and any number of independently
parameterized trajectories. Each trajectory uses exactly one MLIP adapter configuration. A schema
version 1 campaign shares one adapter across every trajectory; schema version 2 defines named
adapter profiles and makes the assignment explicit on each trajectory. ReactionFlow does not
create a central scheduler or a monitor service. Each process owns exactly one trajectory,
propagates MD on its one visible GPU, and runs that trajectory's bond monitor on CPU cores local to
the same node at each observation boundary.

There is no ReactionFlow campaign-size ceiling. On Perlmutter, four workers fit on each four-GPU
node. A 4-trajectory campaign uses one node, 32 trajectories use eight nodes, and larger campaigns
use the same mapping until they reach the allocation limits imposed by the site or queue.

## Multiple models in one submission

Use schema version 2 when trajectories in one submission should use different models. Define each
complete adapter configuration once under `adapter_profiles`, then set `adapter_profile` on every
trajectory:

```json
{
  "schema_version": 2,
  "structure": "structure.extxyz",
  "output_root": "runs",
  "require_gpu": true,
  "adapter_profiles": {
    "model-a": {
      "factory": "my_mlip.reactionflow:create_adapter",
      "options": {"checkpoint": "/models/model-a.ckpt"}
    },
    "model-b": {
      "factory": "my_mlip.reactionflow:create_adapter",
      "options": {"checkpoint": "/models/model-b.ckpt"}
    }
  },
  "reaction_run": {
    "observation_interval": 10,
    "detector": {"persistence_frames": 3},
    "candidate_stability_frames": 3
  },
  "trajectories": [
    {
      "id": "model-a-seed11",
      "adapter_profile": "model-a",
      "total_steps": 100000,
      "timestep_fs": 1.0,
      "temperature_K": 100.0,
      "pressure_GPa": 20.0,
      "seed": 11,
      "conditions": {"hydrostatic": true}
    },
    {
      "id": "model-b-seed22",
      "adapter_profile": "model-b",
      "total_steps": 100000,
      "timestep_fs": 1.0,
      "temperature_K": 100.0,
      "pressure_GPa": 20.0,
      "seed": 22,
      "conditions": {"hydrostatic": true}
    }
  ]
}
```

The same representation works for 8, 200, or 500 trajectories: add one trajectory object per GPU
task and assign any profile to each object. A profile can be referenced once or hundreds of times,
and profiles may use different adapter factories as well as different options. Each worker imports
and loads only its selected profile. That profile supplies the calculators for MD, endpoint
relaxation, NEB, CI-NEB, and frequency validation before the worker restores and continues the same
MD trajectory.

The assignment is intentionally explicit rather than inferred from trajectory order, task rank, or
GPU number. This keeps generated campaign files easy to audit and prevents inserting or reordering
a trajectory from silently changing its model. `reactionflow plan` reports the trajectory count,
resource estimate, and assignment counts under `adapter_profile_counts` before submission.

Schema version 2 has no profile inheritance or per-trajectory option merging: every named profile
is a complete adapter specification. All profiles in one submission must be usable in the same
software environment and fit the requested per-task resources. Use separate campaigns when models
need incompatible environments or different GPU shapes.

The bundled ANI-1xnr campaign remains a schema version 1 example. Its top-level `adapter` applies
to all trajectories and continues to be supported unchanged for single-model work.

Paths are relative to the campaign file. Trajectory IDs are unique output-directory names. The
standard temperature, pressure, time-step, seed, and step-count fields give adapters a common
baseline; `conditions` carries additional JSON parameters without putting MLIP- or
integrator-specific settings into ReactionFlow core. Model selection belongs in `adapter_profile`,
not `conditions`.

Validate and size a campaign without importing its MLIP:

```bash
reactionflow validate campaign.json
reactionflow plan campaign.json --gpus-per-node 4
```

## MLIP adapter

### Use an ASE calculator directly

For a deterministic, stateless MLIP that already exposes an ASE `Calculator`, use the built-in
generic adapter. No ReactionFlow-specific Python class is required. This schema version 1 block can
also be used unchanged as the value of a named schema version 2 adapter profile:

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
`calculator_kwargs` and requires it to return an ASE `Calculator`. The normal Perlmutter setup
continues to install TorchANI and the pinned ANI-1xnr model as the ready-to-run default. To add
another calculator to that same checkout environment:

```bash
./scripts/setup-perlmutter-ani1xnr.sh
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

Each profile's `factory` (or `adapter.factory` in schema version 1) is an explicit
`module:callable` reference. ReactionFlow calls the selected factory once in each worker:

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

## Campaign status

`reactionflow status campaign.json` summarizes a campaign without changing it, so it is safe to
run while trajectories are still running. It opens each trajectory's registry read-only and never
imports the MLIP.

The first table has one row per trajectory: model profile, pressure, phase, step, detected events,
converged pathways out of those refined, and the latest error. That error is the permanent
failure for a `failed` trajectory, and otherwise the contents of `last-error.json`. Trajectories
without output are listed as `not started`. A trajectory whose files cannot be read is reported
with that error while the rest of the report still prints.

The second table merges reaction classes across trajectories that share a model profile and
pressure, because only those barriers are comparable. Classes merge by the same exact topology
identity that each trajectory uses, including forward/reverse equivalence. Each row lists the bond
changes, the number of trajectories and events, converged pathways, frequency diagnostics with one
significant imaginary mode, saddle connectivity checks that connected out of those run, and the
minimum, median, and maximum barrier. A line below the table totals the connectivity outcomes
(connect, do not connect, inconclusive, failed) across the campaign. NVT barriers are potential
energies; NPT barriers are enthalpies. `--json` prints the same data, including every barrier
value, connectivity status count, and trajectory ID, for scripted analysis.

## Perlmutter mapping

[`examples/perlmutter/run-campaign.sbatch`](../examples/perlmutter/run-campaign.sbatch) requests
four Slurm tasks per GPU node, one GPU per task, and 32 logical CPU cores per task. `srun` launches
the same command in every worker. `SLURM_PROCID` selects one campaign entry, while Slurm restricts
that process to one `CUDA_VISIBLE_DEVICES` entry. These resource flags follow NERSC's
[Perlmutter GPU job](https://docs.nersc.gov/systems/perlmutter/running-jobs/) and
[process/GPU affinity](https://docs.nersc.gov/jobs/affinity/) guidance. The script loads the same
`.perlmutter-python` environment as the ANI-1xnr example from the checkout it is submitted from, or
from `REACTIONFLOW_ROOT`, and stops with a setup message if that environment is missing.

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
conditions, reaction settings, and resolved adapter factory and options to that directory; changing
the selected model or its scientific inputs is rejected instead of being mislabeled as an exact
resume. Renaming a profile without changing its resolved adapter does not change the scientific
contract.

A worker that stops with an error prints one JSON line with its trajectory ID and the error, and
writes the same record to `last-error.json` in its trajectory directory. The next attempt removes
that file, so a record is present only when the latest attempt ended in an error.
