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
job size remain site policy rather than ReactionFlow settings.

Every trajectory writes to `output_root/<trajectory-id>`. Resubmitting the same campaign exactly
resumes incomplete workers from their last complete observation checkpoint and treats completed
workers as idempotent no-ops. A per-trajectory contract binds the structure digest, trajectory
conditions, reaction settings, adapter factory, and adapter options to that directory; changing
those scientific inputs is rejected instead of being mislabeled as an exact resume.
