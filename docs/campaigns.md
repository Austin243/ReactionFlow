# Independent trajectory campaigns

A campaign is a JSON file containing starting structures and any number of independently
parameterized trajectories. Each trajectory uses exactly one MLIP adapter configuration: the
campaign defines named adapter profiles and assigns one explicitly to each trajectory. ReactionFlow
does not create a central scheduler or a monitor service. Each process owns exactly one trajectory,
propagates MD on its one visible GPU, and runs that trajectory's bond monitor on CPU cores local to
the same node at each observation boundary.

There is no ReactionFlow campaign-size ceiling. On Perlmutter, four workers fit on each four-GPU
node. A 4-trajectory campaign uses one node, 32 trajectories use eight nodes, and larger campaigns
use the same mapping until they reach the allocation limits imposed by the site or queue.

## Write a campaign by answering questions

`reactionflow init campaign.json` asks about the trajectories and writes the file. Trajectories
come in groups. A group runs every combination of its starting structures, models, temperatures,
and pressures, as many times as you ask: one structure, two models, four temperatures, and eight
runs make 64 trajectories. Add another group for the part of a campaign that differs, such as 16
runs with one model and 8 with another, or a second structure at other pressures. Each new group
offers the previous group's answers as defaults.

Answers are checked as they are given. Structure files are read and their atom counts printed, and
wildcards such as `structures/*.extxyz` work. Temperatures and pressures take lists or
`start:stop:step` ranges, and `none` as a pressure means constant volume. Models that need separate
Python environments are refused together, and constant pressure needs a periodic cell. AIMNet2,
UMA `omol`, and MACE-POLAR read the charge
and spin multiplicity from the structure, so `init` warns when a file has none.

Trajectory IDs name the structure and the model when a campaign has several, then the temperature,
the pressure, and the run number, for example `acn-0300K-20GPa-01`. Every trajectory gets its own
seed. The file is written only after it passes the same checks as `reactionflow validate`, and
Ctrl-C leaves nothing behind. Detection and refinement settings other than the observation
interval keep their defaults; change them under `reaction_run` afterwards.

## Multiple models in one submission

Define each complete adapter configuration once under `adapter_profiles`, then set
`adapter_profile` on every trajectory. A single-model campaign defines one profile; trajectories in
one submission may also use different models:

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
relaxation, NEB, CI-NEB, and the saddle check before the worker restores and continues the same
MD trajectory.

The assignment is intentionally explicit rather than inferred from trajectory order, task rank, or
GPU number. This keeps generated campaign files easy to audit and prevents inserting or reordering
a trajectory from silently changing its model. `reactionflow validate` reports the trajectory count
and the assignment counts under `adapter_profile_counts` before submission.

There is no profile inheritance or per-trajectory option merging: every named profile is a
complete adapter specification. All profiles in one submission must be usable in the same
software environment and fit the requested per-task resources. Use separate campaigns when models
need incompatible environments or different GPU shapes.

The bundled ANI-1xnr campaign defines one `ani1xnr` profile and assigns it to all four
trajectories.

The structure and output paths are relative to the campaign file; built-in model checkpoint and
cache paths must be absolute. The top-level `structure` applies to every trajectory that does not
name its own `structure`, and can be left out when every trajectory names one. Trajectory IDs are unique output-directory names. The
standard temperature, pressure, time-step, seed, and step-count fields give adapters a common
baseline; `conditions` carries additional JSON parameters without putting MLIP- or
integrator-specific settings into ReactionFlow core. Model selection belongs in `adapter_profile`,
not `conditions`.

Validate a campaign without importing its MLIP:

```bash
reactionflow validate campaign.json
```

## MLIP adapter

### Built-in models

ORB, MatterSim, CHGNet, SevenNet, NEP89, and other models run through the
[generic ASE adapter](models.md#other-models-through-the-ase-adapter). MACE, UMA, AIMNet2/RXN,
and ANI-1xnr have built-in factories and explicit setup/download commands:

```bash
reactionflow models
reactionflow prepare campaign.json --install
reactionflow run campaign.json --index 0
```

`--install` optionally installs the selected backend in the current Python environment. Omit it
when dependencies are already installed. Preparation supports `--index` and does not create any
trajectory state. Normal runs use only prepared local weights; `run --download` explicitly
allows preparation before the selected run. See [models and preparation](models.md) for profile
examples, environment compatibility, UMA authentication, and exact-restart constraints. The pinned
MACE/UMA dependency stacks need separate environments/campaigns; named profiles can mix models
only when all their dependencies are compatible.

`prepare` reports generic and custom adapters as `external_setup`; those adapters do not support
`run --download`.

### Use an ASE calculator directly

For a deterministic, stateless MLIP that already exposes an ASE `Calculator`, use the built-in
generic adapter. No ReactionFlow-specific Python class is required. Use this block as the value of
a named adapter profile:

```json
{
  "factory": "reactionflow.adapters.ase:create_adapter",
  "options": {
    "calculator_factory": "my_mlip.calculator:create_calculator",
    "calculator_kwargs": {
      "checkpoint": "/absolute/path/to/model.ckpt",
      "device": "cuda"
    },
    "model_files": ["/absolute/path/to/model.ckpt"],
    "packages": ["my-mlip-package", "torch"],
    "torch_device": "cuda"
  }
}
```

`calculator_factory` can name a calculator class or a function; ReactionFlow calls it with
`calculator_kwargs` and requires it to return an ASE `Calculator`. The normal Perlmutter setup
continues to install TorchANI and the pinned ANI-1xnr model for the bundled example. Install other
calculator packages in a compatible environment used by your workers, for example:

```bash
python -m pip install my-mlip-package
```

Use absolute model paths in portable batch configurations. Every path in `model_files` is required
at runtime and SHA-256 hashed into each exact checkpoint. ReactionFlow also records the configured
kwargs, Python/ASE/NumPy versions, calculator source hash, its installed distribution version, and
the versions named in `packages`. A changed model file, factory, or kwargs makes resume fail
clearly; a changed package version or calculator source continues with a warning and is recorded
in `restart-environment.jsonl`.
For a Torch model, `torch_device` (`cpu` or `cuda`) turns on deterministic Torch inference before
the calculator is built and records the Torch environment, as the built-in adapters do. Leave it out
for a model that does not use Torch.

This adapter supplies the same exactly restartable Langevin BAOAB NVT/NPT runtime used by the
built-in ANI example. It is intentionally limited to deterministic calculators whose inference
state is fully described by their constructor arguments, files, and package versions. A model with
mutable calculator state or its own RNG should use the small custom adapter interface below so that
state can be captured explicitly.

### Write a custom adapter

Each profile's `factory` is an explicit `module:callable` reference. ReactionFlow calls the
selected factory once in each worker:

```python
def create_adapter(*, trajectory, options):
    return MyAdapter(trajectory=trajectory, **options)
```

The returned object has only three responsibilities:

- `start(atoms)` context-manages a fresh exact MD runtime;
- `restore(snapshot)` context-manages that runtime from an `ExactRestartSnapshot`; and
- `calculator(stage)` context-manages a calculator for `relax_reactant`, `relax_frame`,
  `relax_product`, or `neb`.

The MD runtime contract is `atoms`, `nsteps`, `run(steps)`, and `snapshot()`. This narrow boundary
lets a user package an ANI, MACE, NequIP, Allegro, CHGNet, or other ASE-compatible MLIP without
adding that stack to ReactionFlow. Strict execution rejects an adapter that cannot supply exact
dynamics and calculator state.

`reactionflow models` lists every built-in factory. Normal ReactionFlow installation and import
remain independent of Torch. The complete four-GPU
ANI-1xnr example is in
[`examples/perlmutter/acn_20gpa_ani1xnr`](../examples/perlmutter/acn_20gpa_ani1xnr/README.md).

## When barriers are refined

By default a trajectory pauses at each confirmed reaction, refines it, and continues MD from the
same checkpoint, so a slow refinement holds up that trajectory's MD. Set `"refine_after_md": true`
at the top of the campaign to run MD straight through to `total_steps`. The events wait in a queue
and are refined when MD ends, one at a time in the order MD found them, by the same process on the
same GPU. Resubmit a job that ends partway, as usual: finished results are kept, and the next
unfinished event is refined from the start.

If a refinement gives no barrier, the model is evaluated once on the last checkpoint. If that fails
or returns NaN, the job stops with the rest of the queue intact, so a lost GPU costs one event, not
the queue. The check reloads the model at that checkpoint, so a calculator that needs its previous
call to converge there, as an SCF can, may stop the job and hold the queue. If MD itself cannot go
on, set the option to `false` and resubmit: the queue is refined before MD is touched. A trajectory
marked failed refines its queue when resubmitted, then reports the failure again.

MD restarts exactly after a pause, and refinement reads only saved frames, so with a model that
keeps no state between evaluations both orders give the same frames, events, and barriers. A model
with internal state, such as ASE's EMT and its neighbor list, can differ at rounding level, as it
does after any restart. The setting is not part of a trajectory's contract, so you can change it
when you resubmit. A trajectory already paused for refinement finishes that refinement first.

`validate`, and `run` when it starts a trajectory, warn about settings that quietly shape the result,
on standard error for `run` and under `warnings` for `validate`: a structure with several frames, of
which the run takes the last; momenta in the structure, which replace `temperature_K` for the
starting velocities; constraints, which hold in MD and are removed in refinement; a cell at a pressure
that refinement cannot use; and a run with too few observations to confirm any reaction. None of
these stops the run.

## Campaign status

`reactionflow status campaign.json` summarizes a campaign without changing it, so it is safe to
run while trajectories are still running. It opens each trajectory's registry read-only and never
imports the MLIP.

The first table has one row per trajectory: model profile, pressure, phase, step, detected events,
converged pathways out of those refined, and the latest error. That error is the permanent
failure for a `failed` trajectory, and otherwise the contents of `last-error.json`. Trajectories
without output are listed as `not started`. A file that cannot be read costs only what depends on it:
the error column counts what was left out, `--json` lists it as `unreadable`, and a line under the
table says the report is incomplete. A class whose candidate bundles cannot all be read, as in a copy
of a run's results alone, keeps its counts and barriers in a row of its own, because classes merge
only by their graphs. Saved frames that are gone appear in the error column and as `missing_frames`.
Pathway counts include every refined occurrence. An event refined in steps counts each step as a
refined pathway, and only its steps can converge. Events still waiting to be refined are not among
them: a line under the table counts them, and `--json` gives `queued`.

The second table merges reaction classes across trajectories that share a model profile and
pressure, because only those barriers are comparable. Classes merge by the local change: the
changed bonds and the atoms within `--radius` bonds of them (default 2), including forward/reverse
equivalence ([local classes](reaction-identity.md#local-classes)). Each row lists the bond changes
with the formula of those atoms, the number of trajectories and events, converged bands out of
those refined, validated saddles, and the minimum, median, and maximum barrier. A barrier is measured
from each event's own reactant, so a class that runs both ways has two barriers: the row's barrier
is for the direction it names, and a further column, `reverse_barriers_eV` in `--json`, is for the
other. Each step of an event refined in steps is an event of its own local class
([steps](pathway-refinement.md#reactant-frame-and-steps)). Barriers come only from validated
saddles ([saddle check](pathway-refinement.md#saddle-check)). A line below the table
totals the saddle check's verdicts across the campaign (validated, no saddle, rejected, unchecked)
and counts separately the bands that converged before the check existed, which carry no verdict.
Events that ended with no converged band are counted by status and message, up to five in the text
and all of them as `no_band` in `--json`.
NVT barriers are potential energies; NPT barriers are enthalpies. `--json` prints the same data,
including every barrier value, the counts of each pathway, frequency and connectivity status,
trajectory ID, and one example occurrence per class, for scripted analysis.

## Time limit per barrier

Refining one event stops after `max_hours_per_barrier` hours, 4 by default. Set it at the top of the
campaign to a positive number, or to `null` for no limit. The event then ends as `time_limit`, with
no barrier and its band as far as it got, and the trajectory moves on. MD has no time limit. The
steps of an event refined in steps share one limit: steps already refined keep their results, and
those not started end as `time_limit` with no images.

The clock counts the time a refinement has run, not the time a job waits in the queue. It carries
across resubmissions, because an interrupted refinement starts over when it resumes, and a job
shorter than the refinement would otherwise repeat it forever. So work lost to an interruption counts
against the limit. The count is saved once a minute, so each resubmission can lose up to a minute of
it, and a job that refines for under a minute adds nothing. It is checked before each model
evaluation, so a single evaluation that never returns is not stopped.

The setting is not part of a trajectory's contract, so you can change it when you resubmit, and the
new limit applies to the event being refined and to later ones. An event that already ended as
`time_limit` stays that way, because results are never overwritten.

Four hours comes from the file times and job logs of 288-atom acetonitrile runs at high pressure
with UMA on one GPU, on older commits. The two refinements that reached a connected saddle, both at
700 K, took 1.1 and 1.8 hours. None of the nine longest, 8.5 to 26.7 hours at 2000 K or hotter,
reached a saddle: four did not converge and five had no imaginary mode. More atoms, more images, a
slower model, or a CPU can need more, and `status` counts the refinements that hit the limit.

## Perlmutter mapping

The bundled ANI environment is prepared by
[`scripts/setup-perlmutter-ani1xnr.sh`](../scripts/setup-perlmutter-ani1xnr.sh). For MACE or UMA,
use the separate backend environment described in [model preparation](models.md).

[`examples/perlmutter/run-campaign.sbatch`](../examples/perlmutter/run-campaign.sbatch) requests
four Slurm tasks per GPU node, one GPU per task, and 32 logical CPU cores per task. `srun` launches
the same command in every worker. `SLURM_PROCID` selects one campaign entry, while Slurm restricts
that process to one `CUDA_VISIBLE_DEVICES` entry. These resource flags follow NERSC's
[Perlmutter GPU job](https://docs.nersc.gov/systems/perlmutter/running-jobs/) and
[process/GPU affinity](https://docs.nersc.gov/jobs/affinity/) guidance. The script loads the same
`.perlmutter-python` environment as the ANI-1xnr example from the checkout it is submitted from, or
from `REACTIONFLOW_ROOT`, and stops with a setup message if that environment is missing.
For MACE or UMA, adapt the environment activation in your own job script while preserving the
one-task-per-trajectory mapping; the supplied scripts configure ANI-1xnr.

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
