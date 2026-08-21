# ReactionFlow

ReactionFlow turns reactive events observed during atomistic molecular dynamics into candidate
transition paths. It monitors bond changes while each trajectory runs. When a persistent bond
formation or breaking event is detected, ReactionFlow checkpoints and pauses that trajectory,
relaxes the structures on both sides of the event, runs NEB followed by climbing-image NEB, then
restores the exact molecular-dynamics state and continues the trajectory.

The goal is to remove the manual step between seeing chemistry happen in an MD trajectory and
calculating the corresponding minimum-energy path. ReactionFlow preserves the integrator,
thermostat/barostat, random-number state, atomic state, and calculator contract needed for an exact
restart. A CI-NEB saddle is a candidate transition state; frequency analysis or another appropriate
validation is still required before calling it a confirmed transition state.

## Perlmutter quick start

The included campaign requests one Perlmutter GPU node and runs four independent ANI-1xnr
hydrostatic-NPT trajectories of a relaxed 192-atom beta-acetonitrile structure at 20 GPa. The four
trajectories run at 100, 300, 500, and 700 K with different random seeds for 1,000 steps at 1 fs per
step. Each Slurm task uses one GPU and runs its live bond monitor on CPU cores on the same node.

### Clone and install

Log in to Perlmutter and run:

```bash
cd "$PSCRATCH"
git clone https://github.com/Austin243/ReactionFlow.git
cd ReactionFlow
./scripts/setup-perlmutter-ani1xnr.sh
```

The setup script loads NERSC's `pytorch/2.11.0` module, installs the pinned Python dependencies and
ReactionFlow into `.perlmutter-python/` inside the checkout, downloads and verifies the pinned
ANI-1xnr model data, and validates the campaign. It can be run again safely after updating the
checkout. No container or separate Conda environment is required.

### Run the campaign

From the repository root, enter the NERSC project that should be charged for GPU time and submit
the job:

```bash
read -r -p "NERSC GPU project: " GPU_PROJECT
sbatch -A "$GPU_PROJECT" -q regular \
  examples/perlmutter/acn_20gpa_ani1xnr/submit.sbatch
```

If that project's GPU allocation is exhausted, submit to the free, low-priority `overrun` QOS:

```bash
read -r -p "NERSC GPU project: " GPU_PROJECT
sbatch -A "$GPU_PROJECT" -q overrun --time-min=00:10:00 \
  examples/perlmutter/acn_20gpa_ani1xnr/submit.sbatch
```

Results are written to `outputs/acn_20gpa_ani1xnr/<trajectory-id>/`. If a confirmed bond change is
found, only that trajectory pauses for endpoint relaxation, NEB, and CI-NEB before resuming from its
exact checkpoint. Submit the same command again after an interruption to resume incomplete
trajectories; completed trajectories are left unchanged.

## Refinement outcomes and recorded data

After a detected reaction is confirmed, ReactionFlow checkpoints the original MD state and refines
every queued pathway serially. A successful CI-NEB or a handled scientific failure is recorded as a
durable outcome. ReactionFlow then restores the checkpoint and continues the original MD
trajectory; it never substitutes a relaxed endpoint or NEB image for the MD state.

| Situation | Recorded status | Behavior |
| --- | --- | --- |
| Endpoint relaxation and CI-NEB converge | `ci_neb_converged` | Save the band, image energies, and barrier; resume MD. |
| Either endpoint does not relax within the configured limits | `relaxation_failed` | Save the attempted relaxed endpoints, skip NEB, and resume MD. |
| Both relaxed endpoints occupy the same bond-topology basin, including a product that relaxes back to the reactant | `collapsed` | Save the relaxed endpoints, skip NEB, and resume MD. |
| The candidate or relaxed endpoint topology remains ambiguous or no longer matches the detected event | `unresolved` | Save every available endpoint image, skip NEB, and resume MD. |
| Initial NEB does not converge | `neb_failed` | Save the current band, skip CI-NEB, and resume MD. |
| Climbing-image NEB does not converge | `ci_neb_failed` | Save the current band and resume MD. |
| Pathway preparation or calculator evaluation raises an unexpected error | `failed` | Save the available images and error message, then resume MD. |

Failures to persist an outcome, restore the exact checkpoint, or otherwise maintain durable run
state are different: ReactionFlow marks the trajectory itself as failed and stops instead of
continuing from uncertain state. Scientific refinement failures are not retried automatically.

Each detected occurrence and its pathway result share an `occurrence-id`:

```text
outputs/acn_20gpa_ani1xnr/<trajectory-id>/
├── candidates/<occurrence-id>/
│   ├── candidate.json
│   ├── reactant.traj
│   └── product.traj
└── pathways/<occurrence-id>/
    ├── result.json
    └── images.traj
```

`candidate.json` records the stable atom IDs in the connected reacting region, reactant and product
bond lists, detector settings, endpoint hashes, and three source-frame fields:

- `reactant_frame`: the last accepted stable observation before the topology change.
- `product_frame`: the first observation containing the proposed product topology.
- `observed_frame`: the observation at which the persistence checks confirmed the event.

These are bond-monitor observation frames, not raw MD step numbers. The complete endpoint
structures and pathway images retain stable IDs for every atom, and segment trajectory boundaries
record both their global MD step and observation-frame counters. `result.json` records the outcome
status, reaction class and occurrence IDs, barrier and image energies when available, and a message
describing any failure.

## Use another ASE-compatible MLIP

TorchANI and ANI-1xnr remain the ready-to-run default. To select another installed,
ASE-compatible MLIP, replace the `adapter` block in the campaign JSON:

```json
"adapter": {
  "factory": "reactionflow.adapters.ase:create_adapter",
  "options": {
    "calculator_factory": "your_mlip.calculators:create_calculator",
    "calculator_kwargs": {
      "checkpoint": "/global/cfs/cdirs/your_project/models/model.ckpt",
      "device": "cuda"
    },
    "model_files": [
      "/global/cfs/cdirs/your_project/models/model.ckpt"
    ]
  }
}
```

`calculator_factory` is the importable `module:callable` for a calculator class or function; it
must return an ASE `Calculator`. ReactionFlow passes `calculator_kwargs` directly to that callable,
so change `checkpoint` and the other keys to the arguments that calculator expects. Put the model's
absolute, compute-node-visible checkpoint path in both the appropriate calculator argument and
`model_files`. Install the calculator package in `.perlmutter-python` before running the campaign.
See the [campaign guide](docs/campaigns.md#use-an-ase-calculator-directly) for the full interface.

## Change temperatures and pressures

Edit each entry in the `trajectories` array of the campaign JSON. Every trajectory can use its own
temperature, pressure, and random seed:

```json
"temperature_K": 300.0,
"pressure_GPa": 20.0,
"seed": 11
```

A numeric `pressure_GPa` runs NPT at that target pressure. Set `"pressure_GPa": null` for NVT.
Thermostat and barostat coupling times can be changed in that trajectory's `conditions` object.

## Change the number of GPU nodes

For a customized campaign, set the resource lines near the top of
`examples/perlmutter/run-campaign.sbatch`. For example, 10 Perlmutter GPU nodes and 40 trajectories
use:

```bash
#SBATCH --nodes=10
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=4
```

The script assigns one Slurm task to each GPU, so the campaign must contain exactly one trajectory
entry per task: four entries per Perlmutter GPU node. Leave the included ANI-1xnr `submit.sbatch`
unchanged when running the bundled four-trajectory example.
