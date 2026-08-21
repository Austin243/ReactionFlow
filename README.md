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
