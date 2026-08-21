# Four high-pressure ACN trajectories on Perlmutter

This example launches four independent hydrostatic NPT trajectories from the same relaxed,
192-atom beta-acetonitrile structure. They use pinned ANI-1xnr model member 0 at 20 GPa and
100, 300, 500, and 700 K with seeds 11, 22, 33, and 44. Each trajectory runs 1,000 steps at
1 fs per step.

The Slurm allocation is one GPU node, four tasks, and one GPU per task. Each task runs its live
bond monitor in-process on local CPU cores. A confirmed event pauses only that trajectory,
relaxes both endpoints, runs an initial NEB followed by CI-NEB, and then restores its exact MD
checkpoint and continues.

From the repository root on Perlmutter:

```bash
./scripts/setup-perlmutter-ani1xnr.sh
sbatch -A <GPU_PROJECT> -q regular \
  examples/perlmutter/acn_20gpa_ani1xnr/submit.sbatch
```

If the entire project GPU allocation is exhausted, NERSC permits the free, low-priority overrun
QOS and requires `--time-min`:

```bash
sbatch -A <GPU_PROJECT> -q overrun --time-min=00:10:00 \
  examples/perlmutter/acn_20gpa_ani1xnr/submit.sbatch
```

Results are written under `outputs/acn_20gpa_ani1xnr/<trajectory-id>/`. Re-submitting the same
campaign resumes incomplete trajectories from their durable exact checkpoints and leaves
completed trajectories unchanged.

Campaign size is not capped at 4 or 32. Add trajectory objects to `campaign.json`, request one
task and GPU per trajectory, and request enough four-GPU nodes. For 32 trajectories:

```bash
sbatch -A <GPU_PROJECT> -q regular --nodes=8 --ntasks=32 \
  examples/perlmutter/acn_20gpa_ani1xnr/submit.sbatch
```

The command intentionally fails when the Slurm task count and campaign trajectory count differ.
