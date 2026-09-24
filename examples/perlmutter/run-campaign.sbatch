#!/bin/bash
#SBATCH --job-name=reactionflow
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=reactionflow-%j.out
#SBATCH --error=reactionflow-%j.err

set -euo pipefail

campaign=${1:?Usage: sbatch [Slurm options] run-campaign.sbatch CAMPAIGN.json}

# Same environment as the ANI-1xnr example: submit from the checkout or set REACTIONFLOW_ROOT.
repo_root="${REACTIONFLOW_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset}}"
repo_root="$(cd "$repo_root" && pwd -P)"
module load pytorch/2.11.0

export PYTHONUSERBASE=${REACTIONFLOW_PYTHONUSERBASE:-"$repo_root/.perlmutter-python"}
export TORCHANI_DATA_DIR=${REACTIONFLOW_TORCHANI_DATA_DIR:-"$repo_root/.cache/torchani"}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=16
unset PYTHONNOUSERSITE || true

[[ -x "$PYTHONUSERBASE/bin/reactionflow" ]] || {
    echo "Run scripts/setup-perlmutter-ani1xnr.sh before submitting." >&2
    exit 1
}

# Slurm assigns one GPU to each task. The ReactionFlow process propagates MD on that GPU and runs
# its bond monitor on the task's local CPU cores. SLURM_PROCID selects exactly one trajectory.
srun --cpu-bind=cores reactionflow run "$campaign"
