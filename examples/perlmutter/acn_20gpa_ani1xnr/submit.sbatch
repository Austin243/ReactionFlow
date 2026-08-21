#!/bin/bash
#SBATCH --job-name=reactionflow-acn
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=reactionflow-acn-%j.out
#SBATCH --error=reactionflow-acn-%j.err

set -euo pipefail

repo_root="${REACTIONFLOW_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset}}"
repo_root="$(cd "$repo_root" && pwd -P)"
module load pytorch/2.11.0

export PYTHONUSERBASE=${REACTIONFLOW_PYTHONUSERBASE:-"$repo_root/.perlmutter-python"}
export TORCHANI_DATA_DIR=${REACTIONFLOW_TORCHANI_DATA_DIR:-"$repo_root/.cache/torchani"}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=16
unset PYTHONNOUSERSITE || true

reactionflow_executable="$PYTHONUSERBASE/bin/reactionflow"
campaign="$repo_root/examples/perlmutter/acn_20gpa_ani1xnr/campaign.json"

[[ -x "$reactionflow_executable" ]] || {
    echo "Run scripts/setup-perlmutter-ani1xnr.sh before submitting." >&2
    exit 1
}

# Slurm gives each process one A100. That process propagates one trajectory on its GPU and
# performs live bond monitoring and any sequential endpoint/NEB/CI-NEB work on the same node.
srun --kill-on-bad-exit=1 --cpu-bind=cores "$reactionflow_executable" run "$campaign"
