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

# Slurm assigns one GPU to each task. The ReactionFlow process propagates MD on that GPU and runs
# its bond monitor on the task's local CPU cores. SLURM_PROCID selects exactly one trajectory.
srun --cpu-bind=cores reactionflow run "$campaign"
