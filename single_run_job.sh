#!/bin/bash

#SBATCH --job-name=ensemble_eval
#SBATCH --output=logs/ensemble_eval_%A_%a.out
#SBATCH --error=logs/ensemble_eval_%A_%a.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem=16GB
#SBATCH --partition=small

# One element of the array that multiple_runs_job.sh submits: one dataset and repeat, all seeds.
# A task has 20 long fits per seed (GES, Multi-GES), so 30 CPUs fit 3 seeds in two rounds;
# the workers share the task data, so memory hardly grows with the CPUs (<5 GB at 16 CPUs).
set -eo pipefail

# The TabArena store is too large for the home quota
export TABARENA_CACHE=${TABARENA_CACHE:-/scratch/htc/$USER/tabarena}

# Activate the project environment (created with `uv sync`)
source .venv/bin/activate

python -u src/generate_data.py "$@" \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --n-jobs "$SLURM_CPUS_PER_TASK"
