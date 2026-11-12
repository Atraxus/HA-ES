#!/bin/bash

#SBATCH --job-name=config_stats
#SBATCH --output=logs/config_stats_%j.out
#SBATCH --error=logs/config_stats_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --partition=small

# Measure the memory and disk usage of every TabArena config (src/config_stats.py), about 2,000
# configs fitted one after another on dummy data, and write data/model_memory_and_disk_usage.csv.
# Submit from the project root after `uv sync --group measure` and src/download_data.py:
#   sbatch config_stats_job.sh [--methods ...]
# All configs are measured in one job, so on one node. Keep the CPU count the same for runs you
# compare: AutoGluon sizes the models' thread pools by it (SLURM_CPUS_PER_TASK), which changes
# the inference memory of some models.
set -eo pipefail

# The TabArena store is too large for the home quota
export TABARENA_CACHE=${TABARENA_CACHE:-/scratch/htc/$USER/tabarena}

echo "Node $(hostname), $(lscpu | sed -n 's/^Model name: *//p'), $SLURM_CPUS_PER_TASK CPUs"

source .venv/bin/activate
python -u src/config_stats.py "$@"
