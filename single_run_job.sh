#!/bin/bash

#SBATCH --job-name=ensemble_eval_single
#SBATCH --output=ensemble_eval_%j.out
#SBATCH --error=ensemble_eval_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --partition=std

# Load the Anaconda module
module add anaconda

# Activate the specific Conda environment
conda activate haes

# Counter passed directly to the script
SEED=${SEED:-0}

echo "Starting job with seed $SEED"

# Executing the Python script with the specified counter
python haes/generate_data.py --seed $SEED

# Deactivate the Conda environment
conda deactivate
