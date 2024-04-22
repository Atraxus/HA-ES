#!/bin/bash

#SBATCH --job-name=ensemble_eval_single
#SBATCH --output=ensemble_eval_%j.out
#SBATCH --error=ensemble_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8GB
#SBATCH --partition=std

# Counter passed directly to the script
SEED=0
TIME=`date +"%Y%m%d_%T"`
FILENAME="${SEED}_${TIME}.txt"

echo "Starting job with seed $SEED"

# Executing the Python script with the specified counter
python haes/main.py --seed $SEED >> $FILENAME
