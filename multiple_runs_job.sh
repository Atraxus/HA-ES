#!/bin/bash

# Submit multiple jobs with different counter values
for SEED in {0..9}; do
    sbatch --export=SEED=$SEED single_run_job.sh
done
