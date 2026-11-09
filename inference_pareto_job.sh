#!/bin/bash

#SBATCH --job-name=inference_pareto
#SBATCH --output=logs/inference_pareto_%A_%a.out
#SBATCH --error=logs/inference_pareto_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --partition=YOUR_PARTITION
#SBATCH --array=0-37

# Inference time against test Elo on the pruned cross-family and foundation-model pools
# (src/inference_pareto.py). One array element per classification dataset, then one job that merges
# them and draws the plot. Replace YOUR_PARTITION (twice) with your Slurm partition, then start it
# from the project root, not with sbatch:
#   export TABARENA_CACHE=/path/to/tabarena/store
#   bash inference_pareto_job.sh [--seed S] [--folds 0 1 2] [--steps 40]
# Outputs: results/inference_pareto/seed_<S>/<dataset>/, data/inference_pareto_*.csv and
# plots/inference-pareto-mean-time.png. Needs `uv sync --group notebooks`.
set -eo pipefail

: "${TABARENA_CACHE:?Set TABARENA_CACHE to the TabArena store}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    mkdir -p logs
    ARRAY_JOB=$(sbatch --parsable "$0" "$@")
    AGGREGATE_ARGS=()
    args=("$@")
    for ((i = 0; i < ${#args[@]} - 1; i++)); do
        case "${args[i]}" in
            --seed | --output-root) AGGREGATE_ARGS+=("${args[i]}" "${args[i + 1]}") ;;
        esac
    done
    AGGREGATE_JOB=$(sbatch --parsable --dependency=afterok:"$ARRAY_JOB" \
        --job-name=inference_pareto_aggregate --partition=YOUR_PARTITION --time=01:00:00 \
        --cpus-per-task=4 --mem=48G --output=logs/inference_pareto_aggregate_%j.out \
        --wrap "source .venv/bin/activate && python src/inference_pareto.py aggregate ${AGGREGATE_ARGS[*]}")
    echo "Submitted array $ARRAY_JOB and aggregate $AGGREGATE_JOB"
    exit 0
fi

source .venv/bin/activate
python -u src/inference_pareto.py run "$@" \
    --dataset-index "$SLURM_ARRAY_TASK_ID" \
    --n-jobs "$SLURM_CPUS_PER_TASK"
