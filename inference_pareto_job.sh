#!/bin/bash

#SBATCH --job-name=inference_pareto
#SBATCH --output=logs/inference_pareto_%A_%a.out
#SBATCH --error=logs/inference_pareto_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --partition=small

# Inference time against test Elo on the pruned cross-family and foundation-model pools
# (src/inference_pareto.py). One array element per classification dataset and repeat of --folds
# fits every seed, so each task is loaded and pruned once for all seeds; then one job merges them
# and draws the plot. Start it from the project root, not with sbatch (set SBATCH_PARTITION for
# another partition than small):
#   bash inference_pareto_job.sh [--seed S [S ...]] [--folds 0 1 2] [--steps 40]
# Several seeds are averaged per task by the aggregate job. Elements skip the seeds whose results
# exist, so resubmitting reruns only what is missing.
# Outputs: results/inference_pareto/seed_<S>/<dataset>/repeat_<R>/, data/inference_pareto_*.csv
# and plots/inference-pareto-mean-time*.png. Needs `uv sync --group notebooks`.
set -eo pipefail

# Classification datasets shared by both pools; TabArena splits a dataset into repeats of 3 folds
N_DATASETS=38
FOLDS_PER_REPEAT=3

# The TabArena store is too large for the home quota
export TABARENA_CACHE=${TABARENA_CACHE:-/scratch/htc/$USER/tabarena}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    mkdir -p logs
    SEEDS=()
    FOLDS=(0 1 2)
    RUN_ARGS=()
    AGGREGATE_ARGS=()
    while (($#)); do
        case "$1" in
            --seed | --folds)
                option=$1
                values=()
                shift
                while (($#)) && [[ "$1" != --* ]]; do
                    values+=("$1")
                    shift
                done
                if [ "$option" = --seed ]; then SEEDS=("${values[@]}"); else FOLDS=("${values[@]}"); fi
                ;;
            --output-root)
                RUN_ARGS+=("$1" "$2")
                AGGREGATE_ARGS+=("$1" "$2")
                shift 2
                ;;
            *)
                RUN_ARGS+=("$1")
                shift
                ;;
        esac
    done
    [ "${#SEEDS[@]}" -gt 0 ] || SEEDS=(0)
    N_REPEATS=$(for fold in "${FOLDS[@]}"; do echo $((fold / FOLDS_PER_REPEAT)); done | sort -u | wc -l)
    ARRAY_JOB=$(sbatch --parsable --array=0-$((N_DATASETS * N_REPEATS - 1)) "$0" \
        --seed "${SEEDS[@]}" --folds "${FOLDS[@]}" "${RUN_ARGS[@]}")
    AGGREGATE_JOB=$(sbatch --parsable --dependency=afterok:"$ARRAY_JOB" \
        --job-name=inference_pareto_aggregate --partition="${SBATCH_PARTITION:-small}" --time=01:00:00 \
        --cpus-per-task=17 --mem=48G --output=logs/inference_pareto_aggregate_%j.out \
        --wrap "source .venv/bin/activate && python src/inference_pareto.py aggregate --seed ${SEEDS[*]} ${AGGREGATE_ARGS[*]}")
    echo "Submitted array $ARRAY_JOB ($((N_DATASETS * N_REPEATS)) elements) and aggregate $AGGREGATE_JOB"
    exit 0
fi

source .venv/bin/activate
python -u src/inference_pareto.py run "$@" \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --n-jobs "$SLURM_CPUS_PER_TASK"
