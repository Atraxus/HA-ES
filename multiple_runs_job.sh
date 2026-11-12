#!/bin/bash

# Submit single_run_job.sh as one array: one element per classification dataset and repeat of
# --folds, each fitting every seed, so a task's predictions are read once for all seeds.
# --split-folds gives each fold its own element and --seeds-per-job N splits the seeds into
# groups of N, for shorter jobs. Run from the project root (not with sbatch); the arguments are
# passed to src/generate_data.py:
#   bash multiple_runs_job.sh [--seed 0 1 ... 9] [--folds 0 1 2] [--split-folds] [--seeds-per-job N]
# Set the time limit with e.g. SBATCH_TIMELIMIT=06:00:00. Elements skip the results that exist,
# so resubmitting reruns only what is missing.
set -eo pipefail

# Classification datasets of the default methods; TabArena splits a dataset into repeats of 3 folds
N_DATASETS=38
FOLDS_PER_REPEAT=3

SEEDS=(0 1 2 3 4 5 6 7 8 9)
FOLDS=(0 1 2)
SPLIT_FOLDS=0
SEEDS_PER_JOB=0
ARGS=()
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
        --split-folds)
            SPLIT_FOLDS=1
            ARGS+=("$1")
            shift
            ;;
        --seeds-per-job)
            SEEDS_PER_JOB=$2
            ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p logs
if ((SPLIT_FOLDS)); then
    N_FOLD_GROUPS=${#FOLDS[@]}
else
    N_FOLD_GROUPS=$(for fold in "${FOLDS[@]}"; do echo $((fold / FOLDS_PER_REPEAT)); done | sort -u | wc -l)
fi
((SEEDS_PER_JOB > 0)) || SEEDS_PER_JOB=${#SEEDS[@]}
N_SEED_GROUPS=$(((${#SEEDS[@]} + SEEDS_PER_JOB - 1) / SEEDS_PER_JOB))
sbatch --array=0-$((N_DATASETS * N_FOLD_GROUPS * N_SEED_GROUPS - 1)) single_run_job.sh \
    --seed "${SEEDS[@]}" --folds "${FOLDS[@]}" "${ARGS[@]}"
