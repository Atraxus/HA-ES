"""Inference time against performance of the ensembles found on two pruned pools.

Port of EnsembleGym's ``src/inference_pareto.py`` to HA-ES: classification only, ROC AUC, phem's
searches. It runs on the 9 classic methods and on the 12 foundation models, each pruned with
TabArena's ``EnsembleScorerMaxModels`` (``PARETO_POOLS`` in ``tabarena_data.py``).

An ensemble's inference time is the sum of its members' ``time_infer_s``. The Infer-QDO-ES variants
return a set of ensembles per task, so each task's set is reduced to its Pareto front on validation
error and inference time, and an entrant is a position along that front, from the fastest ensemble
(0) to the one with the best validation error (1). No test label is used to pick an ensemble.

GES, QO-ES and QDO-ES return one ensemble per task and are single points, as is each family's best
config on validation. TabArena's ``compare`` scores every entrant in one run with RandomForest's
default at 1000. Elo is relative to the entrants, so compare results only within one ``aggregate``.

``run`` handles one dataset and seed, so a SLURM array per seed covers the datasets
(``inference_pareto_job.sh``); ``aggregate`` merges them, scores and plots. Given several seeds, it
averages each entrant's error and time per task over the seeds and scores the averages in one run.
Run from the repository root.
"""

# First: generate_data sets the BLAS thread limits, which must happen before numpy is imported.
from generate_data import (
    _build_ensemble,
    _task_data,
    _warm_up_jit,
    compute_performance,
    load_and_process_base_models,
)

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fast_metrics import FastRocAuc
from plot_inference_pareto import (
    ALGORITHM_LABELS,
    BASELINE_LABELS,
    FAMILY,
    POOL_LABELS,
    mark_visible_families,
    plot_inference_pareto,
)
from tabarena_data import (
    DEFAULT_FOLDS,
    PARETO_POOLS,
    classification_datasets,
    get_context,
    initialize_tasks,
    load_pool_repo,
    pareto_method_metadata,
    pool_pruner,
    pruned_pool_configs,
)

RESULTS_ROOT = Path("results/inference_pareto")
ENSEMBLES_FILE = "ensembles.parquet"
FAMILIES_FILE = "families.parquet"
FRONT_POSITIONS = tuple(float(position) for position in np.linspace(0.0, 1.0, 9))
ENSEMBLE_STEPS = 40
METRIC_NAME = "roc_auc"
_GROUP_COLUMNS = ["pool", "algorithm", "dataset", "fold"]
_ENTRANT_KEYS = ["method", "dataset", "fold"]
_AVERAGED_COLUMNS = ["metric_error", "metric_error_val", "time_infer_s"]
_SUITE = "HA-ES"

# Each search's method id in generate_data._build_ensemble
SEARCH_METHODS = {
    "infer_qdo": "INFER_TIME_QDO",
    "infer_qdo_3_batches": "INFER_TIME_QDO_3_BATCHES",
    "infer_qdo_sparse_p20": "INFER_TIME_QDO_SPARSE_P20",
}
BASELINE_METHODS = {"ges": "GES", "qoes": "QO", "qdoes": "QDO"}
_METHODS = {**SEARCH_METHODS, **BASELINE_METHODS}

# RandomForest's default is the Elo reference, and it is that family's model in its pool.
REFERENCE_METHOD = "RandomForest"
REFERENCE_CONFIG = "RandomForest_c1_BAG_L1"
REFERENCE_NAME = "RF (default)"
# Where the plot's name differs from TabArena's display name.
_DISPLAY_NAMES = {"RealMLP_GPU": "RealMLP-GPU"}


def _fit_search(algorithm: str) -> list[dict[str, object]]:
    """Fit one search on the current task and return one row per ensemble it found."""
    d = _task_data
    ensemble = _build_ensemble(
        _METHODS[algorithm], d["base_models"], d["metric"], d["seed"], n_iterations=d["steps"]
    )
    ensemble.ensemble_fit(d["predictions_val"], d["y_val"])
    if algorithm in BASELINE_METHODS:
        weights = np.asarray(ensemble.weights_)[np.newaxis]
    else:
        weights = np.unique(np.array([elite.sol for elite in ensemble.archive]), axis=0)
    rows = []
    for index, vector in enumerate(weights):
        ensemble.weights_ = vector
        error_val, error_test = compute_performance(
            ensemble,
            d["metric"],
            d["predictions_val"],
            d["predictions_test"],
            d["y_val"],
            d["y_test"],
        )
        members = np.flatnonzero(vector)
        rows.append(
            {
                "algorithm": algorithm,
                "ensemble": index,
                "metric_error": float(error_test),
                "metric_error_val": float(error_val),
                "time_infer_s": float(d["times"][members].sum()),
                "n_members": len(members),
                "members": [d["base_models"][i].name for i in members],
                "weights": vector[members].tolist(),
            }
        )
    return rows


def family_rows(
    pool: str, repo, methods, metrics, dataset, fold, metric, y_val, y_test, problem_type
) -> list[dict[str, object]]:
    """Each family's best config on validation; RandomForest's default for RandomForest."""
    context = get_context()
    task_metrics = metrics.loc[(dataset, fold)]
    available = set(task_metrics.index)
    rows = []
    for method, member in zip(methods, repo.repos, strict=True):
        metadata = pareto_method_metadata(context, method)
        if method == REFERENCE_METHOD:
            candidates, name = [REFERENCE_CONFIG], REFERENCE_NAME
        else:
            candidates = sorted(set(member.configs()) & available)
            display = _DISPLAY_NAMES.get(method, metadata.display_name)
            name = f"{display} ({'tuned' if metadata.can_hpo else 'default'})"
        predictions = repo.predict_val_multi(
            dataset=dataset, fold=fold, configs=candidates, binary_as_multiclass=True
        )
        losses = [float(metric(y_val, prediction, to_loss=True)) for prediction in predictions]
        best = int(np.argmin(losses))
        test = repo.predict_test_multi(
            dataset=dataset, fold=fold, configs=[candidates[best]], binary_as_multiclass=True
        )[0]
        rows.append(
            {
                "pool": pool,
                "method": name,
                "dataset": dataset,
                "fold": fold,
                "metric_error": float(metric(y_test, test, to_loss=True)),
                "metric_error_val": losses[best],
                "problem_type": problem_type,
                "metric": METRIC_NAME,
                "imputed": False,
                "time_train_s": 0.0,
                "time_infer_s": float(task_metrics.loc[candidates[best], "time_infer_s"]),
                "ta_name": f"Pareto-family-{name}",
                "ta_suite": _SUITE,
            }
        )
    return rows


def evaluate_task(
    pool, repo, methods, pruner, metrics, task, *, seed, steps, n_jobs, warm_up
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Every ensemble the searches find on one task, and the task's family rows."""
    dataset = repo.task_to_dataset(task)
    fold = repo.task_to_fold(task)
    configs = pruned_pool_configs(pruner, metrics, dataset, fold)
    base_models, predictions_val, predictions_test = load_and_process_base_models(
        metrics, repo, dataset, fold, None, configs=configs
    )
    y_val = repo.labels_val(dataset=dataset, fold=fold)
    y_test = repo.labels_test(dataset=dataset, fold=fold)
    number_of_classes = predictions_val.shape[-1]
    metric = FastRocAuc(is_binary=number_of_classes == 2, labels=list(range(number_of_classes)))
    problem_type = repo.dataset_info(dataset=dataset)["problem_type"]

    if warm_up and n_jobs > 1:
        _warm_up_jit(base_models, predictions_val, y_val, metric)
    _task_data.update(
        base_models=base_models,
        predictions_val=predictions_val,
        predictions_test=predictions_test,
        y_val=y_val,
        y_test=y_test,
        metric=metric,
        seed=seed,
        steps=steps,
        times=np.array([model.model_metadata["test_predict_time"] for model in base_models]),
    )
    algorithms = list(_METHODS)
    try:
        if n_jobs == 1:
            per_search = [_fit_search(algorithm) for algorithm in algorithms]
        else:
            # Forked workers share the task data with the main process instead of copying it
            with ProcessPoolExecutor(
                max_workers=min(n_jobs, len(algorithms)),
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                per_search = list(executor.map(_fit_search, algorithms))
    finally:
        _task_data.clear()
    identity = {
        "pool": pool,
        "dataset": dataset,
        "fold": fold,
        "problem_type": problem_type,
        "metric": METRIC_NAME,
        "pool_size": len(configs),
    }
    ensembles = [{**identity, **row} for rows in per_search for row in rows]
    families = family_rows(
        pool, repo, methods, metrics, dataset, fold, metric, y_val, y_test, problem_type
    )
    return ensembles, families


def run(
    *,
    datasets: list[str] | None,
    dataset_index: int | None,
    folds: list[int],
    seed: int,
    steps: int,
    n_jobs: int,
    output_root: Path,
) -> None:
    """Find the ensembles of the selected datasets on both pools."""
    repos = {pool: load_pool_repo(methods) for pool, methods in PARETO_POOLS.items()}
    shared = sorted(set.intersection(*(set(classification_datasets(r)) for r in repos.values())))
    if dataset_index is not None:
        datasets = [shared[dataset_index]]
    elif datasets is None:
        datasets = shared
    for dataset in datasets:
        ensembles: list[dict[str, object]] = []
        families: list[dict[str, object]] = []
        for pool, methods in PARETO_POOLS.items():
            repo = repos[pool]
            metrics = repo.metrics(datasets=[dataset], folds=folds).sort_index()
            pruner = pool_pruner(repo)
            for i, task in enumerate(initialize_tasks(repo, [dataset], folds)):
                print(f"{pool}: {task}", flush=True)
                task_ensembles, task_families = evaluate_task(
                    pool,
                    repo,
                    methods,
                    pruner,
                    metrics,
                    task,
                    seed=seed,
                    steps=steps,
                    n_jobs=n_jobs,
                    warm_up=i == 0,
                )
                ensembles.extend(task_ensembles)
                families.extend(task_families)
        output_dir = output_root / f"seed_{seed}" / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ensembles).to_parquet(output_dir / ENSEMBLES_FILE, index=False)
        pd.DataFrame(families).to_parquet(output_dir / FAMILIES_FILE, index=False)


def validation_front(ensembles: pd.DataFrame) -> pd.DataFrame:
    """The ensembles no other beats on both validation error and time, fastest first."""
    ordered = ensembles.sort_values(["time_infer_s", "metric_error_val"], kind="stable")
    previous_best = ordered["metric_error_val"].cummin().shift(fill_value=np.inf)
    return ordered[ordered["metric_error_val"] < previous_best]


def entrant_name(pool: str, algorithm: str, position: float) -> str:
    return f"{POOL_LABELS[pool]} {ALGORITHM_LABELS[algorithm]} (front {position:.3f})"


def baseline_name(pool: str, algorithm: str) -> str:
    return f"{POOL_LABELS[pool]} {BASELINE_LABELS[algorithm]}"


def _entrant_row(chosen, method: str, pool: str, algorithm: str) -> dict[str, object]:
    return {
        "method": method,
        "dataset": chosen["dataset"],
        "fold": chosen["fold"],
        "metric_error": chosen["metric_error"],
        "metric_error_val": chosen["metric_error_val"],
        "problem_type": chosen["problem_type"],
        "metric": chosen["metric"],
        "imputed": False,
        "time_train_s": 0.0,
        "time_infer_s": chosen["time_infer_s"],
        "ta_name": f"Pareto-{pool}-{algorithm}",
        "ta_suite": _SUITE,
    }


def front_entrants(ensembles: pd.DataFrame, positions=FRONT_POSITIONS) -> pd.DataFrame:
    """TabArena's per-split rows of every (pool, search, front position) entrant."""
    rows = []
    for (pool, algorithm, _, _), group in ensembles.groupby(_GROUP_COLUMNS, sort=False):
        front = validation_front(group)
        for position in positions:
            chosen = front.iloc[int(np.floor(position * (len(front) - 1) + 0.5))]
            rows.append(_entrant_row(chosen, entrant_name(pool, algorithm, position), pool, algorithm))
    return pd.DataFrame(rows)


def baseline_entrants(ensembles: pd.DataFrame) -> pd.DataFrame:
    """TabArena's per-split rows of GES, QO-ES and QDO-ES, with one ensemble per task."""
    rows = [
        _entrant_row(row, baseline_name(row["pool"], row["algorithm"]), row["pool"], row["algorithm"])
        for _, row in ensembles.iterrows()
    ]
    return pd.DataFrame(rows)


def tabarena_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    """TabArena's leaderboard for per-split ``results``, indexed by method.

    RandomForest's default is the Elo reference at 1000, and every other setting is ``compare``'s
    default. bencheval's bootstrap depends on row order, so the rows are sorted first.
    """
    results = results.sort_values(["dataset", "fold", "method"], kind="stable", ignore_index=True)
    leaderboard = get_context().compare(
        output_dir=None,
        ta_results=results,
        plot=False,
        leaderboard_kwargs={"include_error": True},
    )
    return leaderboard.set_index("method")


def check_coverage(entrants: pd.DataFrame, expected: set[tuple[str, int]]) -> None:
    """Every entrant must have every task: TabArena would otherwise impute the gaps."""
    for method, group in entrants.groupby("method", sort=False):
        missing = expected - set(zip(group["dataset"], group["fold"], strict=True))
        if missing:
            raise ValueError(f"{method} misses {len(missing)} tasks: {sorted(missing)[:5]}")


def pareto_points(
    leaderboard: pd.DataFrame, searches: list[str], baselines: list[str], families: dict
) -> pd.DataFrame:
    """The plotted points: Elo and mean time of every entrant, the reference left out.

    A baseline has one ensemble per task and a family's model one config per task, so their
    ``position`` is NaN. RandomForest's default counts as a family in its pool.
    """
    keys = [
        (pool, algorithm, position, entrant_name(pool, algorithm, position))
        for pool in PARETO_POOLS
        for algorithm in searches
        for position in FRONT_POSITIONS
    ]
    keys += [
        (pool, algorithm, float("nan"), baseline_name(pool, algorithm))
        for pool in PARETO_POOLS
        for algorithm in baselines
    ]
    keys += [
        (pool, FAMILY, float("nan"), name)
        for pool, names in families.items()
        for name in names
    ]
    columns = [
        "elo",
        "elo+",
        "elo-",
        "metric_error",
        "time_infer_s_per_1K",
        "median_time_infer_s_per_1K",
        "median_time_infer_s",
    ]
    return pd.DataFrame(
        [
            {
                "pool": pool,
                "algorithm": algorithm,
                "position": position,
                "entrant": name,
                **leaderboard.loc[name, columns].to_dict(),
            }
            for pool, algorithm, position, name in keys
        ]
    )


def seed_entrants(seed: int, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One seed's per-split entrant rows and family rows, merged over its per-dataset jobs."""
    element_dirs = sorted(
        path.parent for path in (output_root / f"seed_{seed}").glob(f"*/{ENSEMBLES_FILE}")
    )
    if not element_dirs:
        raise FileNotFoundError(f"No {ENSEMBLES_FILE} under {output_root}/seed_{seed}")
    ensembles = pd.concat(
        [pd.read_parquet(d / ENSEMBLES_FILE) for d in element_dirs], ignore_index=True
    )
    family = pd.concat(
        [pd.read_parquet(d / FAMILIES_FILE) for d in element_dirs], ignore_index=True
    )
    is_search = ensembles["algorithm"].isin(SEARCH_METHODS)
    entrants = pd.concat(
        [
            front_entrants(ensembles[is_search]),
            baseline_entrants(ensembles[~is_search]),
            family.drop(columns="pool"),
        ],
        ignore_index=True,
    )
    reference = family[family["method"] == REFERENCE_NAME]
    check_coverage(entrants, set(zip(reference["dataset"], reference["fold"], strict=True)))
    return entrants, family


def average_over_seeds(per_seed: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per (entrant, task) with the mean error and time over the seeds.

    Each seed's fronts are built separately, so a front position averages the ensembles the seeds
    chose for it. Averaging rather than adding seeds as splits keeps the tasks ``compare`` sees.
    """
    tasks = [set(rows[_ENTRANT_KEYS].itertuples(index=False, name=None)) for rows in per_seed]
    for seed_tasks in tasks[1:]:
        if seed_tasks != tasks[0]:
            raise ValueError(
                f"Seeds cover different tasks, e.g. {sorted(seed_tasks ^ tasks[0])[:5]}"
            )
    merged = pd.concat(per_seed, ignore_index=True)
    how = {
        column: "mean" if column in _AVERAGED_COLUMNS else "first"
        for column in merged.columns
        if column not in _ENTRANT_KEYS
    }
    return merged.groupby(_ENTRANT_KEYS, sort=False, as_index=False).agg(how)


def aggregate(
    *, seeds: list[int], output_root: Path, data_dir: Path, plots_dir: Path, name: str | None
) -> pd.DataFrame:
    """Merge the per-dataset jobs of the seeds, score them once and plot."""
    per_seed, families = zip(*(seed_entrants(seed, output_root) for seed in seeds), strict=True)
    entrants = average_over_seeds(list(per_seed))
    family = families[0]

    leaderboard = tabarena_leaderboard(entrants)
    # Several seeds get their own files, so they do not replace a single seed's.
    if name is None and len(seeds) > 1:
        name = f"{len(seeds)}_seeds"
    suffix = f"_{name}" if name else ""
    data_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(data_dir / f"inference_pareto_leaderboard{suffix}.csv")
    points = pareto_points(
        leaderboard,
        [search for search in ALGORITHM_LABELS if search in SEARCH_METHODS],
        [baseline for baseline in BASELINE_LABELS if baseline in BASELINE_METHODS],
        {
            pool: sorted(family.loc[family["pool"] == pool, "method"].unique())
            for pool in PARETO_POOLS
        },
    )
    points = mark_visible_families(points)
    points.to_csv(data_dir / f"inference_pareto_points{suffix}.csv", index=False)
    plot_name = f"inference-pareto-mean-time{suffix.replace('_', '-')}.png"
    path = plot_inference_pareto(points, plots_dir / plot_name)
    print(f"Merged {entrants['dataset'].nunique()} datasets, {len(seeds)} seed(s); wrote {path}")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Find the ensembles on the selected datasets")
    run_parser.add_argument("--datasets", nargs="+", default=None)
    run_parser.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help="Run only the n-th classification dataset (for a SLURM array).",
    )
    run_parser.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--steps", type=int, default=ENSEMBLE_STEPS)
    run_parser.add_argument("--n-jobs", type=int, default=1)
    run_parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    aggregate_parser = commands.add_parser("aggregate", help="Merge, score and plot a finished run")
    aggregate_parser.add_argument(
        "--seed", type=int, nargs="+", default=[0], help="Several seeds are averaged per task."
    )
    aggregate_parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    aggregate_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    aggregate_parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    aggregate_parser.add_argument("--name", default=None, help="Suffix of the output files")
    args = parser.parse_args()
    if args.command == "run":
        run(
            datasets=args.datasets,
            dataset_index=args.dataset_index,
            folds=args.folds,
            seed=args.seed,
            steps=args.steps,
            n_jobs=args.n_jobs,
            output_root=args.output_root,
        )
    else:
        aggregate(
            seeds=args.seed,
            output_root=args.output_root,
            data_dir=args.data_dir,
            plots_dir=args.plots_dir,
            name=args.name,
        )


if __name__ == "__main__":
    main()
