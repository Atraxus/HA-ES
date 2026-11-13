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

``run`` handles one dataset and repeat per SLURM array element (``inference_pareto_job.sh``) and
fits every seed on each task, so a task's pruning, predictions and family rows are loaded once for
all seeds. ``aggregate`` merges the elements, scores and plots. Given several seeds, it averages
each entrant's error and time per task over the seeds and scores the averages in one run.
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
    FOLDS_PER_REPEAT,
    PARETO_POOLS,
    array_element,
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
# Positions plotted per front, evenly spaced from the fastest ensemble to the best on validation
FRONT_POINTS = 3
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


def _fit_search(job: tuple[int, str]) -> list[dict[str, object]]:
    """Fit one search with one seed on the current task and return one row per ensemble it found."""
    seed, algorithm = job
    d = _task_data
    ensemble = _build_ensemble(
        _METHODS[algorithm], d["base_models"], d["metric"], seed, n_iterations=d["steps"]
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
                "seed": seed,
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
    pool, repo, methods, pruner, metrics, task, *, seeds, steps, n_jobs, warm_up
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Every ensemble the searches find on one task with each of `seeds` (column ``seed``), and
    the task's family rows, which do not depend on the seed."""
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
        steps=steps,
        times=np.array([model.model_metadata["test_predict_time"] for model in base_models]),
    )
    jobs = [(seed, algorithm) for algorithm in _METHODS for seed in seeds]
    try:
        if n_jobs == 1:
            per_search = [_fit_search(job) for job in jobs]
        else:
            # Forked workers share the task data with the main process instead of copying it
            with ProcessPoolExecutor(
                max_workers=min(n_jobs, len(jobs)),
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                per_search = list(executor.map(_fit_search, jobs))
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


def element_dir(output_root: Path, seed: int, dataset: str, folds: list[int]) -> Path:
    """Where ``run`` writes the results of one seed on one repeat of a dataset."""
    return output_root / f"seed_{seed}" / dataset / f"repeat_{folds[0] // FOLDS_PER_REPEAT}"


def run(
    *,
    datasets: list[str] | None,
    array_index: int | None,
    folds: list[int],
    seeds: list[int],
    steps: int,
    n_jobs: int,
    output_root: Path,
) -> None:
    """Find the ensembles of the selected datasets on both pools, per dataset and repeat.

    Seeds whose results of a repeat exist are skipped, and a repeat with none left is not loaded.
    """
    repos = {pool: load_pool_repo(methods) for pool, methods in PARETO_POOLS.items()}
    shared = sorted(set.intersection(*(set(classification_datasets(r)) for r in repos.values())))
    if array_index is not None:
        dataset, element_folds, _ = array_element(datasets or shared, folds, array_index)
        elements = [(dataset, element_folds)]
    else:
        repeats = sorted({fold // FOLDS_PER_REPEAT for fold in folds})
        elements = [
            (dataset, [f for f in folds if f // FOLDS_PER_REPEAT == repeat])
            for dataset in datasets or shared
            for repeat in repeats
        ]
    for dataset, element_folds in elements:
        pending = [
            seed
            for seed in seeds
            if not (element_dir(output_root, seed, dataset, element_folds) / FAMILIES_FILE).exists()
        ]
        if not pending:
            print(f"All seeds of {dataset}, folds {element_folds} exist", flush=True)
            continue
        ensembles: list[dict[str, object]] = []
        families: list[dict[str, object]] = []
        for pool, methods in PARETO_POOLS.items():
            repo = repos[pool]
            metrics = repo.metrics(datasets=[dataset], folds=element_folds).sort_index()
            pruner = pool_pruner(repo)
            for i, task in enumerate(initialize_tasks(repo, [dataset], element_folds)):
                print(f"{pool}: {task}, seeds {pending}", flush=True)
                task_ensembles, task_families = evaluate_task(
                    pool,
                    repo,
                    methods,
                    pruner,
                    metrics,
                    task,
                    seeds=pending,
                    steps=steps,
                    n_jobs=n_jobs,
                    warm_up=i == 0,
                )
                ensembles.extend(task_ensembles)
                families.extend(task_families)
        if not families:
            # A repeat the dataset does not have
            continue
        ensembles_df = pd.DataFrame(ensembles)
        for seed in pending:
            output_dir = element_dir(output_root, seed, dataset, element_folds)
            output_dir.mkdir(parents=True, exist_ok=True)
            seed_ensembles = ensembles_df[ensembles_df["seed"] == seed].drop(columns="seed")
            seed_ensembles.to_parquet(output_dir / ENSEMBLES_FILE, index=False)
            # Written last: its existence marks the seed as done
            pd.DataFrame(families).to_parquet(output_dir / FAMILIES_FILE, index=False)


def validation_front(ensembles: pd.DataFrame) -> pd.DataFrame:
    """The ensembles no other beats on both validation error and time, fastest first."""
    ordered = ensembles.sort_values(["time_infer_s", "metric_error_val"], kind="stable")
    previous_best = ordered["metric_error_val"].cummin().shift(fill_value=np.inf)
    return ordered[ordered["metric_error_val"] < previous_best]


def front_positions(n_points: int = FRONT_POINTS) -> tuple[float, ...]:
    """`n_points` evenly spaced positions along a front, from 0 (fastest) to 1 (best validation)."""
    if n_points < 2:
        raise ValueError(f"A front needs at least 2 points, got {n_points}")
    return tuple(float(position) for position in np.linspace(0.0, 1.0, n_points))


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


def front_entrants(ensembles: pd.DataFrame, positions: tuple[float, ...]) -> pd.DataFrame:
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
    leaderboard: pd.DataFrame,
    searches: list[str],
    baselines: list[str],
    families: dict,
    positions: tuple[float, ...],
) -> pd.DataFrame:
    """The plotted points: Elo and mean time of every entrant, the reference left out.

    A baseline has one ensemble per task and a family's model one config per task, so their
    ``position`` is NaN. RandomForest's default counts as a family in its pool.
    """
    keys = [
        (pool, algorithm, position, entrant_name(pool, algorithm, position))
        for pool in PARETO_POOLS
        for algorithm in searches
        for position in positions
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


def seed_entrants(
    seed: int, output_root: Path, positions: tuple[float, ...], searches: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One seed's per-split entrant rows and family rows, merged over its array elements.

    Only the searches in `searches` enter, with every baseline and family.
    """
    element_dirs = sorted(
        path.parent for path in (output_root / f"seed_{seed}").rglob(FAMILIES_FILE)
    )
    if not element_dirs:
        raise FileNotFoundError(f"No {FAMILIES_FILE} under {output_root}/seed_{seed}")
    ensembles = pd.concat(
        [pd.read_parquet(d / ENSEMBLES_FILE) for d in element_dirs], ignore_index=True
    )
    family = pd.concat(
        [pd.read_parquet(d / FAMILIES_FILE) for d in element_dirs], ignore_index=True
    )
    duplicated = family[family.duplicated(["pool", "method", "dataset", "fold"])]
    if not duplicated.empty:
        raise ValueError(
            f"Tasks stored by several elements under {output_root}/seed_{seed}, e.g. "
            f"{sorted(set(zip(duplicated['dataset'], duplicated['fold'])))[:5]}; results of the "
            "earlier per-dataset layout (<dataset>/ instead of <dataset>/repeat_<r>/) may remain"
        )
    is_search = ensembles["algorithm"].isin(SEARCH_METHODS)
    is_entered = ensembles["algorithm"].isin(searches)
    entrants = pd.concat(
        [
            front_entrants(ensembles[is_search & is_entered], positions),
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
    *,
    seeds: list[int],
    output_root: Path,
    data_dir: Path,
    plots_dir: Path,
    name: str | None,
    front_points: int = FRONT_POINTS,
    searches: list[str] | None = None,
) -> pd.DataFrame:
    """Merge the per-dataset jobs of the seeds, score them once and plot.

    Each search enters with `front_points` positions along its fronts; fewer entrants change the
    others' Elo, so compare results only between runs with the same number. Likewise, only the
    searches in `searches` (by default all of ``SEARCH_METHODS``) enter.
    """
    positions = front_positions(front_points)
    searches = [
        search
        for search in ALGORITHM_LABELS
        if search in SEARCH_METHODS and (searches is None or search in searches)
    ]
    per_seed, families = zip(
        *(seed_entrants(seed, output_root, positions, searches) for seed in seeds), strict=True
    )
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
        searches,
        [baseline for baseline in BASELINE_LABELS if baseline in BASELINE_METHODS],
        {
            pool: sorted(family.loc[family["pool"] == pool, "method"].unique())
            for pool in PARETO_POOLS
        },
        positions,
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
        "--array-index",
        type=int,
        default=None,
        help=(
            "Run only one SLURM array element: one dataset (of --datasets, by default the shared "
            "classification datasets) and one repeat of --folds, indexed by dataset, then repeat."
        ),
    )
    run_parser.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS)
    run_parser.add_argument(
        "--seed", type=int, nargs="+", default=[0], help="Each task is loaded once for all seeds."
    )
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
    aggregate_parser.add_argument(
        "--front-points",
        type=int,
        default=FRONT_POINTS,
        help="Positions per search front, evenly spaced from the fastest ensemble to the best on "
        "validation. Elo is relative to the entrants, so this changes every Elo.",
    )
    aggregate_parser.add_argument(
        "--searches",
        nargs="+",
        choices=list(SEARCH_METHODS),
        default=None,
        help="Searches that enter, by default all. The others are left out of scoring and plot.",
    )
    args = parser.parse_args()
    if args.command == "run":
        run(
            datasets=args.datasets,
            array_index=args.array_index,
            folds=args.folds,
            seeds=args.seed,
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
            front_points=args.front_points,
            searches=args.searches,
        )


if __name__ == "__main__":
    main()
