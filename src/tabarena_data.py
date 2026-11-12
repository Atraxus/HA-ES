"""Shared helpers for loading TabArena data and the per-config resource measurements."""

import dataclasses
import numbers
import os
import warnings

import pandas as pd
from tabarena.contexts.tabarena.context import TabArenaContext
from tabarena.repository import EvaluationRepositoryCollection
from tabarena.simulation.ensemble_selection_config_scorer import EnsembleScorerMaxModels

# CPU config methods from TabArena whose base models are ensembled. Each method is a model family
# with its own hyperparameter configs (e.g. "LightGBM_c1_BAG_L1", "LightGBM_r1_BAG_L1", ...).
DEFAULT_METHODS = [
    "CatBoost",
    "ExplainableBM",
    "ExtraTrees",
    "KNeighbors",
    "LightGBM",
    "LinearModel",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    "XGBoost",
]

# The pools of the inference-time Pareto comparison (src/inference_pareto.py).
PARETO_POOLS = {
    "cross_family": [
        "CatBoost",
        "ExtraTrees",
        "LightGBM",
        "NeuralNetFastAI",
        "NeuralNetTorch",
        "RandomForest",
        "RealMLP_GPU",
        "TabM",
        "XGBoost",
    ],
    "across_foundation_models": [
        "EXAONE-Tabular",
        "LimiX",
        "RealTabPFN-v2.5",
        "SAP-RPT-OSS",
        "TabDPT-Turbo",
        "TabDPT_GPU",
        "TabFM",
        "TabICLv2",
        "TabPFN-3",
        "TabPFN-v2.6",
        "TabSTAR",
        "TabSwift",
    ],
}

# The pinned TabArena registers these methods under tabarena-2026-09-16. The Pareto comparison uses
# the artifacts of the suites below; left as registered, load_repo would download other ones.
PARETO_METHOD_SUITES = {
    "EXAONE-Tabular": "tabarena-2026-08-06",
    "TabFM": "tabarena-2026-07-13",
    "TabICLv2": "tabarena-2026-07-13",
    "TabPFN-3": "tabarena-2026-07-13",
    "TabSwift": "tabarena-2026-07-13",
}

# TabArena numbers the splits of a dataset (repeats x folds) as folds; 0-2 are the folds of the first repeat.
DEFAULT_FOLDS = [0, 1, 2]
FOLDS_PER_REPEAT = 3

CLASSIFICATION_PROBLEM_TYPES = ("binary", "multiclass")

RESOURCE_USAGE_PATH = "data/model_memory_and_disk_usage.csv"

# Keys TabArena adds to configs for AutoGluon bookkeeping (e.g. "ag.verbosity").
# `include_ag_args=False` only drops "ag_args".
AG_BOOKKEEPING_PREFIXES = ("ag_args", "ag.")


def get_context() -> TabArenaContext:
    return TabArenaContext(backend="native")


def load_repo(methods: list[str] | None = None) -> EvaluationRepositoryCollection:
    """Load the processed TabArena artifacts of `methods`, downloading missing ones (several GB each)."""
    if methods is None:
        methods = DEFAULT_METHODS
    return get_context().load_repo(methods=methods)


def pareto_method_metadata(context: TabArenaContext, method: str):
    """TabArena's metadata of `method`, moved to the suite the Pareto comparison uses."""
    metadata = context.method_metadata(method=method)
    suite = PARETO_METHOD_SUITES.get(method)
    return metadata if suite is None else dataclasses.replace(metadata, suite=suite)


def load_pool_repo(methods: list[str]) -> EvaluationRepositoryCollection:
    """Load one Pareto pool's methods as one collection, `repo.repos` in the order of `methods`."""
    context = get_context()
    return context.load_repo(methods=[pareto_method_metadata(context, m) for m in methods])


def pool_pruner(repo: EvaluationRepositoryCollection) -> EnsembleScorerMaxModels:
    """TabArena's ensemble scorer, used only for its `filter_models` pruning.

    "auto" is AutoGluon's rule: 1 to 12 configs per method, by dataset size.
    """
    return EnsembleScorerMaxModels(
        repo=repo, task_metrics_metadata={}, max_models_per_type="auto"
    )


def pruned_pool_configs(pruner, metrics: pd.DataFrame, dataset: str, fold: int) -> list[str]:
    """The task's configs that the pruning keeps, in sorted order."""
    configs = sorted(metrics.loc[(dataset, fold)].index)
    kept = set(pruner.filter_models(dataset=dataset, fold=fold, models=configs))
    return [config for config in configs if config in kept]


def classification_datasets(repo) -> list[str]:
    return repo.datasets(problem_type=list(CLASSIFICATION_PROBLEM_TYPES))


def initialize_tasks(repo, datasets: list[str], folds: list[int]) -> list[str]:
    """A task is a fold of a dataset. Folds that a dataset does not have are skipped."""
    return [
        repo.task_name(dataset=dataset, fold=fold)
        for dataset in datasets
        for fold in folds
        if fold in repo.dataset_to_folds(dataset)
    ]


def array_element(
    datasets: list[str],
    folds: list[int],
    index: int,
    *,
    seeds: list[int] = (),
    split_folds: bool = False,
    seeds_per_job: int | None = None,
) -> tuple[str, list[int], list[int]]:
    """The dataset, folds and seeds of SLURM array element `index`.

    By default an element is one dataset and one repeat with all `seeds`; `split_folds` gives each
    fold its own element and `seeds_per_job` splits the seeds into groups of that size. Elements
    are ordered by dataset, then fold group, then seed group. A repeat or fold the dataset does not
    have leaves the element without tasks (TabArena's small datasets have 10 repeats, the others
    3). Every element loads the repo's metadata and labels once, and a task's predictions are read
    once per element that holds the task, however many of its seeds that element fits.
    """
    if split_folds:
        fold_groups = [[fold] for fold in folds]
    else:
        repeats = sorted({fold // FOLDS_PER_REPEAT for fold in folds})
        fold_groups = [[f for f in folds if f // FOLDS_PER_REPEAT == r] for r in repeats]
    seeds = list(seeds)
    size = seeds_per_job or max(len(seeds), 1)
    seed_groups = [seeds[i : i + size] for i in range(0, len(seeds), size)] or [[]]
    n_elements = len(datasets) * len(fold_groups) * len(seed_groups)
    if not 0 <= index < n_elements:
        raise IndexError(
            f"Array index {index} out of range for {len(datasets)} datasets x "
            f"{len(fold_groups)} fold groups x {len(seed_groups)} seed groups"
        )
    index, seed_group = divmod(index, len(seed_groups))
    dataset, fold_group = divmod(index, len(fold_groups))
    return datasets[dataset], fold_groups[fold_group], seed_groups[seed_group]


def config_metadata(config_type: str, hyperparameters: dict) -> dict:
    """The config dict compared by phem's config space Gower similarity (QDO).

    Like phem's Auto-sklearn configs: keys are namespaced as "<config_type>:<key>", "model_type"
    names the family, numbers are numeric and everything else (strings, lists, None) is
    categorical. Configs of different families then share only "model_type", so their similarity
    is 0, and each numeric range covers one family. See docs/qdo_config_metadata.md.
    """
    metadata = {"model_type": config_type}
    for key, value in hyperparameters.items():
        if key.startswith(AG_BOOKKEEPING_PREFIXES):
            continue
        numeric = isinstance(value, numbers.Number)
        metadata[f"{config_type}:{key}"] = float(value) if numeric else str(value)
    return metadata


def categorical_if_mixed(configs: list[dict]) -> list[dict]:
    """Make keys whose values mix numbers and strings (e.g. max_features 0.5 or "sqrt")
    categorical, as phem would otherwise subtract a string from a number."""
    mixed = {
        key
        for key in {key for config in configs for key in config}
        if len({isinstance(config[key], str) for config in configs if key in config}) > 1
    }
    return [
        {key: str(value) if key in mixed else value for key, value in config.items()}
        for config in configs
    ]


def load_resource_usage(path: str = RESOURCE_USAGE_PATH) -> pd.DataFrame:
    """Per-config memory and disk measurements produced by `src/config_stats.py`, indexed by config."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python src/config_stats.py` to measure the memory "
            "and disk usage of the TabArena configs."
        )
    return pd.read_csv(path).set_index("Model")


def config_resource_usage(
    df_usage: pd.DataFrame, config: str, config_type: str | None
) -> tuple[float, float]:
    """Return (inference memory, model size) of `config`.

    Configs without a measurement fall back to the median of the configs of the same type.
    """
    if config in df_usage.index:
        row = df_usage.loc[config]
    else:
        same_type = df_usage[df_usage["Model_Type"] == config_type]
        if same_type.empty:
            raise KeyError(
                f"No resource usage measured for config '{config}' or any config of type "
                f"'{config_type}'. Re-run `src/config_stats.py` for this method."
            )
        warnings.warn(
            f"No resource usage measured for config '{config}', using the median of "
            f"type '{config_type}'."
        )
        row = same_type[["Inference_Memory_Usage", "Models_Size"]].median()
    return float(row["Inference_Memory_Usage"]), float(row["Models_Size"])
