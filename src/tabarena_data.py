"""Shared helpers for loading TabArena data and the per-config resource measurements."""

import numbers
import os
import warnings

import pandas as pd
from tabarena.contexts.tabarena.context import TabArenaContext
from tabarena.repository import EvaluationRepositoryCollection

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

# TabArena numbers the splits of a dataset (repeats x folds) as folds; 0-2 are the folds of the first repeat.
DEFAULT_FOLDS = [0, 1, 2]

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
