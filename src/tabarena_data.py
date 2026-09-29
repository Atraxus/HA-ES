"""Shared helpers for loading TabArena data and the per-config resource measurements."""

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
