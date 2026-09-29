import os

# The ensembles of a task are fitted in parallel worker processes (see --n-jobs), so each process
# must not start its own BLAS/OpenMP thread pool. Set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorSpace
from tabarena.repository import EvaluationRepositoryCollection

from phem.methods.ensemble_selection import EnsembleSelection
from phem.methods.ensemble_selection.qdo.behavior_spaces import (
    get_bs_configspace_similarity_and_loss_correlation,
    get_bs_ensemble_size_and_loss_correlation,
)
from phem.methods.ensemble_selection.qdo.qdo_es import QDOEnsembleSelection
from phem.methods.ensemble_selection.qdo.behavior_functions.basic import (
    LossCorrelationMeasure,
)
from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorFunction
from phem.base_utils.metrics import AbstractMetric

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import time
import argparse

from fast_metrics import FastRocAuc
from tabarena_data import (
    CLASSIFICATION_PROBLEM_TYPES,
    DEFAULT_FOLDS,
    DEFAULT_METHODS,
    classification_datasets,
    config_resource_usage,
    initialize_tasks,
    load_repo,
    load_resource_usage,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ensemble evaluations with a specific seed."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for RNG initialization."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="TabArena methods whose configs are used as base models.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=DEFAULT_FOLDS,
        help="TabArena folds (splits) to evaluate per dataset.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of ensembles fitted in parallel per task (default: all available CPUs).",
    )
    parser.add_argument(
        "--legacy-multi-ges",
        action="store_true",
        help=(
            "Reproduce the Multi-GES results of the publications: fit all weights with one "
            "ensemble object, which limits each weight to the best iteration of the previous one "
            "instead of running 100 iterations per weight."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Restrict to these datasets (default: all classification datasets).",
    )
    return parser.parse_args()


@dataclass
class FakedFittedAndValidatedClassificationBaseModel:
    """Fake sklearn-like base model (classifier) usable by ensembles in the same way as real base models.

    To simulate validation and test predictions, we start by default with returning validation predictions.
    Then, after fitting the ensemble on the validation predictions, we switch to returning test predictions using `switch_to_test_simulation`.

    Parameters
    ----------
    name: str
        Name of the base model.
    val_probabilities: list[np.ndarray]
        The predictions of the base model on the validation data.
    test_probabilities: list[np.ndarray]
        The predictions of the base model on the test data.
    return_val_data : bool, default=True
        If True, the val_probabilities are returned. If False, the test_probabilities are returned.
    """

    name: str
    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
    return_val_data: bool = True
    model_metadata: dict = field(default_factory=dict)

    @property
    def probabilities(self):
        if self.return_val_data:
            return self.val_probabilities

        return self.test_probabilities

    def predict(self, X):
        return np.argmax(self.probabilities, axis=1)

    def predict_proba(self, X):
        return self.probabilities

    def switch_to_test_simulation(self):
        self.return_val_data = False

    def switch_to_val_simulation(self):
        self.return_val_data = True


def ensemble_inference_time(input_metadata: list[dict]):
    """A custom behavior function.

    Some Notes:
        - The input_metadata here is the metadata for each base model in the ensemble. How this is called is defined in
            phem.methods.ensemble_selection.qdo.qdo_es.evaluate_single_solution.
        - The behavior function definition and arguments depends on its definition in the BehaviorFunction class (see below).
            For all options, see phem.methods.ensemble_selection.qdo.behavior_space.BehaviorFunction.
    """
    return sum([md["val_predict_time"] for md in input_metadata])


def ensemble_memory_usage(input_metadata: list[dict]):
    """A custom behavior function for memory usage.

    Some Notes:
        - The input_metadata here is the metadata for each base model in the ensemble. How this is called is defined in
            phem.methods.ensemble_selection.qdo.qdo_es.evaluate_single_solution.
        - The behavior function definition and arguments depend on its definition in the BehaviorFunction class.
    """
    return sum([md["memory"] for md in input_metadata])


def ensemble_disk_usage(input_metadata: list[dict]):
    """A custom behavior function for disk space usage.

    Some Notes:
        - The input_metadata here is the metadata for each base model in the ensemble. How this is called is defined in
            phem.methods.ensemble_selection.qdo.qdo_es.evaluate_single_solution.
        - The behavior function definition and arguments depend on its definition in the BehaviorFunction class.
    """
    return sum([md["diskspace"] for md in input_metadata])


def get_custom_behavior_space_with_inference_time(
    max_possible_inference_time: float,
) -> BehaviorSpace:
    # Using ensemble size (an existing behavior function) and a custom behavior function to create a 2D behavior space.

    EnsembleInferenceTime = BehaviorFunction(
        ensemble_inference_time,  # function to call.
        # define the required arguments for the function `ensemble_inference_time`
        required_arguments=["input_metadata"],
        # Define the initial starting range of the behavior space (due to using a sliding boundaries archive, this will be re-mapped anyhow)
        range_tuple=(0, max_possible_inference_time + 1),  # +1 for safety.
        # Defines which kind of prediction data is needed as input (if  any)
        required_prediction_format="none",
        name="Ensemble Inference Time",
    )

    return BehaviorSpace([LossCorrelationMeasure, EnsembleInferenceTime])


def get_custom_behavior_space_with_memory_usage(
    max_possible_memory_usage: float,
) -> BehaviorSpace:
    # Using ensemble size (an existing behavior function) and a custom behavior function to create a 2D behavior space.

    EnsembleMemoryUsage = BehaviorFunction(
        ensemble_memory_usage,  # function to call.
        # define the required arguments for the function `ensemble_memory_usage`
        required_arguments=["input_metadata"],
        # Define the initial starting range of the behavior space (due to using a sliding boundaries archive, this will be re-mapped anyhow)
        range_tuple=(0, max_possible_memory_usage + 1),  # +1 for safety.
        # Defines which kind of prediction data is needed as input (if any)
        required_prediction_format="none",
        name="Ensemble Memory Usage",
    )

    return BehaviorSpace([LossCorrelationMeasure, EnsembleMemoryUsage])


def get_custom_behavior_space_with_disk_usage(
    max_possible_disk_usage: float,
) -> BehaviorSpace:
    # Using ensemble size (an existing behavior function) and a custom behavior function to create a 2D behavior space.

    EnsembleDiskUsage = BehaviorFunction(
        ensemble_disk_usage,  # function to call.
        # define the required arguments for the function `ensemble_disk_usage`
        required_arguments=["input_metadata"],
        # Define the initial starting range of the behavior space (due to using a sliding boundaries archive, this will be re-mapped anyhow)
        range_tuple=(0, max_possible_disk_usage + 1),  # +1 for safety.
        # Defines which kind of prediction data is needed as input (if any)
        required_prediction_format="none",
        name="Ensemble Disk Usage",
    )

    return BehaviorSpace([LossCorrelationMeasure, EnsembleDiskUsage])


# Static inference time weights of Multi-GES
MULTI_GES_TIME_WEIGHTS = np.linspace(0, 1, num=20)


def evaluate_ensemble(
    name: str,
    ensemble: EnsembleSelection,
    repo: EvaluationRepositoryCollection,
    task: str,
    predictions_val: list[np.ndarray],
    predictions_test: list[np.ndarray],
    y_val,
    y_test,
    metric: AbstractMetric,
    seed: int = 1,
    time_weight: float | None = None,
):
    for bm in ensemble.base_models:
        bm.switch_to_val_simulation()

    if name == "GES":
        # Ensure correct weights to avoid pollution from other tests
        ensemble.time_weight = 0.0
        ensemble.loss_weight = 1.0
        ensemble.ensemble_fit(predictions_val, y_val)
        performances = process_ges_iterations(
            ensemble,
            predictions_val,
            predictions_test,
            y_val,
            y_test,
            metric,
            name_prefix="GES",
        )
        save_performances(
            performances,
            task,
            repo,
            name,
            seed,
        )
    elif name == "MULTI_GES":
        if time_weight is None:
            # Legacy behavior: all weights are fitted one after another with the same ensemble
            # object. Each fit reduces the object's number of iterations to its best iteration
            # (phem's use_best), so every weight is limited by the fits of the previous weights.
            time_weights = MULTI_GES_TIME_WEIGHTS
        else:
            time_weights = [time_weight]
        for time_weight in time_weights:
            ensemble.time_weight = time_weight
            ensemble.loss_weight = 1 - time_weight
            ensemble.ensemble_fit(predictions_val, y_val)

            performances = process_ges_iterations(
                ensemble,
                predictions_val,
                predictions_test,
                y_val,
                y_test,
                metric,
                name_prefix=f"MULTI_GES_{time_weight:.2f}",
            )

            save_performances(
                performances,
                task,
                repo,
                name,
                seed,
                filename_suffix=f"-{time_weight:.2f}",
            )
    elif isinstance(ensemble, QDOEnsembleSelection):
        ensemble.ensemble_fit(predictions_val, y_val)
        performances = process_qdo_ensemble(
            ensemble,
            predictions_val,
            predictions_test,
            y_val,
            y_test,
            name,
            metric,
        )
        save_performances(
            performances,
            task,
            repo,
            name,
            seed,
        )
    else:
        pass


def process_ges_iterations(
    ensemble,
    predictions_val,
    predictions_test,
    y_val,
    y_test,
    metric: AbstractMetric,
    name_prefix,
    time_weight=None,
):
    indices_so_far = []
    index_counts = Counter()
    performances = []

    for idx in ensemble.indices_:
        indices_so_far.append(idx)
        index_counts.update([idx])

        # Calculate weights based on occurrence of each index
        ensemble.weights_ = np.zeros(len(ensemble.base_models))
        for index, count in index_counts.items():
            ensemble.weights_[index] = count / len(indices_so_far)

        # Compute performance
        roc_auc_val, roc_auc_test = compute_performance(
            ensemble,
            metric,
            predictions_val,
            predictions_test,
            y_val,
            y_test,
        )

        # Prepare performance dictionary
        perf_dict = {
            "name": f"{name_prefix}_{len(indices_so_far)}",
            "iteration": len(indices_so_far),
            "roc_auc_val": roc_auc_val,
            "roc_auc_test": roc_auc_test,
            "models_used": [ensemble.base_models[i].name for i in index_counts.keys()],
            "weights": [ensemble.weights_[i] for i in index_counts.keys()],
        }
        if time_weight is not None:
            perf_dict["time_weight"] = time_weight
        performances.append(perf_dict)

    return performances


def process_qdo_ensemble(
    ensemble,
    predictions_val,
    predictions_test,
    y_val,
    y_test,
    name,
    metric: AbstractMetric,
):
    solutions = [np.array(e.sol) for e in ensemble.archive]
    unique_solutions = {tuple(sol) for sol in solutions}
    performances = []
    for i, solution in enumerate(unique_solutions):
        ensemble.weights_ = np.array(solution)
        roc_auc_val, roc_auc_test = compute_performance(
            ensemble,
            metric,
            predictions_val,
            predictions_test,
            y_val,
            y_test,
        )

        weight_indices = np.where(ensemble.weights_ != 0)[0]
        perf_dict = {
            "name": f"{name}_{i}",
            "roc_auc_val": roc_auc_val,
            "roc_auc_test": roc_auc_test,
            "models_used": [ensemble.base_models[i].name for i in weight_indices],
            "weights": ensemble.weights_[weight_indices],
        }
        performances.append(perf_dict)
    return performances


def compute_performance(
    ensemble, metric: AbstractMetric, predictions_val, predictions_test, y_val, y_test
):
    # Only pass the predictions of the models in the ensemble; phem then skips the (identical)
    # contributions of the zero-weight models instead of iterating over all base models
    used = np.flatnonzero(ensemble.weights_)
    y_pred_val = ensemble.ensemble_predict_proba(predictions_val[used])
    y_pred_test = ensemble.ensemble_predict_proba(predictions_test[used])
    roc_auc_val = metric(y_val, y_pred_val, to_loss=True)
    roc_auc_test = metric(y_test, y_pred_test, to_loss=True)
    return roc_auc_val, roc_auc_test


def save_performances(
    performances, task, repo: EvaluationRepositoryCollection, name, seed, filename_suffix=""
):
    performance_df = pd.DataFrame(performances)
    performance_df["task"] = task
    performance_df["dataset"] = repo.task_to_dataset(task)
    performance_df["fold"] = repo.task_to_fold(task)
    performance_df["method"] = name
    if not os.path.exists(f"results/seed_{seed}"):
        os.makedirs(f"results/seed_{seed}")
    filename = f"results/seed_{seed}/{name}{filename_suffix}_{task}.json"
    performance_df.to_json(filename)


def load_and_process_base_models(
    metrics, repo: EvaluationRepositoryCollection, dataset, fold, df_usage
):
    # Only use the configs that have results for this task
    task_metrics = metrics.loc[(dataset, fold)]
    configs = list(task_metrics.index)

    # Binary predictions are returned as (n_configs, n_rows, 2) like the multiclass ones
    predictions_val = repo.predict_val_multi(
        dataset=dataset, fold=fold, configs=configs, binary_as_multiclass=True
    )
    predictions_test = repo.predict_test_multi(
        dataset=dataset, fold=fold, configs=configs, binary_as_multiclass=True
    )

    # Iterate over each config to create a base model representation
    base_models = []
    for i, config in enumerate(configs):
        time_infer_s = task_metrics.loc[config, "time_infer_s"]
        time_train_s = task_metrics.loc[config, "time_train_s"]

        config_type = repo.config_type(config=config)
        config_hyperparameters = (
            repo.config_hyperparameters(config=config, include_ag_args=False) or {}
        )
        config_dict = {}
        for key, value in config_hyperparameters.items():
            try:
                config_dict[key] = float(value)
            except (ValueError, TypeError):
                continue

        config_dict["model_type"] = config_type

        memory_used, disk_space_used = config_resource_usage(
            df_usage, config, config_type
        )

        # Wrap predictions in the FakedFittedAndValidatedClassificationBaseModel
        model = FakedFittedAndValidatedClassificationBaseModel(
            name=config,
            val_probabilities=predictions_val[i],
            test_probabilities=predictions_test[i],
            model_metadata={
                "fit_time": time_train_s,
                "test_predict_time": time_infer_s,
                "val_predict_time": time_infer_s,
                "memory": memory_used,
                "diskspace": disk_space_used,
                "config": config_dict,
                "auto-sklearn-model": "PLACEHOLDER",
            },
        )

        base_models.append(model)

    return base_models, predictions_val, predictions_test


def evaluate_single_best_model(
    base_models: list[FakedFittedAndValidatedClassificationBaseModel],
    repo: EvaluationRepositoryCollection,
    task: str,
    metric: AbstractMetric,
    predictions_val,
    predictions_test,
    y_val,
    y_test,
    seed: int = 1,
):
    dataset = repo.task_to_dataset(task)
    fold = repo.task_to_fold(task)

    # Initialize best_score to positive infinity since lower loss is better
    best_score = np.inf
    best_model = None
    best_idx = -1  # Keep track of the index

    for idx in range(len(base_models)):
        score = metric(y_val, predictions_val[idx], to_loss=True)
        if score < best_score:
            best_score = score
            best_model = base_models[idx]
            best_idx = idx  # Update the best index

    # Use the best index to get the test score
    test_score = metric(y_test, predictions_test[best_idx], to_loss=True)

    # Creating performance dictionary and saving to DataFrame
    performance_dict = {
        "name": "SINGLE_BEST",
        "roc_auc_val": best_score,
        "roc_auc_test": test_score,
        "task_id": task.split("_")[0],
        "fold": fold,
        "models_used": [best_model.name],
        "weights": [1.0],
        "method": "SINGLE_BEST",
    }

    performance_df = pd.DataFrame([performance_dict])
    performance_df["dataset"] = dataset
    performance_df["method"] = "SINGLE_BEST"
    performance_df["task"] = task
    performance_df["fold"] = fold

    # Check and create directory if needed
    result_path = f"results/seed_{seed}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Save DataFrame to JSON
    performance_df.to_json(f"{result_path}/SINGLE_BEST_{task}.json")


# Data of the current task, set before the worker processes are forked so that they share it
_task_data: dict = {}


def _build_ensemble(method: str, base_models, metric, random_seed: int):
    """Create the ensemble method `method`. phem's own multiprocessing is disabled (n_jobs=1)
    because the ensembles of a task are already fitted in parallel."""
    if method in ("GES", "MULTI_GES"):
        return EnsembleSelection(
            base_models=base_models,
            n_iterations=100,
            metric=metric,
            random_state=random_seed,
            n_jobs=1,
        )

    qdo_kwargs = dict(
        base_models=base_models,
        n_iterations=3,
        score_metric=metric,
        random_state=random_seed,
        n_jobs=1,
    )
    if method == "QO":
        return QDOEnsembleSelection(archive_type="quality", **qdo_kwargs)
    if method == "QDO":
        return QDOEnsembleSelection(
            behavior_space=get_bs_configspace_similarity_and_loss_correlation(),
            **qdo_kwargs,
        )
    if method == "ENS_SIZE_QDO":
        return QDOEnsembleSelection(
            behavior_space=get_bs_ensemble_size_and_loss_correlation(), **qdo_kwargs
        )

    # QDO with a hardware cost metric and loss correlation
    cost_behavior_spaces = {
        "INFER_TIME_QDO": ("test_predict_time", get_custom_behavior_space_with_inference_time),
        "MEMORY_QDO": ("memory", get_custom_behavior_space_with_memory_usage),
        "DISK_QDO": ("diskspace", get_custom_behavior_space_with_disk_usage),
    }
    metadata_key, get_behavior_space = cost_behavior_spaces[method]
    max_possible_ensemble_cost = sum(bm.model_metadata[metadata_key] for bm in base_models)
    return QDOEnsembleSelection(
        behavior_space=get_behavior_space(max_possible_ensemble_cost),
        base_models_metadata_type="custom",
        **qdo_kwargs,
    )


def _run_job(job: tuple[str, float | None]):
    method, time_weight = job
    d = _task_data
    ensemble = _build_ensemble(method, d["base_models"], d["metric"], d["seed"])
    evaluate_ensemble(
        method,
        ensemble,
        d["repo"],
        d["task"],
        d["predictions_val"],
        d["predictions_test"],
        d["y_val"],
        d["y_test"],
        d["metric"],
        seed=d["seed"],
        time_weight=time_weight,
    )
    return method if time_weight is None else f"{method}-{time_weight:.2f}"


def _warm_up_jit(base_models, predictions_val, y_val, metric):
    """Compile the numba functions of the QDO archives once in the main process.

    Forked workers inherit the compiled functions; otherwise every worker compiles them again for
    every task (~3 s per QDO fit). Uses a small fit on a subset of the base models with both
    archive types used by the experiments.
    """
    n = min(len(base_models), 20)
    kwargs = dict(
        base_models=base_models[:n], n_iterations=3, score_metric=metric, random_state=0, n_jobs=1
    )
    QDOEnsembleSelection(archive_type="quality", **kwargs).ensemble_fit(predictions_val[:n], y_val)
    QDOEnsembleSelection(
        behavior_space=get_bs_ensemble_size_and_loss_correlation(), **kwargs
    ).ensemble_fit(predictions_val[:n], y_val)


def main(
    random_seed: int = 0,
    methods: list[str] | None = None,
    folds: list[int] = DEFAULT_FOLDS,
    datasets: list[str] | None = None,
    n_jobs: int = 1,
    legacy_multi_ges: bool = False,
    run_singleBest: bool = False,
    run_multi_ges: bool = False,
    run_ges: bool = False,
    run_qo: bool = False,
    run_qdo: bool = False,
    run_infer_time_qdo: bool = False,
    run_ens_size_qdo: bool = False,
    run_memory_qdo: bool = False,
    run_disk_qdo: bool = False,
):
    repo = load_repo(methods)
    if datasets is None:
        datasets = classification_datasets(repo)
    # A task is a fold of a dataset
    tasks = initialize_tasks(repo, datasets, folds)

    metrics = repo.metrics(datasets=datasets, folds=folds)
    df_usage = load_resource_usage()

    def result_file_exists(method_name, extra_info=""):
        file_path = f"results/seed_{random_seed}/{method_name}{extra_info}_{task}.json"
        return os.path.exists(file_path)

    # Ensemble methods to fit per task: (method, Multi-GES time weight). GES first, as it takes longest.
    jobs = []
    if run_ges:
        jobs.append(("GES", None))
    if run_multi_ges:
        if legacy_multi_ges:
            # One job fitting all weights sequentially (time_weight=None)
            jobs.append(("MULTI_GES", None))
        else:
            jobs += [("MULTI_GES", time_weight) for time_weight in MULTI_GES_TIME_WEIGHTS]
    for method, run in [
        ("QO", run_qo),
        ("QDO", run_qdo),
        ("INFER_TIME_QDO", run_infer_time_qdo),
        ("ENS_SIZE_QDO", run_ens_size_qdo),
        ("MEMORY_QDO", run_memory_qdo),
        ("DISK_QDO", run_disk_qdo),
    ]:
        if run:
            jobs.append((method, None))

    jit_warmed_up = False

    # Evaluate ensemble selection methods for each task
    current_time = time.time()
    for i, task in enumerate(tasks):
        print(
            f"Task {i+1}/{len(tasks)}: {task}, time for last task: {time.time() - current_time:.2f} s",
            flush=True,
        )
        current_time = time.time()

        dataset = repo.task_to_dataset(task)
        fold = repo.task_to_fold(task)

        if repo.dataset_info(dataset=dataset)["problem_type"] not in CLASSIFICATION_PROBLEM_TYPES:
            continue  # Only support classification for now

        base_models, predictions_val, predictions_test = load_and_process_base_models(
            metrics, repo, dataset, fold, df_usage
        )
        y_test = repo.labels_test(dataset=dataset, fold=fold)
        y_val = repo.labels_val(dataset=dataset, fold=fold)

        # Adjusting the metric based on the task
        number_of_classes = predictions_val.shape[-1]
        labels = list(range(number_of_classes))
        metric = FastRocAuc(is_binary=(number_of_classes == 2), labels=labels)

        # Single best model evaluation
        if run_singleBest:
            evaluate_single_best_model(
                base_models,
                repo,
                task,
                metric,
                predictions_val,
                predictions_test,
                y_val,
                y_test,
                seed=random_seed,
            )

        # Multi-GES is skipped if all its results already exist
        task_jobs = [
            job
            for job in jobs
            if not (
                job[0] == "MULTI_GES"
                and all(result_file_exists(f"MULTI_GES-{w:.2f}") for w in MULTI_GES_TIME_WEIGHTS)
            )
        ]
        if not task_jobs:
            continue

        if not jit_warmed_up and n_jobs > 1:
            _warm_up_jit(base_models, predictions_val, y_val, metric)
            jit_warmed_up = True

        _task_data.update(
            repo=repo,
            task=task,
            base_models=base_models,
            predictions_val=predictions_val,
            predictions_test=predictions_test,
            y_val=y_val,
            y_test=y_test,
            metric=metric,
            seed=random_seed,
        )
        if n_jobs == 1:
            for job in task_jobs:
                _run_job(job)
        else:
            # Forked workers share the task data with the main process instead of copying it
            with ProcessPoolExecutor(
                max_workers=min(n_jobs, len(task_jobs)),
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                # Iterating the results re-raises exceptions from the workers
                for _ in executor.map(_run_job, task_jobs):
                    pass
        _task_data.clear()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.seed,
        methods=args.methods,
        folds=args.folds,
        datasets=args.datasets,
        n_jobs=args.n_jobs,
        legacy_multi_ges=args.legacy_multi_ges,
        run_singleBest=True,
        run_ges=True,
        run_multi_ges=True,
        run_qo=True,
        run_qdo=True,
        run_infer_time_qdo=True,
        run_ens_size_qdo=True,
        run_memory_qdo=True,
        run_disk_qdo=True,
    )
