from collections import Counter
from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorSpace
from tabarena.repository import EvaluationRepositoryCollection

from phem.methods.ensemble_selection import EnsembleSelection
from phem.methods.ensemble_selection.qdo.behavior_spaces import (
    get_bs_configspace_similarity_and_loss_correlation,
    get_bs_ensemble_size_and_loss_correlation,
)
from phem.application_utils.supported_metrics import msc
from phem.methods.ensemble_selection.qdo.qdo_es import QDOEnsembleSelection
from phem.methods.ensemble_selection.qdo.behavior_functions.basic import (
    LossCorrelationMeasure,
)
from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorFunction
from phem.base_utils.metrics import AbstractMetric

from dataclasses import dataclass, field

import os
import numpy as np
import pandas as pd

import time
import argparse

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
        num_solutions = 20
        infer_time_weights = np.linspace(0, 1, num=num_solutions)
        for time_weight in infer_time_weights:
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
    y_pred_val = ensemble.ensemble_predict_proba(predictions_val)
    y_pred_test = ensemble.ensemble_predict_proba(predictions_test)
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


def main(
    random_seed: int = 0,
    methods: list[str] | None = None,
    folds: list[int] = DEFAULT_FOLDS,
    datasets: list[str] | None = None,
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

    # Evaluate ensemble selection methods for each task
    current_time = time.time()
    for i, task in enumerate(tasks):
        print(
            f"Task {i+1}/{len(tasks)}: {task}, time for last task: {time.time() - current_time:.2f} s"
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
        metric = msc(
            metric_name="roc_auc", is_binary=(number_of_classes == 2), labels=labels
        )

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

        # GES evaluation
        if run_ges:
            ges = EnsembleSelection(
                base_models=base_models,
                n_iterations=100,
                metric=metric,
                random_state=random_seed,
            )
            evaluate_ensemble(
                "GES",
                ges,
                repo,
                task,
                predictions_val,
                predictions_test,
                y_val,
                y_test,
                metric,
                seed=random_seed,
            )
        # Multi-GES evaluation
        if run_multi_ges and not all(
            result_file_exists(f"MULTI_GES-{weight:.2f}")
            for weight in np.linspace(0, 1, 20)
        ):
            multi_ges = EnsembleSelection(
                base_models=base_models,
                n_iterations=100,
                metric=metric,
                random_state=random_seed,
            )
            evaluate_ensemble(
                "MULTI_GES",
                multi_ges,
                repo,
                task,
                predictions_val,
                predictions_test,
                y_val,
                y_test,
                metric,
                seed=random_seed,
            )
        # QO evaluation
        if run_qo:
            qo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=metric,
                random_state=random_seed,
                archive_type="quality",
            )
            evaluate_ensemble(
                "QO",
                qo,
                repo,
                task,
                predictions_val,
                predictions_test,
                y_val,
                y_test,
                metric,
                seed=random_seed,
            )

        # QDO evaluation
        if run_qdo:
            qdo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=metric,
                random_state=random_seed,
                behavior_space=get_bs_configspace_similarity_and_loss_correlation(),
            )
            evaluate_ensemble(
                "QDO",
                qdo,
                repo,
                task,
                predictions_val,
                predictions_test,
                y_val,
                y_test,
                metric,
                seed=random_seed,
            )

        # QDO evaluation with inference time and loss correlation
        if run_infer_time_qdo:
            max_possible_ensemble_infer_time = sum(
                [bm.model_metadata["test_predict_time"] for bm in base_models],
            )
            infer_time_qdo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=metric,
                random_state=random_seed,
                behavior_space=get_custom_behavior_space_with_inference_time(
                    max_possible_ensemble_infer_time
                ),
                base_models_metadata_type="custom",
            )
            evaluate_ensemble(
                "INFER_TIME_QDO",
                infer_time_qdo,
                repo,
                task,
                predictions_val,
                predictions_test,
                y_val,
                y_test,
                metric,
                seed=random_seed,
            )

        # QDO evaluation with ensemble size and loss correlation
        if run_ens_size_qdo:
            ens_size_qdo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=metric,
                random_state=random_seed,
                behavior_space=get_bs_ensemble_size_and_loss_correlation(),
            )
            evaluate_ensemble(
                "ENS_SIZE_QDO",
                ens_size_qdo,
                repo,
                task,
                predictions_val,
                predictions_test,
                y_val,
                y_test,
                metric,
                seed=random_seed,
            )

        # QDO evaluation with memory usage and loss correlation
        if run_memory_qdo:
            max_possible_ensemble_memory_usage = sum(
                [bm.model_metadata["memory"] for bm in base_models],
            )
            memory_qdo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=metric,
                random_state=random_seed,
                behavior_space=get_custom_behavior_space_with_memory_usage(
                    max_possible_ensemble_memory_usage
                ),
                base_models_metadata_type="custom",
            )
            evaluate_ensemble(
                "MEMORY_QDO",
                memory_qdo,
                repo,
                task,
                predictions_val,
                predictions_test,
                y_val,
                y_test,
                metric,
                seed=random_seed,
            )

        # QDO evaluation with disk usage and loss correlation
        if run_disk_qdo:
            max_possible_ensemble_disk_usage = sum(
                [bm.model_metadata["diskspace"] for bm in base_models],
            )
            disk_qdo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=metric,
                random_state=random_seed,
                behavior_space=get_custom_behavior_space_with_disk_usage(
                    max_possible_ensemble_disk_usage
                ),
                base_models_metadata_type="custom",
            )
            evaluate_ensemble(
                "DISK_QDO",
                disk_qdo,
                repo,
                task,
                predictions_val,
                predictions_test,
                y_val,
                y_test,
                metric,
                seed=random_seed,
            )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.seed,
        methods=args.methods,
        folds=args.folds,
        datasets=args.datasets,
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
