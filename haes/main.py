from collections import Counter
from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorSpace
from tabrepo import load_repository, EvaluationRepository, get_context

from phem.methods.ensemble_selection import EnsembleSelection
from phem.methods.ensemble_selection.qdo.behavior_spaces import (
    get_bs_configspace_similarity_and_loss_correlation,
    get_bs_ensemble_size_and_loss_correlation,
)
from phem.application_utils.supported_metrics import msc
from phem.methods.ensemble_selection.qdo.qdo_es import QDOEnsembleSelection

from sklearn.metrics import roc_auc_score
from dataclasses import dataclass, field

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ribs.visualize import sliding_boundaries_archive_heatmap

import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ensemble evaluations with a specific seed."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for RNG initialization."
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


def expand_binary_predictions(predictions):
    # Calculate probabilities for the negative class (class 0)
    negative_class_probs = 1 - predictions
    # Stack the negative and positive class probabilities along a new dimension
    expanded_predictions = np.stack([negative_class_probs, predictions], axis=-1)
    return expanded_predictions


def get_custom_behavior_space_with_inference_time(
    max_possible_inference_time: float,
) -> BehaviorSpace:
    # Using ensemble size (an existing behavior function) and a custom behavior function to create a 2D behavior space.
    from phem.methods.ensemble_selection.qdo.behavior_functions.basic import (
        LossCorrelationMeasure,
    )
    from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorFunction

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


def ensemble_inference_time(input_metadata: list[dict]):
    """A custom behavior function.

    Some Notes:
        - The input_metadata here is the metadata for each base model in the ensemble. How this is called is defined in
            phem.methods.ensemble_selection.qdo.qdo_es.evaluate_single_solution.
        - The behavior function definition and arguments depends on its definition in the BehaviorFunction class (see below).
            For all options, see phem.methods.ensemble_selection.qdo.behavior_space.BehaviorFunction.
    """
    return sum([md["val_predict_time"] for md in input_metadata])


def plot_archive(qdo_es: QDOEnsembleSelection, name: str):
    # Plot Archive
    n_elites = len(list(qdo_es.archive))
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(qdo_es.archive, cmap="viridis", square=False)
    plt.title(f"Final Archive Heatmap (Validation Loss) for {n_elites} elites")

    ax = plt.gca()
    x_boundary = qdo_es.archive.boundaries[0]
    y_boundary = qdo_es.archive.boundaries[1]
    ax.vlines(
        x_boundary,
        qdo_es.archive.lower_bounds[1],
        qdo_es.archive.upper_bounds[1],
        color="k",
        linewidth=0.5,
        alpha=0.5,
    )
    ax.hlines(
        y_boundary,
        qdo_es.archive.lower_bounds[0],
        qdo_es.archive.upper_bounds[0],
        color="k",
        linewidth=1,
        alpha=0.5,
    )
    ax.set(xlabel="Diversity", ylabel="Inference Time")
    ax.set_xlim(
        min(x_boundary) * 0.95,
        max(x_boundary) * 1.05,
    )
    ax.set_ylim(
        min(y_boundary) * 0.95 - 0.0005,
        max(y_boundary) * 1.05,
    )

    plt.savefig(f"archive_plots/{name}.png", dpi=300)


def evaluate_ensemble(
    name: str,
    ensemble: EnsembleSelection,
    repo: EvaluationRepository,
    task: str,
    predictions_val: list[np.ndarray],
    predictions_test: list[np.ndarray],
    seed: int = 1,
):
    dataset = repo.task_to_dataset(task)
    fold = repo.task_to_fold(task)

    y_test = repo.labels_test(dataset=dataset, fold=fold)
    y_val = repo.labels_val(dataset=dataset, fold=fold)
    predictions_val = np.array(predictions_val)
    predictions_test = np.array(predictions_test)

    number_of_classes = int(
        repo.dataset_metadata(dataset=dataset)["NumberOfClasses"]
    )  # 0 for regression
    if number_of_classes == 2:
        predictions_val = expand_binary_predictions(predictions_val)
        predictions_test = expand_binary_predictions(predictions_test)

    for bm in ensemble.base_models:
        bm.switch_to_val_simulation()

    ensemble.ensemble_fit(predictions_val, y_val)

    performances = []
    if name == "GES":
        indices_so_far = []
        index_counts = Counter()
        for idx in ensemble.indices_:
            indices_so_far.append(idx)
            index_counts.update([idx])

            # Calculate weights based on occurrence of each index
            ensemble.weights_ = np.zeros(len(ensemble.base_models))
            for index, count in index_counts.items():
                ensemble.weights_[index] = count / len(indices_so_far)

            selected_indices = list(index_counts.keys())
            y_pred_val = ensemble.ensemble_predict_proba(
                predictions_val[selected_indices, :]
            )
            y_pred_test = ensemble.ensemble_predict_proba(
                predictions_test[selected_indices, :]
            )
            if number_of_classes == 2:
                y_pred_val = y_pred_val[:, 1]
                y_pred_test = y_pred_test[:, 1]

            roc_auc_val = roc_auc_score(
                y_val, y_pred_val, multi_class="ovr", average="macro"
            )
            roc_auc_test = roc_auc_score(
                y_test, y_pred_test, multi_class="ovr", average="macro"
            )

            perf_dict = {
                "iteration": len(indices_so_far),
                "roc_auc_val": roc_auc_val,
                "roc_auc_test": roc_auc_test,
                "models_used": [
                    ensemble.base_models[i].name for i in set(indices_so_far)
                ],
                "weights": [1 / len(indices_so_far)]
                * len(set(indices_so_far)),  # Uniform weights as example
            }
            performances.append(perf_dict)
    elif type(ensemble) == QDOEnsembleSelection:
        solutions = [np.array(e.sol) for e in ensemble.archive]
        unique_tuples = set(tuple(array) for array in solutions)
        unique_solutions = [np.array(t) for t in unique_tuples]
        meta = [e.meta for e in ensemble.archive]

        for i, solution in enumerate(unique_solutions):
            ensemble.weights_ = solution
            y_pred_val = ensemble.ensemble_predict_proba(predictions_val)
            y_pred_test = ensemble.ensemble_predict_proba(predictions_test)
            if number_of_classes == 2:
                y_pred_val = y_pred_val[:, 1]
                y_pred_test = y_pred_test[:, 1]

            roc_auc_val = roc_auc_score(
                y_val, y_pred_val, multi_class="ovr", average="macro"
            )
            roc_auc_test = roc_auc_score(
                y_test, y_pred_test, multi_class="ovr", average="macro"
            )

            weight_indices = np.where(ensemble.weights_ != 0)[0]
            perf_dict = {
                "name": f"{name}_{i}",
                "roc_auc_val": roc_auc_val,
                "roc_auc_test": roc_auc_test,
                "models_used": [ensemble.base_models[i].name for i in weight_indices],
                "weights": ensemble.weights_[weight_indices],
                "meta": meta[i],
            }
            performances.append(perf_dict)
    else:
        pass

    # Create a DataFrame from the list of performance dictionaries
    performance_df = pd.DataFrame(performances)
    performance_df["task"] = task
    performance_df["dataset"] = dataset
    performance_df["fold"] = fold
    performance_df["method"] = name
    if not os.path.exists(f"results/seed_{seed}"):
        os.makedirs(f"results/seed_{seed}")
    performance_df.to_json(f"results/seed_{seed}/{name}_{task}.json")


def load_data(repo, context_name) -> tuple[list[str], list[str], list[int], dict]:
    context = get_context(name=context_name)
    all_config_hyperparameters = context.load_configs_hyperparameters()
    datasets = repo.datasets()
    configs = repo.configs()
    folds = [0, 1, 2]
    return datasets, configs, folds, all_config_hyperparameters


def initialize_tasks(repo, datasets, folds) -> list[str]:
    tasks = [
        repo.task_name(dataset=dataset, fold=fold)
        for dataset in datasets
        for fold in folds
    ]
    return tasks


def load_and_process_base_models(
    metrics, repo, task, configs, all_config_hyperparameters
):
    dataset = repo.task_to_dataset(task)
    fold = repo.task_to_fold(task)

    # Iterate over each config to create a base model representation
    base_models = []
    predictions_val = []
    predictions_test = []
    for config in configs:
        try:
            time_infer_s = metrics.loc[(dataset, fold, config), "time_infer_s"]
            time_train_s = metrics.loc[(dataset, fold, config), "time_train_s"]
        except KeyError as e:
            #! Not all configs have entries; is it ok to assume previous value is good enough?
            print(f"Error accessing data: {e}")

        config_key = config.rsplit("_BAG_L1", 1)[
            0
        ]  # ? Why do they all end with _BAG_L1?
        config_type = all_config_hyperparameters[config_key]["model_type"]
        config_hyperparameters = all_config_hyperparameters[config_key][
            "hyperparameters"
        ]
        config_dict = {}
        for key, value in config_hyperparameters.items():
            try:
                config_dict[key] = float(value)
            except (ValueError, TypeError):
                continue

        config_dict["model_type"] = config_type

        # Fetch predictions for each dataset and fold
        predictions_val.append(
            repo.predict_val(dataset=dataset, fold=fold, config=config)
        )
        predictions_test.append(
            repo.predict_test(dataset=dataset, fold=fold, config=config)
        )

        # Wrap predictions in the FakedFittedAndValidatedClassificationBaseModel
        model = FakedFittedAndValidatedClassificationBaseModel(
            name=config,
            val_probabilities=predictions_val[-1],
            test_probabilities=predictions_test[-1],
            model_metadata={
                "fit_time": time_train_s,
                "test_predict_time": time_infer_s,
                "val_predict_time": time_infer_s,  #!
                "config": config_dict,
                "auto-sklearn-model": "PLACEHOLDER",  #! What to pick here as equivalent? Is it even used later on?
            },
        )

        base_models.append(model)

    return base_models, predictions_val, predictions_test


def evaluate_single_best_model(
    ensemble: list[FakedFittedAndValidatedClassificationBaseModel],
    repo: EvaluationRepository,
    task: str,
    seed: int = 1,
):
    dataset = repo.task_to_dataset(task)
    fold = repo.task_to_fold(task)
    y_val = repo.labels_val(dataset=dataset, fold=fold)
    y_test = repo.labels_test(dataset=dataset, fold=fold)

    # Determine which model performs best on the validation data
    best_score = 0
    best_model = None
    for model in ensemble:
        model.switch_to_val_simulation()  # Ensure the model returns validation predictions
        predicted_probs = model.predict_proba(None)  # Since predictions are pre-stored, no need to pass X
        score = roc_auc_score(y_val, predicted_probs, multi_class="ovr", average="macro")
        if score > best_score:
            best_score = score
            best_model = model

    # Now evaluate the best model on test data
    best_model.switch_to_test_simulation()  # Switch to test predictions
    predicted_probs_test = best_model.predict_proba(None)  # No X needed as predictions are pre-stored
    test_score = roc_auc_score(y_test, predicted_probs_test, multi_class="ovr", average="macro")

    # Also on validation data
    best_model.switch_to_val_simulation()
    predicted_probs_val = best_model.predict_proba(None)
    val_score = roc_auc_score(y_val, predicted_probs_val, multi_class="ovr", average="macro")

    # Creating performance dictionary and saving to DataFrame
    performance_dict = {
        "name": best_model.name,
        "roc_auc_val": val_score,
        "roc_auc_test": test_score,
        "task_id": task.split('_')[0],
        "fold": fold,
        "method": "SINGLE_BEST"
    }
    
    performance_df = pd.DataFrame([performance_dict])  # Convert dict to DataFrame
    performance_df["dataset"] = dataset
    performance_df["method"] = "SINGLE_BEST"

    # Check and create directory if needed
    result_path = f"results/seed_{seed}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    # Save DataFrame to JSON
    performance_df.to_json(f"{result_path}/SINGLE_BEST_{task}.json")

    # Optionally print out results for verification
    print(f"Best Model {best_model.name} ROC AUC Test Score for {task}: {test_score}")
    print(f"Best Model {best_model.name} ROC AUC Validation Score for {task}: {val_score}")



def main(
    random_seed: int = 0,
    run_singleBest: bool = False,
    run_ges: bool = False,
    run_qo: bool = False,
    run_qdo: bool = False,
    run_infer_time_qdo: bool = False,
    run_ens_size_qdo: bool = False,
):
    # Define the context for the ensemble evaluation
    context_name = "D244_F3_C1530_100"
    # Load the repository with the specified context
    repo: EvaluationRepository = load_repository(context_name, cache=True)
    # Load the data
    datasets, configs, folds, all_config_hyperparameters = load_data(repo, context_name)
    # A task is a fold of a dataset
    tasks = initialize_tasks(repo, datasets, folds)

    metrics = repo.metrics(datasets=datasets, folds=folds, configs=configs)

    # Evaluate ensemble selection methods for each task
    current_time = time.time()
    for i, task in enumerate(tasks):
        print(
            f"Task {i+1}/{len(tasks)}: {task}, time for last task: {time.time() - current_time:.2f} s"
        )
        current_time = time.time()
        base_models, predictions_val, predictions_test = load_and_process_base_models(
            metrics, repo, task, configs, all_config_hyperparameters
        )

        # Adjusting the metric based on the task
        dataset = repo.task_to_dataset(task)
        task_type = repo.dataset_metadata(dataset=dataset)["task_type"]
        number_of_classes = int(
            repo.dataset_metadata(dataset=dataset)["NumberOfClasses"]
        )  # 0 for regression
        if task_type == "Supervised Classification":
            is_binary = number_of_classes == 2
            metric_name = "roc_auc"
            labels = list(range(number_of_classes))
        elif task_type == "Supervised Regression":
            continue  # Skip regression tasks for now
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Single best model evaluation
        if run_singleBest:
            evaluate_single_best_model(base_models, repo, task, seed=random_seed)

        # GES evaluation
        if run_ges:
            ges = EnsembleSelection(
                base_models=base_models,
                n_iterations=100,
                metric=msc(metric_name=metric_name, is_binary=is_binary, labels=labels),
                random_state=random_seed,
            )
            evaluate_ensemble(
                "GES",
                ges,
                repo,
                task,
                predictions_val,
                predictions_test,
                seed=random_seed,
            )

        # QO evaluation
        if run_qo:
            qo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=msc(
                    metric_name=metric_name, is_binary=is_binary, labels=labels
                ),
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
                seed=random_seed,
            )

        # QDO evaluation
        if run_qdo:
            qdo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=msc(
                    metric_name=metric_name, is_binary=is_binary, labels=labels
                ),
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
                score_metric=msc(
                    metric_name=metric_name, is_binary=is_binary, labels=labels
                ),
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
                seed=random_seed,
            )

        # QDO evaluation with ensemble size and loss correlation
        if run_ens_size_qdo:
            ens_size_qdo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=msc(
                    metric_name=metric_name, is_binary=is_binary, labels=labels
                ),
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
                seed=random_seed,
            )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.seed,
        run_singleBest=True,
        run_ges=False,
        run_qo=False,
        run_qdo=False,
        run_infer_time_qdo=False,
        run_ens_size_qdo=False,
    )
