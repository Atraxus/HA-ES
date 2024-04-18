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

import numpy as np


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


def evaluate_ensemble(
    name: str,
    ensemble: EnsembleSelection,
    repo: EvaluationRepository,
    task: str,
    predictions_val: list[np.ndarray],
    predictions_test: list[np.ndarray],
):
    dataset = repo.task_to_dataset(task)
    fold = repo.task_to_fold(task)

    y_val = repo.labels_val(dataset=dataset, fold=fold)
    # Format (n_configs, n_samples, n_classes)
    predictions_val = np.array(predictions_val)
    predictions_test = np.array(predictions_test)

    number_of_classes = int(
        repo.dataset_metadata(dataset=dataset)["NumberOfClasses"]
    )  # 0 for regression
    if number_of_classes == 2:
        predictions_val = expand_binary_predictions(predictions_val)
        predictions_test = expand_binary_predictions(predictions_test)

    # Switch to simulating predictions on validation data  (i.e., the training data of the ensemble)
    for bm in ensemble.base_models:
        bm.switch_to_val_simulation()

    print(f"Fitting {name} for task {task}, dataset {dataset}, fold {fold}")
    ensemble.ensemble_fit(predictions_val, y_val)

    # Output the ROC AUC score on the validation data
    y_pred = ensemble.ensemble_predict_proba(predictions_val)
    if number_of_classes == 2:
        y_pred = y_pred[:, 1]
    print(
        f"\tROC AUC Validation Score for {task}: {roc_auc_score(y_val, y_pred, multi_class='ovr', average='macro')}"
    )

    # Simulate to simulating predictions on test data  (i.e., the test data of the ensemble and base models)
    for bm in ensemble.base_models:
        bm.switch_to_test_simulation()

    y_pred = ensemble.ensemble_predict_proba(predictions_test)
    if number_of_classes == 2:
        y_pred = y_pred[:, 1]
    y_test = repo.labels_test(dataset=dataset, fold=fold)
    print(
        f"\tROC AUC Test Score for {task}: {roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro')}"
    )
    if name == "GES":
        print(
            f"\tNumber of different base models in the ensemble: {len(set(ensemble.indices_))}"
        )
        models_used = [ensemble.base_models[i].name for i in set(ensemble.indices_)]
        print(f"\tModels used: {models_used}")
    elif type(ensemble) == QDOEnsembleSelection:
        weight_indices = np.where(ensemble.weights_ != 0)[0]
        print(
            f"\tNumber of different base models in the ensemble: {len(weight_indices)}"
        )
        models_used = [ensemble.base_models[i].name for i in weight_indices]
        print(f"\tModels used: {models_used}")
    else:
        pass


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


def load_and_process_base_models(repo, task, configs, all_config_hyperparameters):
    dataset = repo.task_to_dataset(task)
    fold = repo.task_to_fold(task)

    # Iterate over each config to create a base model representation
    base_models = []
    predictions_val = []
    predictions_test = []
    for config in configs:
        # Fetch metadata
        metrics = repo.metrics(datasets=[dataset], configs=[config])
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
        predicted_probs = model.predict_proba(
            None
        )  # Since predictions are pre-stored, no need to pass X
        score = roc_auc_score(
            y_val, predicted_probs, multi_class="ovr", average="macro"
        )
        if score > best_score:
            best_score = score
            best_model = model

    # Now evaluate the best model on test data
    best_model.switch_to_test_simulation()  # Switch to test predictions
    predicted_probs_test = best_model.predict_proba(
        None
    )  # No X needed as predictions are pre-stored
    test_score = roc_auc_score(
        y_test, predicted_probs_test, multi_class="ovr", average="macro"
    )
    print(f"Best Model {best_model.name} ROC AUC Test Score for {task}: {test_score}")

    # Also on validation data
    best_model.switch_to_val_simulation()
    predicted_probs_val = best_model.predict_proba(None)
    val_score = roc_auc_score(
        y_val, predicted_probs_val, multi_class="ovr", average="macro"
    )
    print(
        f"Best Model {best_model.name} ROC AUC Validation Score for {task}: {val_score}"
    )


def main(
    run_singleBest: bool,
    run_ges: bool,
    run_qo: bool,
    run_qdo: bool,
    run_infer_time_qdo: bool,
    run_ens_size_qdo: bool,
):
    # Define the context for the ensemble evaluation
    context_name = "D244_F3_C1530_30"
    # Load the repository with the specified context
    repo: EvaluationRepository = load_repository(context_name, cache=True)
    # Load the data
    datasets, configs, folds, all_config_hyperparameters = load_data(repo, context_name)
    # A task is a fold of a dataset
    tasks = initialize_tasks(repo, datasets, folds)

    # Evaluate ensemble selection methods for each task
    for task in tasks:
        base_models, predictions_val, predictions_test = load_and_process_base_models(
            repo, task, configs, all_config_hyperparameters
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
            evaluate_single_best_model(base_models, repo, task)

        # GES evaluation
        if run_ges:
            ges = EnsembleSelection(
                base_models=base_models,
                n_iterations=100,
                metric=msc(metric_name=metric_name, is_binary=is_binary, labels=labels),
                random_state=1,
            )
            evaluate_ensemble("GES", ges, repo, task, predictions_val, predictions_test)

        # QO evaluation
        if run_qo:
            qo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=msc(
                    metric_name=metric_name, is_binary=is_binary, labels=labels
                ),
                random_state=1,
                archive_type="quality",
            )
            evaluate_ensemble("QO", qo, repo, task, predictions_val, predictions_test)

        # QDO evaluation
        if run_qdo:
            qdo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=msc(
                    metric_name=metric_name, is_binary=is_binary, labels=labels
                ),
                random_state=1,
                behavior_space=get_bs_configspace_similarity_and_loss_correlation(),
            )
            evaluate_ensemble("QDO", qdo, repo, task, predictions_val, predictions_test)

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
                random_state=1,
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
            )

        # QDO evaluation with ensemble size and loss correlation
        if run_ens_size_qdo:
            ens_size_qdo = QDOEnsembleSelection(
                base_models=base_models,
                n_iterations=3,
                score_metric=msc(
                    metric_name=metric_name, is_binary=is_binary, labels=labels
                ),
                random_state=1,
                behavior_space=get_bs_ensemble_size_and_loss_correlation(),
            )
            evaluate_ensemble(
                "ENS_SIZE_QDO",
                ens_size_qdo,
                repo,
                task,
                predictions_val,
                predictions_test,
            )


if __name__ == "__main__":
    main(
        run_singleBest=True,
        run_ges=False,
        run_qo=False,
        run_qdo=False,
        run_infer_time_qdo=False,
        run_ens_size_qdo=False,
    )
