from tabrepo import load_repository, EvaluationRepository
from phem.methods.ensemble_selection import EnsembleSelection
from phem.application_utils.supported_metrics import msc
from phem.methods.ensemble_selection.qdo.qdo_es import QDOEnsembleSelection
from sklearn.metrics import roc_auc_score

import numpy as np
from dataclasses import dataclass

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


def evaluate_ensemble(name: str, ensemble: EnsembleSelection, repo: EvaluationRepository, task: str, predictions_val: list[np.ndarray], predictions_test: list[np.ndarray]):
    dataset = repo.task_to_dataset(task)
    fold = repo.task_to_fold(task)
    
    y_val = repo.labels_val(dataset=dataset, fold=fold)
    # Format (n_configs, n_samples, n_classes)
    predictions_val = np.array(predictions_val)
    predictions_test = np.array(predictions_test)

    number_of_classes = int(repo.dataset_metadata(dataset=dataset)["NumberOfClasses"]) # 0 for regression
    if (number_of_classes == 2):
        predictions_val = expand_binary_predictions(predictions_val)
        predictions_test = expand_binary_predictions(predictions_test)

    # Switch to simulating predictions on validation data  (i.e., the training data of the ensemble)
    for bm in ensemble.base_models:
        bm.switch_to_val_simulation()

    print(f"Fitting {name} for task {task}, dataset {dataset}, fold {fold}")
    ensemble.fit(predictions_val, y_val)

    # Simulate to simulating predictions on test data  (i.e., the test data of the ensemble and base models)
    for bm in ensemble.base_models:
        bm.switch_to_test_simulation()

    y_pred = ensemble.ensemble_predict_proba(predictions_test)
    if number_of_classes == 2:
        y_pred = y_pred[:, 1]
    y_test = repo.labels_test(dataset=dataset, fold=fold)
    print(f"\tROC AUC Test Score for {task}: {roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro')}")
    if(name == "GES"):
        print(f"\tFinal ensemble size: {len(ensemble.indices_)}")
        print(f"\tNumber of different base models in the ensemble: {len(set(ensemble.indices_))}")
    elif(name == "QDO"):
        print(f"\tFinal ensemble size: {len(ensemble.archive)}")
        print(f"\tNumber of different base models in the ensemble: {len(set(ensemble.archive))}")
    else: 
        pass

def main():
    # Define the context for the ensemble evaluation
    context_name = "D244_F3_C1530_30"
    # Load the repository with the specified context
    repo: EvaluationRepository = load_repository(context_name, cache=True)
    
    datasets = repo.datasets()
    configs = repo.configs()
    folds = [0,1,2]

    # A task is a fold of a dataset
    tasks = [
        repo.task_name(dataset=dataset, fold=fold)
        for dataset in datasets
        for fold in folds
    ]

    # Setup the ensemble evaluation
    # for each task, evaluate the ensemble of size n
    ensemble_size = 100
    for task in tasks:
        dataset = repo.task_to_dataset(task)
        fold = repo.task_to_fold(task)
        
        base_models = []
        predictions_val = []
        predictions_test = []
        # Iterate over each config to create a base model representation
        for config in configs:
            # Fetch predictions for each dataset and fold
            predictions_val.append(repo.predict_val(dataset=dataset, fold=fold, config=config))
            predictions_test.append(repo.predict_test(dataset=dataset, fold=fold, config=config))

            # Wrap predictions in the FakedFittedAndValidatedClassificationBaseModel
            # Note: You might need to adjust how predictions are combined or selected
            model = FakedFittedAndValidatedClassificationBaseModel(
                name=config,
                val_probabilities=predictions_val[-1],
                test_probabilities=predictions_test[-1],
            )

            base_models.append(model)

        # "Supervised Classification" or "Supervised Regression"
        task_type = repo.dataset_metadata(dataset=dataset)["task_type"]
        number_of_classes = int(repo.dataset_metadata(dataset=dataset)["NumberOfClasses"]) # 0 for regression
        
        # Adjusting the metric based on the task
        if task_type == "Supervised Classification":
            is_binary = number_of_classes == 2
            metric_name = "roc_auc" 
            labels = list(range(number_of_classes))  
        elif task_type == "Supervised Regression":
            continue # Skip regression tasks for now
            
        # Initialize EnsembleSelection with the selected metric
        ges = EnsembleSelection(
            base_models=base_models,
            n_iterations=50,
            metric=msc(metric_name=metric_name, is_binary=is_binary, labels=labels),
            random_state=1,
        )
        qdo = QDOEnsembleSelection(
            base_models=base_models,
            n_iterations=50,
            score_metric=msc(metric_name=metric_name, is_binary=is_binary, labels=labels),
            random_state=1,
            archive_type="quality",
        )
        
        evaluate_ensemble("GES", ges, repo, task, predictions_val, predictions_test)
        evaluate_ensemble("QDO", qdo, repo, task, predictions_val, predictions_test)


if __name__ == '__main__':
    main()