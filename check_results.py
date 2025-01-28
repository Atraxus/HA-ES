import os
import numpy as np
import csv
from tabrepo import load_repository, get_context

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

# Define the context for the ensemble evaluation
context_name = "D244_F3_C1530_100"

# Load the repository with the specified context
repo = load_repository(context_name, cache=True)

# Load the data
datasets, configs, folds, all_config_hyperparameters = load_data(repo, context_name)

# Initialize the tasks
tasks = initialize_tasks(repo, datasets, folds)

# Generate the list of methods
basic_methods = [
    'SINGLE_BEST',
    'QO',
    'QDO',
    'ENS_SIZE_QDO',
    'INFER_TIME_QDO',
    'MEMORY_QDO',
    'DISK_SPACE_QDO',
    'GES',
]

# Generate MULTI_GES methods with 20 samples from 0 to 1
multi_ges_methods = [f"MULTI_GES-{t:.2f}" for t in np.linspace(0, 1, 20)]

# Combine all methods
methods = basic_methods + multi_ges_methods

# Define seeds (seed_0 to seed_9)
seeds = [f'seed_{i}' for i in range(10)]

# Dictionary to keep track of missing files per method
missing_files = {method: [] for method in methods}

# Iterate over each combination and check for missing files
for method in methods:
    for seed in seeds:
        seed_dir = os.path.join('results', seed)
        for task in tasks:
            dataset = repo.task_to_dataset(task)
            task_type = repo.dataset_metadata(dataset=dataset)["task_type"]
            if task_type != "Supervised Classification":
                continue  # Only support classification for now
            filename = f"{method}_{task}.json"
            filepath = os.path.join(seed_dir, filename)
            if not os.path.isfile(filepath):
                missing_files[method].append(filepath)

# Report overview of methods with missing files
print("Methods with missing files:")
for method, files in missing_files.items():
    if files:
        print(f"{method}: {len(files)} missing files")

# Create a CSV of missing files per method
with open('missing_files.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Method', 'MissingFilePath'])
    for method, files in missing_files.items():
        for filepath in files:
            writer.writerow([method, filepath])

print("\nCSV file 'missing_files.csv' has been created with the list of missing files per method.")
