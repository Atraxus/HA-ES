import os
import numpy as np
import csv
import argparse
import sys

sys.path.append("src")
from tabarena_data import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_METHODS,
    classification_datasets,
    initialize_tasks,
    load_repo,
)

parser = argparse.ArgumentParser(description="List missing result files.")
parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
parser.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS)
args = parser.parse_args()

repo = load_repo(args.methods)

# Only classification is supported for now
datasets = classification_datasets(repo)
tasks = initialize_tasks(repo, datasets, args.folds)

# Generate the list of methods
basic_methods = [
    'SINGLE_BEST',
    'QO',
    'QDO',
    'ENS_SIZE_QDO',
    'INFER_TIME_QDO',
    'MEMORY_QDO',
    'DISK_QDO',
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
