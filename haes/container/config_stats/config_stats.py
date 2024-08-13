# Script to measure the resource impact of configs in TabRepo.
# Specifically memory usage and required disk space are being measured.
# The script reads in the .json config files from TabRepo and uses Autogluon
# to create the model instances.

import json
import os
import numpy as np
import pandas as pd
import tracemalloc
from autogluon.tabular import TabularPredictor
from tqdm.notebook import tqdm

# Load the configurations from the JSON files
config_path = "../extern/tabrepo/data/configs/"
files = {
    "xt": "configs_xt.json",
    "xgboost": "configs_xgboost.json",
    "tabpfn": "configs_tabpfn.json",
    "rf": "configs_rf.json",
    "nn_torch": "configs_nn_torch.json",
    "lr": "configs_lr.json",
    "lightgbm": "configs_lightgbm.json",
    "knn": "configs_knn.json",
    "ftt": "configs_ftt.json",
    "fastai": "configs_fastai.json",
    "catboost": "configs_catboost.json"
}

# Combine configurations into a single dictionary
raw_hyperparameters = {}
for model, filename in files.items():
    with open(os.path.join(config_path, filename), 'r') as file:
        raw_hyperparameters.update(json.load(file))

# Adjust the configurations using the provided adjustment code logic
adjusted_hyperparameters = {}
configs_hps = raw_hyperparameters.copy()  # Assuming repo._zeroshot_context.configs_hyperparameters equivalent
portfolio_configs = list(raw_hyperparameters.keys())  # Assuming portfolio_configs is a list of all config names

for _config_prio, config in enumerate(portfolio_configs):
    tabrepo_config_name = config.replace("_BAG_L1", "")
    new_config = configs_hps[tabrepo_config_name].copy()
    model_type = new_config.pop("model_type")
    new_config = new_config["hyperparameters"]

    if model_type not in adjusted_hyperparameters:
        adjusted_hyperparameters[model_type] = []
    new_config["ag_args"] = new_config.get("ag_args", {})
    new_config["ag_args"]["priority"] = 0 - _config_prio
    adjusted_hyperparameters[model_type].append(new_config)

# Create dummy data
num_samples = 100
num_features = 10
X_dummy = pd.DataFrame(np.random.random((num_samples, num_features)), columns=[f'feature_{i}' for i in range(num_features)])
X_dummy['target'] = np.random.randint(2, size=num_samples)

# Prepare a list to collect results
results = []

# Measure memory and disk usage for each model type
for model_name, configs in tqdm(adjusted_hyperparameters.items(), desc="Model Types", leave=False):
    for i, config in enumerate(tqdm(configs, desc=f"{model_name} Configurations", leave=False)):
        try:
            tracemalloc.start()
            
            # Initialize and train the model on dummy data
            predictor = TabularPredictor(label='target', problem_type='binary', verbosity=0)
            predictor.fit(
                train_data=X_dummy,
                hyperparameters={f"{model_name}": config},
                time_limit=60,  # short time limit for fitting on dummy data
                verbosity=0,
            )
            
            # Measure memory usage
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            memory_usage = sum(stat.size for stat in top_stats)

            # Save the model
            predictor.save()
            
            # Measure disk usage
            predictor_file = os.path.join(predictor.path, "predictor.pkl")
            learner_file = os.path.join(predictor.path, "learner.pkl")
            model_files = []
            for root, dirs, files in os.walk(predictor.path):
                for file in files:
                    if file == "model.pkl":
                        model_files.append(os.path.join(root, file))
                        
            predictor_size = os.path.getsize(predictor_file)
            learner_size = os.path.getsize(learner_file)
            models_size = sum(os.path.getsize(f) for f in model_files)
            total_deployed_size = predictor_size + learner_size + models_size

            # Append the results to the list
            results.append({
                "Model": f"{model_name}_{i}",
                "Memory used (bytes)": memory_usage,
                "Predictor size (bytes)": predictor_size,
                "Learner size (bytes)": learner_size,
                "Models size (bytes)": models_size,
                "Total deployment size (bytes)": total_deployed_size
            })
            
            tracemalloc.stop()

        except KeyError as e:
            print(f"Error with model {model_name}_{i}: {e}")
        except Exception as e:
            print(f"General error with model {model_name}_{i}: {e}")

# Convert the results list to a DataFrame
df_results = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df_results.to_csv('model_memory_and_disk_usage.csv', index=False)

# Output the first few rows for debugging purposes
print(df_results.head())