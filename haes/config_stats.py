import json
import os
import numpy as np
import pandas as pd
import tracemalloc
from autogluon.tabular import TabularPredictor

# Load the configurations from the JSON files
config_path = "./configs/"
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
    "catboost": "configs_catboost.json",
}

# Combine configurations into a single dictionary
raw_hyperparameters = {}
for model, filename in files.items():
    with open(os.path.join(config_path, filename), "r") as file:
        raw_hyperparameters.update(json.load(file))

# Adjust the configurations using the provided adjustment code logic
adjusted_hyperparameters = {}
configs_hps = raw_hyperparameters.copy()
portfolio_configs = list(raw_hyperparameters.keys())

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
X_dummy = pd.DataFrame(
    np.random.random((num_samples, num_features)),
    columns=[f"feature_{i}" for i in range(num_features)],
)
X_dummy["target"] = np.random.randint(2, size=num_samples)

initialized = {}  # To ensure initialization is not measured
results = []
# Create a flat list of the models in portfolio_configs with an associated counter
portfolio_model_map = []
for _config_prio, config in enumerate(portfolio_configs):
    model_type = config.replace("_BAG_L1", "").split("_")[0]
    portfolio_model_map.append((model_type, config))

# Initialize a counter for mapping the portfolio configurations
portfolio_counter = 0

# Measure memory and disk usage for each model type
for model_name, configs in adjusted_hyperparameters.items():
    for i, config in enumerate(configs):
        print(f"\tConfiguration {i+1}/{len(configs)} for model {model_name}")  # Print progress for each configuration

        try:
            # Initialize and train the first time without measurement
            if i == 0:
                print(f"First run (no measurement) for {model_name}_{i}")
                predictor = TabularPredictor(
                    label="target", problem_type="binary", verbosity=0
                )
                predictor.fit(
                    train_data=X_dummy,
                    hyperparameters={f"{model_name}": config},
                    time_limit=60,  # short time limit for fitting on dummy data
                    verbosity=0,
                )
                del predictor  # Remove the model from memory

            # Now, measure during the second run (or subsequent runs)
            tracemalloc.start()

            # Initialize and train the model on dummy data again for measurement
            predictor = TabularPredictor(
                label="target", problem_type="binary", verbosity=0
            )
            predictor.fit(
                train_data=X_dummy,
                hyperparameters={f"{model_name}": config},
                time_limit=60,  # short time limit for fitting on dummy data
                verbosity=0,
            )

            # Measure memory usage
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")
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

            # Get the correct model name from portfolio_model_map
            correct_model_name = portfolio_model_map[portfolio_counter][1]
            portfolio_counter += 1  # Increment the counter to the next model configuration

            # Append the results to the list
            results.append(
                {
                    "Model_ID": f"{model_name}_{i}",
                    "Model": correct_model_name,  # Use the correct model name from portfolio_configs
                    "Memory": memory_usage,
                    "Predictor_Size": predictor_size,
                    "Learner_Size": learner_size,
                    "Models_Size": models_size,
                    "Total_Size": total_deployed_size,
                }
            )

            tracemalloc.stop()

        except KeyError as e:
            print(f"Error with model {model_name}_{i}: {e}")
        except Exception as e:
            print(f"General error with model {model_name}_{i}: {e}")


# Convert the results list to a DataFrame
df_results = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df_results.to_csv("/usr/src/app/output/model_memory_and_disk_usage.csv", index=False)

# Output the first few rows for debugging purposes
print(df_results.head())
