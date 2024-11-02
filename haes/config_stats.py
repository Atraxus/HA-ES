import json
import os
import numpy as np
import pandas as pd
import psutil
import tracemalloc
from autogluon.tabular import TabularPredictor
from multiprocessing import Process, Queue
from tqdm import tqdm

# Load the configurations from the JSON files
config_path = "./extern/tabrepo/data/configs/"
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

# Adjust the configurations
adjusted_hyperparameters = {}
configs_hps = raw_hyperparameters.copy()
portfolio_configs = list(raw_hyperparameters.keys())

for _config_prio, config in enumerate(portfolio_configs):
    tabrepo_config_name = config.replace("_BAG_L1", "")
    new_config = configs_hps[tabrepo_config_name].copy()
    model_type = new_config.pop("model_type")
    new_config = new_config["hyperparameters"]

    model_params = {}
    optimization_params = {}
    other_params = {}

    for key, value in list(new_config.items()):
        if key.startswith('model.ft_transformer.'):
            param_name = key[len('model.ft_transformer.'):]
            model_params[param_name] = value
            del new_config[key]
        elif key.startswith('optimization.'):
            param_name = key[len('optimization.'):]
            optimization_params[param_name] = value
            del new_config[key]
        else:
            other_params[key] = value

    if model_params:
        new_config.setdefault('model', {})
        new_config['model']['ft_transformer'] = model_params

    if optimization_params:
        new_config.setdefault('optimization', {})
        new_config['optimization'].update(optimization_params)

    new_config.update(other_params)

    if model_type not in adjusted_hyperparameters:
        adjusted_hyperparameters[model_type] = []
    new_config["ag_args"] = new_config.get("ag_args", {})
    new_config["ag_args"]["priority"] = 0 - _config_prio
    adjusted_hyperparameters[model_type].append(new_config)

# Create dummy data
num_samples = 500
num_features = 15
X_dummy = pd.DataFrame(
    np.random.random((num_samples, num_features)),
    columns=[f"feature_{i}" for i in range(num_features)],
)
X_dummy["target"] = np.random.randint(2, size=num_samples)

# Split into train and test sets
train_data = X_dummy.sample(frac=0.8, random_state=42)
test_data = X_dummy.drop(train_data.index)

results = []
portfolio_model_map = []
for _config_prio, config in enumerate(portfolio_configs):
    model_type = config.replace("_BAG_L1", "").split("_")[0]
    portfolio_model_map.append((model_type, config))

portfolio_counter = 0

# Function to measure memory during inference
def measure_inference_memory(model_name, config, model_id, result_queue):
    try:
        predictor = TabularPredictor(
            label="target", problem_type="binary", verbosity=0
        )
        predictor.fit(
            train_data=train_data,
            hyperparameters={f"{model_name}": config},
            time_limit=60,
            verbosity=0,
        )

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        tracemalloc.start()

        predictions = predictor.predict(test_data.drop(columns=["target"]))

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_after = process.memory_info().rss
        mem_usage_inference = mem_after - mem_before

        predictor.save()

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

        correct_model_name = portfolio_model_map[portfolio_counter][1]

        result_queue.put({
            "Model_ID": model_id,
            "Model": correct_model_name,
            "Inference_Memory_Usage": mem_usage_inference,
            "Peak_Memory_During_Inference": peak,
            "Predictor_Size": predictor_size,
            "Learner_Size": learner_size,
            "Models_Size": models_size,
            "Total_Size": total_deployed_size,
        })

    except Exception as e:
        result_queue.put({
            "Model_ID": model_id,
            "Error": str(e)
        })

# Incremental saving setup
save_interval = 10  # Save every 10 models processed
save_path = "model_memory_and_disk_usage.csv"

# Initialize progress bar
total_configs = sum(len(configs) for configs in adjusted_hyperparameters.values())
pbar = tqdm(total=total_configs, desc='Total Progress')

# Process models
for model_name, configs in adjusted_hyperparameters.items():
    for i, config in enumerate(configs):
        model_id = f"{model_name}_{i}"
        pbar.set_description(f"Processing {model_id}")

        result_queue = Queue()
        p = Process(target=measure_inference_memory, args=(model_name, config, model_id, result_queue))
        p.start()
        p.join()

        if not result_queue.empty():
            result = result_queue.get()
            if "Error" in result:
                print(f"Error with model {model_id}: {result['Error']}")
            else:
                results.append(result)
        else:
            print(f"No result for {model_id}")

        portfolio_counter += 1
        pbar.update(1)

        # Save intermediary results at set intervals
        if len(results) % save_interval == 0:
            pd.DataFrame(results).to_csv(save_path, index=False)

# Final save after all processing
pbar.close()
pd.DataFrame(results).to_csv(save_path, index=False)

# Output the first few rows for debugging purposes
print(pd.DataFrame(results).head())
