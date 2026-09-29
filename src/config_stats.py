"""Measure inference memory and disk usage of every TabArena config on dummy data.

Writes data/model_memory_and_disk_usage.csv, which generate_data.py and process_data.py read.
"""

import argparse
import os
import shutil
import tempfile
import tracemalloc
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import psutil
from autogluon.tabular import TabularPredictor
from tqdm import tqdm

from tabarena_data import DEFAULT_METHODS, RESOURCE_USAGE_PATH, get_context


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure memory and disk usage of TabArena configs."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="TabArena methods whose configs are measured.",
    )
    parser.add_argument("--output", default=RESOURCE_USAGE_PATH)
    return parser.parse_args()


def create_dummy_data(num_samples: int = 500, num_features: int = 15):
    X_dummy = pd.DataFrame(
        np.random.random((num_samples, num_features)),
        columns=[f"feature_{i}" for i in range(num_features)],
    )
    X_dummy["target"] = np.random.randint(2, size=num_samples)

    # Split into train and test sets
    train_data = X_dummy.sample(frac=0.8, random_state=42)
    test_data = X_dummy.drop(train_data.index)
    return train_data, test_data


# Function to measure memory during inference
def measure_inference_memory(
    config_name, model_type, hyperparameters, train_data, test_data, result_queue
):
    # Each predictor is only needed for the measurement, so it is written to a temporary directory
    predictor_path = tempfile.mkdtemp(prefix="config_stats_")
    try:
        predictor = TabularPredictor(
            label="target", problem_type="binary", verbosity=0, path=predictor_path
        )
        predictor.fit(
            train_data=train_data,
            hyperparameters={model_type: [hyperparameters]},
            time_limit=60,
            verbosity=0,
        )

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        tracemalloc.start()

        predictor.predict(test_data.drop(columns=["target"]))

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

        result_queue.put({
            "Model": config_name,
            "Model_Type": model_type,
            "Inference_Memory_Usage": mem_usage_inference,
            "Peak_Memory_During_Inference": peak,
            "Predictor_Size": predictor_size,
            "Learner_Size": learner_size,
            "Models_Size": models_size,
            "Total_Size": total_deployed_size,
        })

    except Exception as e:
        result_queue.put({
            "Model": config_name,
            "Error": str(e)
        })
    finally:
        shutil.rmtree(predictor_path, ignore_errors=True)


def main(methods: list[str], save_path: str):
    # {config_name: {"model_type": ..., "hyperparameters": {...}}}
    configs_hyperparameters = get_context().load_configs_hyperparameters(
        methods=methods, download="auto"
    )
    train_data, test_data = create_dummy_data()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Incremental saving setup
    save_interval = 10  # Save every 10 models processed
    results = []
    pbar = tqdm(total=len(configs_hyperparameters), desc="Total Progress")

    for config_name, config in configs_hyperparameters.items():
        pbar.set_description(f"Processing {config_name}")

        result_queue = Queue()
        p = Process(
            target=measure_inference_memory,
            args=(
                config_name,
                config["model_type"],
                config["hyperparameters"],
                train_data,
                test_data,
                result_queue,
            ),
        )
        p.start()
        p.join()

        if not result_queue.empty():
            result = result_queue.get()
            if "Error" in result:
                print(f"Error with model {config_name}: {result['Error']}")
            else:
                results.append(result)
        else:
            print(f"No result for {config_name}")

        pbar.update(1)

        # Save intermediary results at set intervals
        if results and len(results) % save_interval == 0:
            pd.DataFrame(results).to_csv(save_path, index=False)

    # Final save after all processing
    pbar.close()
    pd.DataFrame(results).to_csv(save_path, index=False)

    # Output the first few rows for debugging purposes
    print(pd.DataFrame(results).head())


if __name__ == "__main__":
    args = parse_args()
    main(args.methods, args.output)
