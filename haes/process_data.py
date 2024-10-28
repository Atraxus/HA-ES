import re
import pandas as pd
import numpy as np
from tabrepo import load_repository, EvaluationRepository
import os


def parse_df_filename(filename):
    match = re.match(r"^(.+?)_(\d+)_(\d+)\.json$", filename)
    if match:
        method_name = match.group(1)
        task_id = match.group(2)
        fold_number = match.group(3)
        return method_name, task_id, fold_number
    else:
        return None


def get_inference_time(entry, repo, metrics):
    fold = int(entry["fold"])
    dataset = repo.task_to_dataset(entry["task"])
    configs = entry["models_used"]  # This is a tuple of configurations

    dataset_fold_metrics = metrics.loc[(dataset, fold)]
    average_time_infer = dataset_fold_metrics["time_infer_s"].mean()

    # Initialize total inference time
    total_inference_time = 0

    # Iterate over each configuration in the tuple
    for config in configs:
        # Select the metrics for each configuration
        if (dataset, fold, config) in metrics.index:
            selected_metrics = metrics.loc[(dataset, fold, config)]
            # Sum the 'time_infer_s' values for each configuration and add to total
            total_inference_time += selected_metrics["time_infer_s"].sum()
        else:
            total_inference_time += average_time_infer

    return total_inference_time


def normalize_data(data):
    """Normalize data to [0, 1] range; handle cases with NaN or zero range."""
    min_val = np.nanmin(data)  # Use nanmin to ignore NaNs
    max_val = np.nanmax(data)  # Use nanmax to ignore NaNs

    if np.isnan(min_val) or np.isnan(max_val):
        raise ValueError("Data contains NaN values which cannot be normalized.")

    if max_val == min_val:
        # Handle the case where all data points are the same or NaNs are present
        return np.zeros_like(data)

    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data


def normalize_per_dataset(df):
    for task_id in df["task_id"].unique():
        mask = df["task_id"] == task_id
        if "roc_auc_test" in df.columns:
            df.loc[mask, "negated_normalized_roc_auc"] = normalize_data(
                -df.loc[mask, "roc_auc_test"]
            )
        if "inference_time" in df.columns:
            df.loc[mask, "normalized_time"] = normalize_data(
                df.loc[mask, "inference_time"]
            )
        if "memory" in df.columns:
            df.loc[mask, "normalized_memory"] = normalize_data(
                df.loc[mask, "memory"]  #
            )
        if "disk_space" in df.columns:
            df.loc[mask, "normalized_diskspace"] = normalize_data(
                df.loc[mask, "disk_space"]
            )

    return df


def parse_dataframes(
    seeds: list[int], repo: EvaluationRepository, method_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = repo.metrics(datasets=repo.datasets(), configs=repo.configs())
    results_path = "results"
    all_dfs = []
    print("Start data loading...")
    for seed in seeds:
        print(f"Seed: {seed}")
        df_list_seed = []
        filepath = f"{results_path}/seed_{seed}/"
        files = [
            f
            for f in os.listdir(filepath)
            if f.endswith(".json") and any(name in f for name in method_names)
        ]
        for file in files:
            df = pd.read_json(filepath + file)
            method_name, task_id, fold_number = parse_df_filename(file)
            df["task"] = f"{task_id}_{fold_number}"
            df["method"] = method_name
            df["task_id"] = task_id
            df["fold"] = fold_number
            df["seed"] = seed
            if method_name == "GES":
                df["name"] = df["iteration"].apply(lambda x: f"{method_name}_{x}")
            df_list_seed.append(df)

        df_seed = pd.concat(df_list_seed)
        df_seed["models_used"] = df_seed["models_used"].apply(tuple)
        df_seed["inference_time"] = df_seed.apply(
            get_inference_time, axis=1, args=(repo, metrics)
        )

        # After calculating inference times, convert 'models_used' back to lists:
        df_seed["models_used"] = df_seed["models_used"].apply(list)
        all_dfs.append(df_seed)

    df = pd.concat(all_dfs)
    normalize_per_dataset(df)
    # df = df.drop(
    #     columns=["task", "iteration", "weights", "models_used", "dataset", "meta"],
    #     errors="ignore",
    # )

    return df


if __name__ == "__main__":
    repo = load_repository("D244_F3_C1530_100", cache=True)

    # Create MULTI_GES method names based on infer_time_weights
    if True:  # Add multi-ges
        infer_time_weights = np.delete(np.linspace(0, 1, num=15), 0)
        infer_time_weights = np.append(
            infer_time_weights, np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        )
        infer_time_weights = np.round(infer_time_weights, 2)
        multi_ges_methods = [
            f"MULTI_GES-{time_weight}" for time_weight in infer_time_weights
        ]
    else:
        multi_ges_methods = []  # Empty list if MULTI_GES is not needed

    # Original method names
    method_names = [
        "SINGLE_BEST",
        "GES",
        # "MULTI_GES",
        # "QO",
        "QDO",
        # "ENS_SIZE_QDO",
        "INFER_TIME_QDO",
        # "DISK_QDO",
        # "MEMORY_QDO",
    ]

    # Append the MULTI_GES method names
    # method_names.extend(multi_ges_methods)

    df = parse_dataframes(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        repo=repo,
        method_names=method_names,
    )
    print(df["method"].unique())
    df.reset_index(drop=True, inplace=True)

    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_json("data/full.json")
    print("Done writing data to data/full.json...")
    df.to_csv("data/full.csv")

    print(f"df shape: {df.shape}")
    print(df.columns)
    print(df.head())

    # Drop specified columns
    df = df.drop(
        columns=[
            "task",
            "iteration",
            "weights",
            "models_used",
            "dataset",
            "meta",
            "name",
            "fold",
            "task_id",
            "time_weight",
            "seed",
        ],
        errors="ignore",
    )
    roc_auc_val = df.groupby("method").agg("mean")["roc_auc_val"]
    print(f"ROC AUC for validation set per method:\n{roc_auc_val}\n")
    roc_auc_test = df.groupby("method").agg("mean")["roc_auc_test"]
    print(f"ROC AUC for test set per method:\n{roc_auc_test}\n")
    inference_time = df.groupby("method").agg("mean")["inference_time"]
    print(f"Inference time per method:\n{inference_time}\n")

    # Total number of entries
    total_entries = df.shape[0]

    # Number of entries with inference_time equal to 0
    zero_infer_count = df[df["inference_time"] == 0].shape[0]

    # Calculate the percentage of zero entries
    percentage_zero = (zero_infer_count / total_entries) * 100

    # Print the results
    print(f"Total entries: {total_entries}")
    print(f"Entries with inference time of 0: {zero_infer_count}")
    print(f"Percentage of zero inference times: {percentage_zero:.2f}%")

    # main()
