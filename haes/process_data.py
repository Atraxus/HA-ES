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


def insert_measurements(df):
    # Read the CSV file
    df_usage_measurements = pd.read_csv("data/model_memory_and_disk_usage.csv")

    # Ensure that the 'Model' column is treated as a string
    df_usage_measurements["Model"] = df_usage_measurements["Model"].astype(str)

    # Explode the 'models_used' column to have one model per row
    df_exploded = df.explode("models_used")

    # Extract 'config_key' by removing the '_BAG_L1' suffix
    df_exploded["config_key"] = df_exploded["models_used"].str.rsplit("_BAG_L1", n=1).str[0]

    # Merge the exploded DataFrame with the usage measurements
    df_merged = df_exploded.merge(
        df_usage_measurements, left_on="config_key", right_on="Model", how="left"
    )

    # Fill NaN values with 0 for memory and disk space
    df_merged[["Memory", "Models_Size"]] = df_merged[["Memory", "Models_Size"]].fillna(0)

    # Group by the original index and sum the memory and disk space
    df_usage = df_merged.groupby(level=0).agg({"Memory": "sum", "Models_Size": "sum"})

    # Assign the aggregated usage data back to the original DataFrame
    df["memory"] = df_usage["Memory"]
    df["disk_space"] = df_usage["Models_Size"]


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
    for task in df["task"].unique():
        mask = df["task"] == task
        if "roc_auc_val" in df.columns:
            df.loc[mask, "normalized_roc_auc_val"] = normalize_data(
                df.loc[mask, "roc_auc_val"]
            )
        if "roc_auc_test" in df.columns:
            df.loc[mask, "normalized_roc_auc_test"] = normalize_data(
                df.loc[mask, "roc_auc_test"]
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
            if f.endswith(".json") and any(name + '_' in f for name in method_names)
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

    return df



if __name__ == "__main__":
    repo = load_repository("D244_F3_C1530_100", cache=True)

    # Create MULTI_GES method names based on infer_time_weights
    if False:  # Add multi-ges
        num_solutions = 20
        infer_time_weights = np.linspace(0, 1, num=num_solutions)
        infer_time_weights = np.round(infer_time_weights, 2)
        multi_ges_methods = [
            f"MULTI_GES-{time_weight:.2f}" for time_weight in infer_time_weights
        ]
    else:
        multi_ges_methods = []  # Empty list if MULTI_GES is not needed

    # Original method names
    method_names = [
        "SINGLE_BEST",
        "GES",
        # "MULTI_GES",
        # "QO",
        #"QDO",
        # "ENS_SIZE_QDO",
        #"INFER_TIME_QDO",
        # "DISK_QDO",
        # "MEMORY_QDO",
    ]

    # Append the MULTI_GES method names
    method_names.extend(multi_ges_methods)

    df = parse_dataframes(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        repo=repo,
        method_names=method_names,
    )
    # insert_measurements(df)
    normalize_per_dataset(df)
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
    filtered_data = df[df['method'].isin(method_names)]
    
    avg_roc_auc = filtered_data.groupby(['dataset', 'method'])['roc_auc_val'].mean().unstack()
    print(f"ROC AUC for validation set per method and dataset:\n{avg_roc_auc}\n")
    avg_roc_auc_test = filtered_data.groupby(['dataset', 'method'])['roc_auc_test'].mean().unstack()
    print(f"ROC AUC for test set per method and dataset:\n{avg_roc_auc_test}\n")
    inference_time = filtered_data.groupby(['dataset', 'method'])['inference_time'].mean().unstack()
    print(f"Inference time per method and dataset:\n{inference_time}\n")

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
