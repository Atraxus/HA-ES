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
    configs = entry["models_used"]

    dataset_fold_metrics = metrics.loc[(dataset, fold)]
    average_time_infer = dataset_fold_metrics["time_infer_s"].mean()

    # Initialize total inference time
    total_inference_time = 0

    # Iterate over each configuration in the tuple
    for config in configs:
        # failed = 0
        # Select the metrics for each configuration
        if (dataset, fold, config) in metrics.index:
            selected_metrics = metrics.loc[(dataset, fold, config)]
            # Sum the 'time_infer_s' values for each configuration and add to total

            if np.isscalar(selected_metrics["time_infer_s"]):
                total_inference_time += selected_metrics["time_infer_s"]
            else:
                raise ValueError(
                    f"Parsing failed: expected scalar for 'time_infer_s' but got {type(selected_metrics['time_infer_s'])} "
                    f"with value: {selected_metrics['time_infer_s']}"
                )
        else:
            # failed += 1
            total_inference_time += average_time_infer

    # if failed > 0:
    #     print(f"Failed to retrieve {failed} of {len(configs)} inference time entries and used average of {average_time_infer} instead.")
    return total_inference_time


def compute_total_resource_usage(df, csv_path="data/model_memory_and_disk_usage.csv"):
    """Compute total memory and diskspace usage for each ensemble in the dataframe."""
    df_usage = pd.read_csv(csv_path)
    resource_usage_dict = df_usage.set_index("Model").to_dict("index")

    def get_total_usage(models_used):
        total_memory = 0
        total_diskspace = 0
        for model_name in models_used:
            model_name_clean = model_name.replace("_BAG_L1", "")
            if model_name_clean in resource_usage_dict:
                usage = resource_usage_dict[model_name_clean]
                total_memory += usage["Inference_Memory_Usage"]
                total_diskspace += usage["Models_Size"]
            else:
                print(
                    f"Warning: Model type '{model_name_clean}' not found in resource usage data."
                )
        return pd.Series({"memory": total_memory, "diskspace": total_diskspace})

    # Apply the function to each row in 'df'
    df[["memory", "diskspace"]] = df["models_used"].apply(get_total_usage)
    return df


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
        for seed in df["seed"].unique():
            mask = (df["task"] == task) & (df["seed"] == seed)
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
            if "diskspace" in df.columns:
                df.loc[mask, "normalized_diskspace"] = normalize_data(
                    df.loc[mask, "diskspace"]
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
            if f.endswith(".json") and any(f.startswith(name + "_") for name in method_names)
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
        df_seed["inference_time"] = df_seed.apply(
            get_inference_time, axis=1, args=(repo, metrics)
        )
        all_dfs.append(df_seed)
    df = pd.concat(all_dfs)

    return df


if __name__ == "__main__":
    repo = load_repository("D244_F3_C1530_100", cache=True)
    # The following is for selecting specific methods to parses
    include_multi_ges = False
    method_names = [
        "SINGLE_BEST",
        "GES",
        # # "MULTI_GES",
        # # "QO",
        "QDO",
        "ENS_SIZE_QDO",
        "INFER_TIME_QDO",
        "DISK_QDO",
        "MEMORY_QDO",
        # "MULTI_GES-0.21",
        # "MULTI_GES-0.79",
    ]
    if include_multi_ges:
        num_solutions = 20
        infer_time_weights = np.round(np.linspace(0, 1, num=num_solutions), 2)
        multi_ges_methods = [
            f"MULTI_GES-{weight:.2f}" for weight in infer_time_weights
        ]
        method_names.extend(multi_ges_methods)
    
    # Parsing
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = (
        parse_dataframes(seeds, repo, method_names)
        .pipe(compute_total_resource_usage)
        .pipe(normalize_per_dataset)
        .reset_index(drop=True)
    )
    print(f"Methods: {df['method'].unique()}")
    #! There is a quirk here. ROC AUC is already inverted by the generation script to be a loss.
    #! That's why there is no ivnersion happening before the HV calculations. Would be nicer to rename but this is how it grew over time.

    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv("data/full.csv")

    # Some analysis
    print(df.info())

    avg_roc_auc = df.groupby(["dataset", "method"])["roc_auc_val"].mean().unstack()
    print(f"ROC AUC for validation set per method and dataset:\n{avg_roc_auc}\n")
    avg_roc_auc_test = (
        df.groupby(["dataset", "method"])["roc_auc_test"].mean().unstack()
    )
    print(f"ROC AUC for test set per method and dataset:\n{avg_roc_auc_test}\n")
    inference_time = (
        df.groupby(["dataset", "method"])["inference_time"].mean().unstack()
    )
    print(f"Inference time per method and dataset:\n{inference_time}\n")
    total_entries = df.shape[0]
    zero_infer_count = df[df["inference_time"] == 0].shape[0]
    percentage_zero = (zero_infer_count / total_entries) * 100

    # Print the results
    print(f"Total entries: {total_entries}")
    print(f"Entries with inference time of 0: {zero_infer_count}")
    print(f"Percentage of zero inference times: {percentage_zero:.2f}%")
