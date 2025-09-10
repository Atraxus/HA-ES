import re
import pandas as pd
import numpy as np
from tabrepo import load_repository, EvaluationRepository
import os
from concurrent.futures import ThreadPoolExecutor  # Import for multithreading

# --------------------------------------------------------------------------
# All helper functions remain the same
# --------------------------------------------------------------------------


def parse_df_filename(filename):
    """Parses method name, task ID, and fold number from a JSON filename."""
    match = re.match(r"^(.+?)_(\d+)_(\d+)\.json$", filename)
    if match:
        method_name = match.group(1)
        task_id = match.group(2)
        fold_number = match.group(3)
        return method_name, task_id, fold_number
    else:
        return None


def get_inference_time(entry, repo, metrics):
    """Calculates the total inference time for a given ensemble."""
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

def load_resource_usage(csv_path="data/model_memory_and_disk_usage.csv"):
    """Load and process resource usage data from a CSV file."""
    df_usage = pd.read_csv(csv_path)
    # Extract model types
    df_usage['ModelType'] = df_usage['Model'].str.extract(r'^(\w+_[rc])')
    # Compute average resource usage per model type
    resource_usage = df_usage.groupby('ModelType').agg({
        'Inference_Memory_Usage': 'mean',
        'Models_Size': 'mean'
    })
    # Create a dictionary for quick lookup
    resource_usage_dict = resource_usage.to_dict('index')
    return resource_usage_dict

def compute_total_resource_usage(df, resource_usage_dict):
    """Compute total memory and diskspace usage for each ensemble in the dataframe."""

    def get_total_usage(models_used):
        total_memory = 0
        total_diskspace = 0
        for model_name in models_used:
            # Remove '_BAG_L1' suffix
            model_name_clean = model_name.replace('_BAG_L1', '')
            # Extract model type
            match = re.match(r'^(\w+_[rc])', model_name_clean)
            if match:
                model_type = match.group(1)
                # Retrieve resource usage if model_type exists
                if model_type in resource_usage_dict:
                    usage = resource_usage_dict[model_type]
                    total_memory += usage['Inference_Memory_Usage']
                    total_diskspace += usage['Models_Size']
                else:
                    print(f"Warning: Model type '{model_type}' not found in resource usage data.")
            else:
                print(f"Warning: Model name '{model_name}' does not match expected format.")
        return pd.Series({'memory': total_memory, 'diskspace': total_diskspace})

    # Apply the function to each row in 'df'
    df[['memory', 'diskspace']] = df['models_used'].apply(get_total_usage)
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
    """Normalizes performance and resource metrics per dataset."""
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
            df.loc[mask, "normalized_memory"] = normalize_data(df.loc[mask, "memory"])
        if "disk_space" in df.columns:
            df.loc[mask, "normalized_diskspace"] = normalize_data(
                df.loc[mask, "disk_space"]
            )
    return df


# --------------------------------------------------------------------------
# New worker function and refactored main parsing function
# --------------------------------------------------------------------------


def process_file(filepath: str, seed: int) -> pd.DataFrame:
    """
    Worker function: Reads and processes a single JSON file.
    This function is executed by each thread in the pool.
    """
    try:
        filename = os.path.basename(filepath)
        df = pd.read_json(filepath)
        method_name, task_id, fold_number = parse_df_filename(filename)
        df["task"] = f"{task_id}_{fold_number}"
        df["method"] = method_name
        df["task_id"] = task_id
        df["fold"] = fold_number
        df["seed"] = seed
        if method_name == "GES":
            df["name"] = df["iteration"].apply(lambda x: f"{method_name}_{x}")
        return df
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def parse_dataframes(
    seeds: list[int], repo: EvaluationRepository, method_names: list[str]
) -> pd.DataFrame:
    """
    Parses dataframes from JSON files using a multithreaded approach for I/O.
    """
    metrics = repo.metrics(datasets=repo.datasets(), configs=repo.configs())
    results_path = "results"

    print("Start data loading...")

    # 1. First, collect all file paths to be processed
    tasks = []
    for seed in seeds:
        print(f"Scanning files for Seed: {seed}")
        filepath = f"{results_path}/seed_{seed}/"
        if not os.path.isdir(filepath):
            print(f"Warning: Directory not found for seed {seed}, skipping.")
            continue

        files = [
            os.path.join(filepath, f)
            for f in os.listdir(filepath)
            if f.endswith(".json") and any(name + "_" in f for name in method_names)
        ]
        # Pair each file with its seed number for the worker function
        for file in files:
            tasks.append((file, seed))

    print(f"\nFound {len(tasks)} files to process. Starting threaded loading...")

    all_dfs = []
    # 2. Use a ThreadPoolExecutor to read files in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        # map applies the function to each item in 'tasks' and returns an iterator of results
        # We use a lambda to unpack the (filepath, seed) tuple for our worker function
        results_iterator = executor.map(lambda p: process_file(*p), tasks)
        all_dfs = list(results_iterator)  # Consume the iterator to get all results

    print("File loading complete. Concatenating and calculating metrics...")

    if not all_dfs:
        print("Warning: No dataframes were loaded. Check paths and method names.")
        return pd.DataFrame()

    # 3. Concatenate all loaded dataframes into a single one
    df = pd.concat(all_dfs, ignore_index=True)

    # 4. Perform CPU-bound post-processing steps
    df["models_used"] = df["models_used"].apply(tuple)
    df["inference_time"] = df.apply(get_inference_time, axis=1, args=(repo, metrics))
    df["models_used"] = df["models_used"].apply(list)

    return df


# --------------------------------------------------------------------------
# Main execution block
# --------------------------------------------------------------------------

if __name__ == "__main__":
    repo = load_repository("D244_F3_C1530_100", cache=True)

    if False:  # Add multi-ges
        num_solutions = 20
        infer_time_weights = np.linspace(0, 1, num=num_solutions)
        infer_time_weights = np.round(infer_time_weights, 2)
        multi_ges_methods = [
            f"MULTI_GES-{time_weight:.2f}" for time_weight in infer_time_weights
        ]
    else:
        multi_ges_methods = []

    method_names = [
        "SINGLE_BEST",
        "GES",
        "MULTI_GES-0.21",
        "MULTI_GES-0.79",
        "QO",
        "QDO",
        "ENS_SIZE_QDO",
        "INFER_TIME_QDO",
        "DISK_QDO",
        "MEMORY_QDO",
    ]
    method_names.extend(multi_ges_methods)

    # The call to parse_dataframes now uses the multithreaded version
    df = parse_dataframes(
        seeds=[0],#, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        repo=repo,
        method_names=method_names,
    )

    if not df.empty:
        compute_total_resource_usage(df, load_resource_usage())
        normalize_per_dataset(df)
        df.reset_index(drop=True, inplace=True)

        if not os.path.exists("data"):
            os.makedirs("data")
        df.to_json("data/full.json")
        print("Done writing data to data/full.json...")
        df.to_csv("data/full.csv")

        print(f"df shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head())

        # Analysis part
        filtered_data = df[df["method"].isin(method_names)]

        avg_roc_auc = (
            filtered_data.groupby(["task", "method"])["roc_auc_val"].mean().unstack()
        )
        print(f"ROC AUC for validation set per method and dataset:\n{avg_roc_auc}\n")
        avg_roc_auc_test = (
            filtered_data.groupby(["task", "method"])["roc_auc_test"].mean().unstack()
        )
        print(f"ROC AUC for test set per method and dataset:\n{avg_roc_auc_test}\n")
        inference_time = (
            filtered_data.groupby(["task", "method"])["inference_time"].mean().unstack()
        )
        print(f"Inference time per method and dataset:\n{inference_time}\n")

        total_entries = df.shape[0]
        zero_infer_count = df[df["inference_time"] == 0].shape[0]
        if total_entries > 0:
            percentage_zero = (zero_infer_count / total_entries) * 100
            print(f"Total entries: {total_entries}")
            print(f"Entries with inference time of 0: {zero_infer_count}")
            print(f"Percentage of zero inference times: {percentage_zero:.2f}%")
    else:
        print("Script finished, but the final DataFrame is empty.")
