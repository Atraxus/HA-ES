import argparse
import re
import pandas as pd
import numpy as np
import os

from tabarena.repository import EvaluationRepositoryCollection

from tabarena_data import (
    DEFAULT_METHODS,
    config_resource_usage,
    load_repo,
    load_resource_usage,
)


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


def compute_total_resource_usage(df, repo: EvaluationRepositoryCollection):
    """Compute total memory and diskspace usage for each ensemble in the dataframe."""
    df_usage = load_resource_usage()
    configs_type = repo.configs_type()

    def get_total_usage(models_used):
        total_memory = 0
        total_diskspace = 0
        for config in models_used:
            memory, diskspace = config_resource_usage(
                df_usage, config, configs_type.get(config)
            )
            total_memory += memory
            total_diskspace += diskspace
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
    seeds: list[int], repo: EvaluationRepositoryCollection, method_names: list[str]
) -> pd.DataFrame:
    metrics = repo.metrics()
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
            if f.endswith(".json")
            and any(f.startswith(name + "_") for name in method_names)
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
    parser = argparse.ArgumentParser(description="Aggregate the results of generate_data.py.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="TabArena methods used as base models when generating the results.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()

    repo = load_repo(args.methods)
    seeds = args.seeds

    # --- Method groups ---
    base = ["SINGLE_BEST", "GES", "QDO"]
    ours = ["MEMORY_QDO", "MULTI_GES-0.68"]
    variants = ["INFER_TIME_QDO", "ENS_SIZE_QDO", "DISK_QDO", "MEMORY_QDO"]
    extra = ["QO"]

    # All MULTI_GES versions
    num_solutions = 20
    infer_time_weights = np.round(np.linspace(0, 1, num=num_solutions), 2)
    infer_time_weights = np.delete(infer_time_weights, 0)
    multi_ges_all = [f"MULTI_GES-{w:.2f}" for w in infer_time_weights]

    # --- Parse everything once (raw metrics) ---
    all_methods = list(set(base + ours + variants + extra + multi_ges_all))
    df_raw = (
        parse_dataframes(seeds, repo, all_methods)
        .pipe(compute_total_resource_usage, repo)
        .reset_index(drop=True)
    )

    if not os.path.exists("data"):
        os.makedirs("data")

    # --- Normalize globally (all methods) ---
    df_globalnorm = normalize_per_dataset(df_raw)

    # --- Save combinations ---
    filters = {
        "base_ours": base + ours,
        "base_variants": base + variants,
        "base_ours_variants_extra": base + ours + variants + extra,
        "ges_multi": ["SINGLE_BEST", "GES"] + multi_ges_all,
    }

    for name, methods in filters.items():
        # --- Global normalization version ---
        df_g = df_globalnorm[df_globalnorm["method"].isin(methods)].reset_index(drop=True)
        df_g.to_csv(f"data/{name}_globalnorm.csv", index=False)

        # --- Local normalization version ---
        df_l = df_raw[df_raw["method"].isin(methods)].reset_index(drop=True)
        df_l = normalize_per_dataset(df_l)
        df_l.to_csv(f"data/{name}_localnorm.csv", index=False)

        print(
            f"Saved {name}: "
            f"{df_g['method'].nunique()} methods, "
            f"{len(df_g)} rows "
            f"(globalnorm & localnorm)"
        )