import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pygmo as pg

from tabrepo import load_repository, EvaluationRepository
import os


def parse_data(filepath: str) -> tuple[dict, int, int]:
    with open(filepath, "r") as f:
        data = f.read()

    # count and remove lines starting with 'Error'
    n_errors = data.count("Error")
    for i in range(n_errors):
        start = data.find("Error")
        end = data.find("\n", start)
        data = data[:start] + data[end + 1 :]

    # count lines with 'Fitting ENS_SIZE_QDO'  to know number of tasks
    n_tasks = data.count("Fitting ENS_SIZE_QDO")

    ensemble_data = []
    pattern = (
        r"Fitting (\w+) for task (\d+_\d+), dataset \b[\w-]+\b, fold \d+\s+"
        r"ROC AUC Validation Score for \d+_\d+: [\d\.]+\s+"
        r"ROC AUC Test Score for \d+_\d+: ([\d\.]+)\s+"
        r"Number of different base models in the ensemble: (\d+)\s+"
        r"Models used: \[([^\]]+)\]"
    )
    data_pattern = re.compile(pattern, re.DOTALL)
    matches = data_pattern.findall(data)
    for match in matches:
        models_used = match[4].replace("'", "").replace('"', "").strip().split(", ")
        ensemble_data.append(
            {
                "method": match[0],
                "task": match[1],
                "roc_auc": float(match[2]),
                "n_base_models": int(match[3]),
                "models_used": models_used,
            }
        )

    # Check if n_tasks is correct
    assert len(ensemble_data) == n_tasks * 5  # w/o single best model

    return ensemble_data, n_errors, n_tasks


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

    # Ensure that the metrics DataFrame is indexed or filtered efficiently
    selected_metrics = metrics.loc[(dataset, fold, configs)]

    # Sum the 'time_infer_s' values
    total_inference_time = selected_metrics["time_infer_s"].sum()
    return total_inference_time


def parse_dataframes(
    seeds: list[int], repo: EvaluationRepository
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_path = "results"
    all_dfs = []
    for seed in seeds:
        dfs_seed = []
        filepath = f"{results_path}/seed_{seed}/"
        for file in os.listdir(filepath):
            if file.endswith(".json"):
                df = pd.read_json(filepath + file)
                method_name, task_id, fold_number = parse_df_filename(file)
                df["method"] = method_name
                df["task"] = f"{task_id}_{fold_number}"
                df["task_id"] = task_id
                df["fold_number"] = fold_number
                df["seed"] = seed
                # df = df.sort_values(by=["roc_auc_test", "roc_auc_val"], ascending=False)
                dfs_seed.append(df)
        all_dfs.extend(dfs_seed)

    # Concatenate all DataFrames from all seeds
    df = pd.concat(all_dfs)

    models_used = df["models_used"].explode().unique().tolist()
    datasets = [repo.task_to_dataset(task) for task in df["task"].unique()]
    metrics = repo.metrics(datasets=datasets, configs=models_used)

    # Calculate inference times
    df["inference_time"] = df.apply(get_inference_time, axis=1, args=(repo, metrics))

    # Group by excluding 'seed' and aggregate
    aggregation_functions = {
        "name": "first",
        "roc_auc_val": "mean",
        "roc_auc_test": "mean",
        "models_used": "first",  # Consider removing if no longer necessary
        "weights": "first",
        "meta": "first",
        "task": "first",
        "dataset": "first",
        "fold": "first",
        "method": "first",
        "task_id": "first",
        "fold_number": "first",
        "inference_time": "mean",  # Added inference time to aggregation
    }

    df_grouped = (
        df.groupby(
            ["name", "task", "dataset", "fold", "method", "task_id", "fold_number"]
        )
        .agg(aggregation_functions)
        .reset_index(drop=True)
    )

    # Select only the best per method and task, taking the first entry which should be the best due to prior sorting
    only_best_df = df.groupby(["method", "task"]).first().reset_index()

    return df_grouped, only_best_df


def parse_single_best() -> dict:
    with open("data/single_best_out2.txt", "r") as f:
        data = f.read()

    # Example: Best Model RandomForest_r114_BAG_L1 ROC AUC Test Score for 3616_2: 0.6602564102564102
    data_pattern = re.compile(
        r"Best Model (\w+) ROC AUC Test Score for (\d+_\d+): ([\d\.]+)\n"
        r"Best Model \w+ ROC AUC Validation Score for \d+_\d+: ([\d\.]+)"
    )
    matches = data_pattern.findall(data)

    single_best = {}
    for match in matches:
        single_best[match[1]] = (match[0], float(match[2]), float(match[3]))

    return single_best


def boxplot(df: pd.DataFrame, y_str: str):
    if y_str not in df.columns:
        raise ValueError(f"Column '{y_str}' not found in DataFrame")

    # Plot ROC AUC scores for each method
    plt.figure(figsize=(14, 7))
    sns.boxplot(x="method", y=y_str, data=df, palette="pastel", linewidth=2)
    plt.title("Performance of Ensemble Methods: " + y_str)
    plt.ylabel(y_str)
    plt.xlabel("Ensemble Method")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # save to file
    plt.savefig("plots/performance_" + y_str + ".png", dpi=300)


def boxplot_ranking(df: pd.DataFrame):
    rankings = []
    for method in df["method"].unique():
        rankings.append(df[df["method"] == method]["dataset_rank"].values)

    x_ticks = df["method"].unique()

    # Plot ranking for each method
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=rankings, palette="pastel", linewidth=2)
    plt.title("Ranking of Ensemble Methods: " + "dataset_rank")
    plt.ylabel("dataset_rank")
    plt.gca().invert_yaxis()
    plt.xlabel("Ensemble Method")
    plt.xticks(rotation=45)
    plt.xticks(range(len(x_ticks)), x_ticks)
    plt.grid(True)
    plt.tight_layout()

    # save to file
    plt.savefig("plots/performance_" + "dataset_rank" + ".png", dpi=300)


def normalized_improvement(df: pd.DataFrame):
    # Assuming 'method', 'task', and 'roc_auc' are columns in df
    df["dataset"] = df["task"].apply(lambda x: x.split("_")[0])

    # Compute the minimum and range of ROC AUC per dataset
    stats = df.groupby("dataset")["roc_auc"].agg(["min", "max"]).reset_index()
    stats["range"] = stats["max"] - stats["min"]

    # Merge these stats back into the original dataframe
    df = df.merge(stats[["dataset", "min", "range"]], on="dataset", how="left")

    # Calculate normalized improvement
    df["normalized_improvement"] = df.apply(
        lambda x: (x["roc_auc"] - x["min"]) / x["range"] if x["range"] != 0 else 0,
        axis=1,
    )

    return df


def create_ranking_for_dataset(df: pd.DataFrame):
    # Split 'task' into 'task_id' (dataset ID) and 'fold'
    df["task_id"] = df["task"].apply(lambda x: x.split("_")[0])
    df["fold"] = df["task"].apply(lambda x: x.split("_")[1])

    # Calculate average ROC AUC for each method within each dataset
    avg_scores = (
        df.groupby(["task_id", "method"])["roc_auc"]
        .mean()
        .reset_index(name="roc_auc_avg")
    )

    # Rank methods within each dataset based on their average ROC AUC
    avg_scores["dataset_rank"] = avg_scores.groupby("task_id")["roc_auc_avg"].rank(
        ascending=False, method="dense"
    )

    # Merge these ranks back to the original dataframe
    df = df.merge(
        avg_scores[["task_id", "method", "dataset_rank"]],
        on=["task_id", "method"],
        how="left",
    )

    return df


def normalize_per_task(df_merged: pd.DataFrame):
    # Normalize ROC AUC scores per task_id
    df_merged["roc_auc_normalized"] = df_merged.groupby("task_id")[
        "roc_auc_test"
    ].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    return df_merged


def is_pareto_efficient(costs, return_mask=True):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        # Check if no point is strictly better than `c`.
        is_efficient[i] = np.all(
            np.any(costs != c, axis=1) | np.any(costs <= c, axis=1)
        )
    return is_efficient if return_mask else costs[is_efficient]


def plot_pareto_front(df, method_name):
    df_method = df[df["method"] == method_name]
    hypervolumes = {}

    plot_dir = "plots/pareto_fronts/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for task in df_method["task"].unique():
        df_task = df_method[df_method["task"] == task]
        plt.figure()
        plt.scatter(
            df_task["roc_auc_test"],
            -df_task["inference_time"],
            c="gray",
            alpha=0.5,
            label="All Solutions",
        )

        # Convert objectives to numpy array
        objectives = np.array(
            [df_task["roc_auc_test"].values, -df_task["inference_time"].values]
        ).T

        # Calculate non-dominated points
        is_efficient = np.isin(
            np.arange(len(objectives)), find_non_dominated(objectives)
        )
        non_dominated = df_task[is_efficient]

        plt.scatter(
            non_dominated["roc_auc_test"],
            -non_dominated["inference_time"],
            c="blue",
            label="Pareto Front",
        )
        plt.title(f"Pareto Front for {task} using Test Scores")
        plt.xlabel("ROC AUC Test")
        plt.ylabel("Inference Time")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}{task}_{method_name}_pareto_front.png")
        plt.close()

        # Ensure the reference point is beyond the worst values for all objectives
        ref_point = np.max(objectives, axis=0) + 1
        assert np.all(
            ref_point > np.max(objectives, axis=0)
        ), "Reference point is not set beyond the worst values of objectives."

        hv = pg.hypervolume(objectives[is_efficient])
        hypervolume = hv.compute(ref_point)
        hypervolumes[task] = hypervolume

    return hypervolumes


def find_non_dominated(points):
    """Identify the indices of non-dominated points."""
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        is_efficient[i] = not np.any(
            np.all(points <= c, axis=1) & np.any(points < c, axis=1)
        )
    return np.where(is_efficient)[0]


def plot_hypervolumes(all_hypervolumes):
    # TODO: create table with hypervolume per dataset (100 entries)
    # Prepare the data for plotting
    methods = list(all_hypervolumes.keys())  # Method names
    hv_values = [list(all_hypervolumes[method].values()) for method in methods]
    data = []

    # Creating a DataFrame suitable for Seaborn
    for method_index, values in enumerate(hv_values):
        for value in values:
            data.append({
                'Method': methods[method_index],
                'Hypervolume': value
            })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots()

    # Use seaborn's violinplot to plot the DataFrame
    sns.violinplot(x='Method', y='Hypervolume', data=df, ax=ax, cut=0)

    ax.set_title("Hypervolume by Method")
    ax.set_ylabel("Hypervolume")
    ax.set_yscale("log")  # Using logarithmic scale for better visualization of wide-ranging data

    # Annotate max and average values on the plot
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]['Hypervolume']
        max_value = method_data.max()
        avg_value = method_data.mean()
        # Display max value
        ax.text(i, max_value, f'Max: {max_value:.2f}', color='red', ha='center', va='bottom')
        # Display average value
        ax.text(i, avg_value, f'Avg: {avg_value:.2f}', color='blue', ha='center', va='bottom')

    plt.xticks(rotation=45)  # Rotate method names for better visibility
    plt.tight_layout()

    # Save the plot instead of showing it interactively
    plt.savefig("hypervolume_comparison.png", dpi=300)
    plt.close()



if __name__ == "__main__":
    # df contains all solutions the methods created during optimization
    # only_best_df contains only the best solution for each task
    # GES and single best only produce one solution per task
    reload = False
    if reload:
        repo = load_repository("D244_F3_C1530_100", cache=True)
        df, only_best_df = parse_dataframes([0, 1, 2], repo=repo)

        # Hash for dataset and models used
        # metrics = repo.metrics(datasets=datasets, configs=models_used)
        # average_inference_time = metrics["time_infer_s"].mean()
        # df["inference_time"] = df.apply(
        #     get_inference_time, axis=1, args=(repo, metrics)
        # )
        df = normalize_per_task(df)

        df.reset_index(drop=True, inplace=True)
        df["task"] = df["task"].astype(str)
        df.to_json("data/full.json")
    else:
        df = pd.read_json("data/full.json")
        # add _ before last digit in all tasks
        df["task"] = df["task"].astype(str).apply(lambda x: x[:-1] + "_" + x[-1])

        methods = ["QO", "QDO", "ENS_SIZE_QDO", "INFER_TIME_QDO"]
        all_hypervolumes = {}

        for method in methods:
            all_hypervolumes[method] = plot_pareto_front(df, method)

        plot_hypervolumes(all_hypervolumes)

    # main()
