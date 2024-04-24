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


def parse_dataframes(seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_path = "results"
    # Load data from all seeds
    dfs = []
    for seed in seeds:
        filepath = f"{results_path}/seed_{seed}/"
        # for all files in the directory
        for file in os.listdir(filepath):
            if file.endswith(".json"):
                df = pd.read_json(filepath + file)
                method_name, task_id, fold_number = parse_df_filename(file)
                df["method"] = method_name
                df["task"] = f"{task_id}_{fold_number}"
                df["task_id"] = task_id
                df["fold_number"] = fold_number
                df["seed"] = seed
                # Sort for first roc_auc_test and then roc_auc_val
                df = df.sort_values(by=["roc_auc_test", "roc_auc_val"], ascending=False)
                dfs.append(df)

    # TODO Average over seeds and keep models_used and weight vectors intact
    # df = pd.concat(dfs)
    # df = df.groupby(["method", "task_id", "fold_number"]).mean().reset_index()
    # print(df)

    df = pd.concat(dfs)
    only_best_df = df.groupby(["method", "task"]).first().reset_index()

    return df, only_best_df


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


def get_inference_time(
    entry, repo: EvaluationRepository, metrics: pd.DataFrame
) -> float:
    fold = int(entry["fold"])
    dataset = repo.task_to_dataset(entry["task"])
    configs = entry["models_used"]

    # Use .loc to select all relevant rows at once
    selected_metrics = metrics.loc[(dataset, fold, configs)]

    # Sum the 'time_infer_s' values
    total_inference_time = selected_metrics["time_infer_s"].sum()
    return total_inference_time


def is_pareto_efficient(costs, return_mask=True):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        # None of the other points should have both lower inference time and higher ROC AUC.
        is_efficient[i] = np.all(np.any(costs[:i] >= c, axis=1)) and np.all(
            np.any(costs[i + 1 :] >= c, axis=1)
        )
    return is_efficient if return_mask else costs[is_efficient]


def plot_pareto_front(df, method_name):
    df_method = df[df["method"] == method_name]

    # Ensure directory exists
    plot_dir = "plots/pareto_fronts/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Iterate over each unique task
    for task in df_method["task"].unique():
        # Filter data for the current task
        df_task = df_method[df_method["task"] == task]

        # Prepare the plot
        plt.figure()
        plt.scatter(
            df_task["roc_auc_test"],
            -df_task["inference_time"],
            c="gray",
            alpha=0.5,
            label="All Solutions",
        )

        # Identify non-dominated points
        objectives = np.array(
            [df_task["roc_auc_test"].values, -df_task["inference_time"].values]
        ).T
        is_efficient = np.isin(
            np.arange(len(objectives)), find_non_dominated(objectives)
        )

        # Extract non-dominated points
        non_dominated = df_task[is_efficient]

        # Plotting non-dominated points distinctly
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

        # Save plot
        plt.tight_layout()
        plt.savefig(f"{plot_dir}{task}_{method_name}_pareto_front.png")
        plt.close()

        # Compute hypervolume
        ref_point = np.max(objectives, axis=0) + 1  # Define a reference point
        hv = pg.hypervolume(objectives[is_efficient])
        hypervolume = hv.compute(ref_point)
        print(
            f"Hypervolume for {task} under method {method_name} using Test Scores: {hypervolume}"
        )


def find_non_dominated(points):
    """Identify the indices of non-dominated points"""
    is_dominated = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if all(points[j] <= points[i]) and any(points[j] < points[i]):
                is_dominated[i] = True
                break
    return np.where(~is_dominated)[0]


def main():
    data, n_errors, n_tasks = parse_data("20240419_0917_out.txt")
    n_configs = 1530
    print(f"Error count: {n_errors}, error rate: {n_errors/(n_tasks*n_configs):.5f}")

    # Find tasks with the worst ROC AUC
    df = pd.DataFrame(data)
    single_best = parse_single_best()

    # Prepare single best model data for merging
    single_best_data = []
    for task, (model, test_score, val_score) in single_best.items():
        single_best_data.append(
            {
                "method": "Single Best",
                "task": task,
                "roc_auc": test_score,
                "roc_auc_val": val_score,
                "models_used": [model],
                "n_base_models": 1,
            }
        )
    # Convert list of single best data into DataFrame
    df_single_best = pd.DataFrame(single_best_data)

    # Concatenate the original data with the single best model data
    df = pd.concat([df, df_single_best], ignore_index=True)

    # Sort over tasks
    df = df.sort_values(by=["task", "method"])

    df = normalized_improvement(df)
    df = create_ranking_for_dataset(df)

    # Load data from tabrepo
    context_name = "D244_F3_C1530_30"
    repo: EvaluationRepository = load_repository(context_name, cache=True)
    df["inference_time"] = df.apply(get_inference_time, axis=1, args=(repo,))

    # Remove task_id 3616
    df = df[df["task_id"] != "3616"]

    # Pareto efficiency
    # Exclude single best
    df = df[df["method"] != "Single Best"]

    # Get the pastel color palette with as many colors as there are methods
    methods = df["method"].unique()
    palette = sns.color_palette("pastel", len(methods))
    color_dict = dict(zip(methods, palette))

    # Iterate through each method and plot the Pareto front
    for method_name in methods:
        plot_pareto_front(df, method_name, color_dict)

    df_merged = (
        df.groupby(["task_id", "method"])
        .agg(
            {
                "dataset_rank": "mean",  # or 'first' since all should be identical
                "roc_auc": "mean",  # Average ROC AUC across folds
                "inference_time": "mean",  # Average inference time across folds
            }
        )
        .reset_index()
    )
    df_merged = normalize_per_task(df_merged)

    boxplot(df_merged, "roc_auc_normalized")
    boxplot(df, "normalized_improvement")
    df_merged_infertime_no_outliers = df_merged[
        (df_merged["inference_time"] < 1.5) & (df_merged["roc_auc_normalized"] > 0.6)
    ]
    boxplot(df_merged_infertime_no_outliers, "inference_time")
    boxplot_ranking(df_merged)


if __name__ == "__main__":
    # df contains all solutions the methods created during optimization
    # only_best_df contains only the best solution for each task
    # GES and single best only produce one solution per task
    reload = False
    if reload:
        repo = load_repository("D244_F3_C1530_30", cache=True)
        df, only_best_df = parse_dataframes([0])

        models_used = df["models_used"].explode().unique().tolist()
        datasets = [repo.task_to_dataset(task) for task in df["task"].unique()]

        # Hash for dataset and models used
        metrics = repo.metrics(datasets=datasets, configs=models_used)
        average_inference_time = metrics["time_infer_s"].mean()

        df["inference_time"] = df.apply(
            get_inference_time, axis=1, args=(repo, metrics)
        )
        df = normalize_per_task(df)

        df.reset_index(drop=True, inplace=True)
        df["task"] = df["task"].astype(str)
        df.to_json("data/full.json")
    else:
        df = pd.read_json("data/full.json")
        # add _ before last digit in all tasks
        df["task"] = df["task"].astype(str).apply(lambda x: x[:-1] + "_" + x[-1])

        plot_pareto_front(df, "QO")
        plot_pareto_front(df, "QDO")
        plot_pareto_front(df, "ENS_SIZE_QDO")
        plot_pareto_front(df, "INFER_TIME_QDO")

    # main()
