import math
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pygmo as pg
from autorank._util import get_sorted_rank_groups
from autorank import autorank, plot_stats

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
    configs = entry["models_used"]  # This is a tuple of configurations

    # Initialize total inference time
    total_inference_time = 0

    # Iterate over each configuration in the tuple
    for config in configs:
        # Select the metrics for each configuration
        if (dataset, fold, config) in metrics.index:
            selected_metrics = metrics.loc[(dataset, fold, config)]
            # Sum the 'time_infer_s' values for each configuration and add to total
            total_inference_time += selected_metrics["time_infer_s"].sum()

    return total_inference_time


# TODO: remove duplicate solutions
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
                df["seed"] = seed
                if method_name == "GES":  # Post processing due to wrong output
                    df["name"] = df.apply(
                        lambda row: f"{method_name}_{row['iteration']}", axis=1
                    )
                dfs_seed.append(df)

        all_dfs.extend(dfs_seed)

    # Concatenate all DataFrames from all seeds
    df = pd.concat(all_dfs)
    print(f"df shape: {df.shape}")

    # Remove duplicate solutions if models_used are identical for task, method and seed
    df["models_used"] = df["models_used"].apply(tuple)  # Make models_used hashable
    df = df.drop_duplicates(
        subset=["task", "seed", "method", "models_used"], keep="first"
    )

    # Remove entries where the number of models used is one or less
    # df = df[df["models_used"].apply(len) > 1]
    print(f"df (unique) shape: {df.shape}")

    # Inference times
    models_used = df["models_used"].explode().unique().tolist()
    datasets = [repo.task_to_dataset(task) for task in df["task"].unique()]
    metrics = repo.metrics(datasets=datasets, configs=models_used)
    df["inference_time"] = df.apply(get_inference_time, axis=1, args=(repo, metrics))

    # Group by folds first and aggregate
    fold_aggregation_functions = {
        "name": "first",
        "roc_auc_val": "mean",
        "roc_auc_test": "mean",
        "meta": "first",
        "task_id": "first",
        "method": "first",
        "inference_time": "mean",
    }

    df_grouped_by_fold = (
        df.groupby(["name", "task_id", "method", "seed"])
        .agg(fold_aggregation_functions)
        .reset_index(drop=True)
    )

    # Now group by seeds and aggregate
    seed_aggregation_functions = {
        "name": "first",
        "roc_auc_val": "mean",
        "roc_auc_test": "mean",
        "meta": "first",
        "task_id": "first",
        "method": "first",
        "inference_time": "mean",
    }

    df_grouped_by_seed = (
        df_grouped_by_fold.groupby(["name", "task_id", "method"])
        .agg(seed_aggregation_functions)
        .reset_index(drop=True)
    )

    # Check for duplicates
    duplicates = df_grouped_by_seed[
        df_grouped_by_seed.duplicated(subset=["name", "task_id"], keep=False)
    ]
    if not duplicates.empty:
        print("Found duplicates:")
        print(duplicates)

    return df_grouped_by_seed, metrics


def parse_single_best(repo) -> pd.DataFrame:
    with open("data/single_best_out2.txt", "r") as f:
        data = f.read()

    data_pattern = re.compile(
        r"Best Model (\w+) ROC AUC Test Score for (\d+_\d+): ([\d\.]+)\n"
        r"Best Model \w+ ROC AUC Validation Score for \d+_\d+: ([\d\.]+)"
    )
    matches = data_pattern.findall(data)

    # Prepare data for DataFrame
    df_records = []
    model_names = set(match[0] for match in matches)  # Collect unique model names
    task_ids = set(match[1] for match in matches)  # Collect unique task IDs

    # Retrieve metrics based on tasks and models
    datasets = [repo.task_to_dataset(task.split("_")[0]) for task in task_ids]
    metrics = repo.metrics(datasets=datasets, configs=list(model_names))

    for match in matches:
        task_id, fold = match[1].split("_")
        name = match[0]
        roc_auc_test = float(match[2])
        roc_auc_val = float(match[3])

        # Calculate inference time for the single best model
        dataset = repo.task_to_dataset(task_id)
        if (dataset, int(fold), name) in metrics.index:
            selected_metrics = metrics.loc[(dataset, int(fold), name)]
            inference_time = selected_metrics["time_infer_s"].sum()
        else:
            inference_time = 0  # Default to 0 if no metrics available

        df_records.append(
            {
                "name": name,
                "roc_auc_val": roc_auc_val,
                "roc_auc_test": roc_auc_test,
                "meta": 1,
                "task_id": task_id,
                "method": "SINGLE_BEST",
                "inference_time": inference_time,
            }
        )

    single_best_df = pd.DataFrame(df_records)
    return single_best_df


def normalize_data(data, high_is_better=True):
    """Normalize data to [0, 1] range; handle cases with NaN or zero range."""
    min_val = np.nanmin(data)  # Use nanmin to ignore NaNs
    max_val = np.nanmax(data)  # Use nanmax to ignore NaNs

    if np.isnan(min_val) or np.isnan(max_val):
        raise ValueError("Data contains NaN values which cannot be normalized.")

    if max_val == min_val:
        # Handle the case where all data points are the same or NaNs are present
        return np.zeros_like(data)

    normalized_data = (data - min_val) / (max_val - min_val)
    if high_is_better:
        normalized_data = 1 - normalized_data

    return normalized_data


def normalize_per_task_and_method(df):
    # Apply normalization based on 'method' and 'task_id'
    for method_name in df["method"].unique():
        for task_id in df[df["method"] == method_name]["task_id"].unique():
            mask = (df["method"] == method_name) & (df["task_id"] == task_id)
            df.loc[mask, "normalized_roc_auc"] = normalize_data(
                -df.loc[mask, "roc_auc_test"], high_is_better=False
            )
            df.loc[mask, "normalized_time"] = normalize_data(
                df.loc[mask, "inference_time"], high_is_better=False
            )
    return df


def boxplot(df: pd.DataFrame, y_str: str, log_y_scale: bool = False):
    if y_str not in df.columns:
        raise ValueError(f"Column '{y_str}' not found in DataFrame")

    # Plot ROC AUC scores for each method
    plt.figure(figsize=(14, 7))
    sns.boxplot(x="method", y=y_str, data=df, palette="pastel", linewidth=2)
    plt.title("Performance of Ensemble Methods: " + y_str)
    plt.ylabel(y_str)
    if log_y_scale:
        plt.yscale("log")
    plt.xlabel("Ensemble Method")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # save to file
    plt.savefig("plots/boxplot_" + y_str + ".png", dpi=300)


def is_pareto_efficient(costs, return_mask=True):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = not np.any(
            np.all(costs <= c, axis=1) & np.any(costs < c, axis=1)
        )
    return is_efficient if return_mask else costs[is_efficient]


def plot_pareto_front(df, method_name):
    df_method = df[df["method"] == method_name]
    hypervolumes = {}

    plot_dir = "plots/pareto_fronts/"
    os.makedirs(plot_dir, exist_ok=True)

    for task_id in df_method["task_id"].unique():
        df_task = df_method[df_method["task_id"] == task_id]

        plt.figure()
        plt.scatter(
            df_task["normalized_roc_auc"],
            df_task["normalized_time"],
            c="gray",
            alpha=0.5,
            label="All Solutions",
        )

        objectives = np.array(
            [df_task["normalized_roc_auc"].values, df_task["normalized_time"].values]
        ).T

        is_efficient = is_pareto_efficient(objectives)
        non_dominated = df_task.iloc[is_efficient]

        plt.scatter(
            non_dominated["normalized_roc_auc"],
            non_dominated["normalized_time"],
            c="blue",
            label="Pareto Front",
        )
        plt.title(f"Pareto Front for {task_id} using Normalized Scores")
        plt.xlabel("Normalized Negated ROC AUC Test")
        plt.ylabel("Normalized Inference Time")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}{task_id}_{method_name}_pareto_front.png")
        plt.close()

        ref_point = [
            1,  # reference point beyond the worst values of objectives for normalized scores
            1,
        ]
        hv = pg.hypervolume(objectives[is_efficient])
        hypervolume = hv.compute(ref_point)
        hypervolumes[task_id] = hypervolume

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
    # Prepare the data for plotting
    methods = list(all_hypervolumes.keys())  # Method names
    hv_values = [list(all_hypervolumes[method].values()) for method in methods]
    data = []

    # Creating a DataFrame suitable for Seaborn
    for method_index, values in enumerate(hv_values):
        for value in values:
            data.append({"Method": methods[method_index], "Hypervolume": value})
    df = pd.DataFrame(data)

    # Set the figure size and style
    sns.set(
        style="whitegrid"
    )  # Set the background to a white grid for better visibility
    plt.figure(figsize=(10, 6))

    # Use seaborn's boxplot to plot the DataFrame and create an axis object
    ax = sns.boxplot(
        x="Method", y="Hypervolume", data=df, palette="Set2"
    )  # You can choose a palette that fits your taste

    # Set titles and labels using the axis object
    ax.set_title("Comparison of Hypervolume by Method", fontsize=16, fontweight="bold")
    ax.set_xlabel("Method", fontsize=14)
    ax.set_ylabel("Hypervolume", fontsize=14)

    # Set font size for ticks directly on the axes
    ax.tick_params(
        axis="x", labelrotation=45, labelsize=12
    )  # Rotate x-ticks to avoid overlap
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()  # Adjust the plot to fit into the figure area nicely
    plt.savefig(
        "hypervolume_comparison.png", dpi=300
    )  # Save the figure with high resolution
    plt.close()


def _custom_cd_diagram(result, reverse, ax, width):
    """
    !TAKEN FROM AUTORANK WITH MODIFICATIONS!
    """

    def plot_line(line, color="k", **kwargs):
        ax.plot(
            [pos[0] / width for pos in line],
            [pos[1] / height for pos in line],
            color=color,
            **kwargs,
        )

    def plot_text(x, y, s, *args, **kwargs):
        ax.text(x / width, y / height, s, *args, **kwargs)

    sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse)
    cd = result.cd

    lowv = min(1, int(math.floor(min(sorted_ranks))))
    highv = max(len(sorted_ranks), int(math.ceil(max(sorted_ranks))))
    cline = 0.4
    textspace = 1
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            relative_rank = rank - lowv
        else:
            relative_rank = highv - rank
        return textspace + scalewidth / (highv - lowv) * relative_rank

    linesblank = 0.2 + 0.2 + (len(groups) - 1) * 0.1

    # add scale
    distanceh = 0.1
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((len(sorted_ranks) + 1) / 2) * 0.2 + minnotsignificant

    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    plot_line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        plot_line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=0.7)

    for a in range(lowv, highv + 1):
        plot_text(rankpos(a), cline - tick / 2 - 0.05, str(a), ha="center", va="bottom")

    for i in range(math.ceil(len(sorted_ranks) / 2)):
        chei = cline + minnotsignificant + i * 0.2
        plot_line(
            [
                (rankpos(sorted_ranks[i]), cline),
                (rankpos(sorted_ranks[i]), chei),
                (textspace - 0.1, chei),
            ],
            linewidth=0.7,
        )
        plot_text(textspace - 0.2, chei, names[i], ha="right", va="center")

    for i in range(math.ceil(len(sorted_ranks) / 2), len(sorted_ranks)):
        chei = cline + minnotsignificant + (len(sorted_ranks) - i - 1) * 0.2
        plot_line(
            [
                (rankpos(sorted_ranks[i]), cline),
                (rankpos(sorted_ranks[i]), chei),
                (textspace + scalewidth + 0.1, chei),
            ],
            linewidth=0.7,
        )
        plot_text(textspace + scalewidth + 0.2, chei, names[i], ha="left", va="center")

    # upper scale
    if not reverse:
        begin, end = rankpos(lowv), rankpos(lowv + cd)
    else:
        begin, end = rankpos(highv), rankpos(highv - cd)
    distanceh += 0.15
    bigtick /= 2
    plot_line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
    plot_line(
        [(begin, distanceh + bigtick / 2), (begin, distanceh - bigtick / 2)],
        linewidth=0.7,
    )
    plot_line(
        [(end, distanceh + bigtick / 2), (end, distanceh - bigtick / 2)], linewidth=0.7
    )
    plot_text((begin + end) / 2, distanceh - 0.05, "CD", ha="center", va="bottom")

    # no-significance lines
    side = 0.05
    no_sig_height = 0.1
    start = cline + 0.2
    for l, r in groups:
        plot_line(
            [
                (rankpos(sorted_ranks[l]) - side, start),
                (rankpos(sorted_ranks[r]) + side, start),
            ],
            linewidth=2.5,
        )
        start += no_sig_height

    return ax


def cd_evaluation(
    hypervolumes,
    maximize_metric=True,
    plt_title="Critical Difference Plot",
):
    """
    hypervolumes: DataFrame with method names as columns and tasks as rows, each cell contains hypervolume.
    maximize_metric: Boolean, True if higher values are better.
    output_path: Where to save the plot, if None, plot will not be saved.
    plt_title: Title of the plot.
    """
    # Prepare data
    rank_data = -hypervolumes if maximize_metric else hypervolumes

    # Run autorank
    result = autorank(rank_data, alpha=0.05, verbose=False, order="ascending")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_stats(result, ax=ax)
    plt.title(plt_title)
    plt.tight_layout()

    plt.savefig("critical_difference.png", bbox_inches="tight", dpi=300)
    plt.close()

    return result


if __name__ == "__main__":
    reload = True
    if reload:
        repo = load_repository("D244_F3_C1530_100", cache=True)
        df = parse_dataframes([0, 1, 2], repo=repo)
        single_best_df = parse_single_best(repo)
        df = pd.concat([df, single_best_df], ignore_index=True)

        df.reset_index(drop=True, inplace=True)
        df.to_json("data/full.json")
    else:
        df = pd.read_json("data/full.json")
        normalize_per_task_and_method(df)
        print(df.head())
        print(df.shape)

        methods = ["GES", "QO", "QDO", "ENS_SIZE_QDO", "INFER_TIME_QDO"]
        all_hypervolumes = {}

        for method in methods:
            all_hypervolumes[method] = plot_pareto_front(df, method)

        print("Plotting hypervolumes...")
        plot_hypervolumes(all_hypervolumes)

        # Create a DataFrame for all hypervolumes
        hypervolumes_df = pd.DataFrame(all_hypervolumes)

        # Save the DataFrame as a CSV file
        print("Saving hypervolume csv...")
        hypervolumes_df.to_csv("hypervolumes.csv", index=False)

        data = []
        for method, tasks in all_hypervolumes.items():
            for task_id, hypervolume in tasks.items():
                data.append(
                    {"Task": task_id, "Method": method, "Hypervolume": hypervolume}
                )

        df_hypervolumes = pd.DataFrame(data)
        pivot_hypervolumes = df_hypervolumes.pivot(
            index="Task", columns="Method", values="Hypervolume"
        )

        # Now you can use the modified cd_evaluation function
        result = cd_evaluation(pivot_hypervolumes, maximize_metric=True)
        print(df.columns)

        # Plot boxplot for inference time and performance
        df["normalized_roc_auc"] = 1 - df["normalized_roc_auc"]
        boxplot(df, "normalized_time")
        boxplot(df, "normalized_roc_auc")

    # main()
