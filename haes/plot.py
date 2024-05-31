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


def parse_dataframes(
    seeds: list[int], repo: EvaluationRepository
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = repo.metrics(datasets=repo.datasets(), configs=repo.configs())
    results_path = "results"
    all_dfs = []
    print("Start data loading...")
    for seed in seeds:
        print(f"Seed: {seed}")
        df_list_seed = []
        filepath = f"{results_path}/seed_{seed}/"
        files = [f for f in os.listdir(filepath) if f.endswith(".json")]
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
        all_dfs.append(df_seed)

    df = pd.concat(all_dfs)
    df.drop(columns=["task", "iteration", "weights", "models_used", "dataset", "meta"])

    print(f"df shape: {df.shape}")
    print(df.columns)
    print(df.groupby("method").agg("mean")["roc_auc_val"])
    print(df.groupby("method").agg("mean")["roc_auc_test"])

    return df


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


def normalize_per_task_method_and_seed(df):
    # Apply normalization based on 'method', 'task', and 'seed'
    for method_name in df["method"].unique():
        for task in df[df["method"] == method_name]["task"].unique():
            for seed in df[(df["method"] == method_name) & (df["task"] == task)][
                "seed"
            ].unique():
                mask = (
                    (df["method"] == method_name)
                    & (df["task"] == task)
                    & (df["seed"] == seed)
                )
                df.loc[mask, "normalized_roc_auc"] = normalize_data(
                    -df.loc[mask, "roc_auc_test"], high_is_better=False
                )
                df.loc[mask, "normalized_time"] = normalize_data(
                    df.loc[mask, "inference_time"], high_is_better=False
                )
    return df


def boxplot(
    df: pd.DataFrame, y_str: str, log_y_scale: bool = False, flip_y_axis: bool = False
):
    if y_str not in df.columns:
        raise ValueError(f"Column '{y_str}' not found in DataFrame")

    # Plot ROC AUC scores for each method
    plt.figure(figsize=(14, 7))
    sns.boxplot(x="method", y=y_str, data=df, palette="pastel", linewidth=2)
    plt.title("Performance of Ensemble Methods: " + y_str)
    plt.ylabel(y_str)
    if log_y_scale:
        plt.yscale("log")
    if flip_y_axis:
        plt.gca().invert_yaxis()
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


def calculate_average_hypervolumes(df, method_name):
    df_method = df[df["method"] == method_name]
    hypervolumes = {}

    # Iterate over unique task_ids
    for task_id in df_method["task_id"].unique():
        seed_hypervolumes = []  # Store hypervolumes for each seed

        for seed in df_method["seed"].unique():
            fold_hypervolumes = []  # Store hypervolumes for each fold under the current seed

            for fold in df_method["fold"].unique():
                df_fold = df_method[
                    (df_method["task_id"] == task_id)
                    & (df_method["seed"] == seed)
                    & (df_method["fold"] == fold)
                ]

                objectives = np.array(
                    [
                        df_fold["normalized_roc_auc"].values,
                        df_fold["normalized_time"].values,
                    ]
                ).T
                is_efficient = is_pareto_efficient(objectives)
                efficient_objectives = objectives[is_efficient]

                ref_point = [
                    1,
                    1,
                ]  # Reference point beyond the worst values of objectives
                hv = pg.hypervolume(efficient_objectives)
                hypervolume = hv.compute(ref_point)
                fold_hypervolumes.append(hypervolume)

            # Average hypervolumes across all folds for a given seed
            if fold_hypervolumes:
                average_fold_hypervolume = np.mean(fold_hypervolumes)
                seed_hypervolumes.append(average_fold_hypervolume)

        # Average the averaged fold hypervolumes across seeds
        if seed_hypervolumes:
            average_seed_hypervolume = np.mean(seed_hypervolumes)
            hypervolumes[task_id] = average_seed_hypervolume

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
    filename="CriticalDifferencePlot.png",
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

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

    return result


if __name__ == "__main__":
    reload = False
    if reload:
        repo = load_repository("D244_F3_C1530_100", cache=True)
        df = parse_dataframes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], repo=repo)
        print(df["method"].unique())
        df.reset_index(drop=True, inplace=True)
        df.to_json("data/full.json")
    else:
        print("Loading data. This might take a while...")
        df = pd.read_json("data/full.json")
        normalize_per_task_method_and_seed(df)
        print(df.head())
        print(df.columns)
        print(df["method"].unique())

        # Hypervolume
        methods = ["GES", "QO", "QDO", "ENS_SIZE_QDO", "INFER_TIME_QDO"]
        all_hypervolumes = {}

        for method in methods:
            all_hypervolumes[method] = calculate_average_hypervolumes(df, method)

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
        result = cd_evaluation(
            pivot_hypervolumes, maximize_metric=True, plt_title="Hypervolume Critical Difference Plot" , filename="CDPHypervolumes.png"
        )
        print(df.columns)
        
        # DF with the best solution per task_id, fold, seed and method
        print("Picking best solutions...")
        best_val_scores = df.loc[df.groupby(['task_id', 'method', 'fold', 'seed'])['roc_auc_val'].idxmax()]
        print("Averaging over folds...")
        avg_over_folds = best_val_scores.groupby(['task_id', 'method', 'seed']).agg({
            'roc_auc_val': 'mean',
            'roc_auc_test': 'mean',
            'inference_time': 'mean'
        }).reset_index()
        print("Averaging over seeds...")
        avg_over_seeds = avg_over_folds.groupby(['task_id', 'method']).agg({
            'roc_auc_val': 'mean',
            'roc_auc_test': 'mean',
            'inference_time': 'mean'
        }).reset_index()

        # Plot boxplot for inference time and performance
        print(f"Shape after averaging: {avg_over_seeds.shape}")
        boxplot(avg_over_seeds, "inference_time", log_y_scale=True)

        # Rank data within each task based on 'roc_auc_test' and add as a new column
        avg_over_seeds['rank'] = avg_over_seeds.groupby('task_id')['roc_auc_test'].rank("dense", ascending=False)
        boxplot(avg_over_seeds, "rank", flip_y_axis=True)
        pivot_ranks = avg_over_seeds.pivot(index="task_id", columns="method", values='rank')
        cd_evaluation(pivot_ranks, maximize_metric=False, plt_title="Rankings Critical Difference Plot" , filename="CDPRankings.png")

    # main()
