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

method_id_name_dict = {
    "GES": "GES",
    "QO": "QO-ES",
    "QDO": "QDO-ES",
    "ENS_SIZE_QDO": "Size-QDO-ES",
    "INFER_TIME_QDO": "Infer-QDO-ES",
}


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
    # df = df.drop(columns=["task", "iteration", "weights", "models_used", "dataset", "meta"], errors="ignore")

    print(f"df shape: {df.shape}")
    print(df.columns)
    # print(df.groupby("method").agg("mean")["roc_auc_val"])
    # print(df.groupby("method").agg("mean")["roc_auc_test"])

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


def normalize_per_task(df):
    """Normalize scores per task across all seeds."""
    for task in df["task"].unique():
        task_mask = df["task"] == task

        # Get the data for the whole task across all seeds
        roc_auc_data = df.loc[task_mask, "roc_auc_test"]
        inference_time_data = df.loc[task_mask, "inference_time"]

        # Apply normalization
        df.loc[task_mask, "negated_normalized_roc_auc"] = normalize_data(
            -roc_auc_data
        )  # Negate ROC AUC if needed
        df.loc[task_mask, "normalized_time"] = normalize_data(inference_time_data)

    return df


def boxplot(
    df: pd.DataFrame,
    y_str: str,
    log_y_scale: bool = False,
    log_x_scale: bool = False,
    flip_y_axis: bool = False,
    flip_x_axis: bool = False,
    orient: str = "v",
    rotation_x_ticks: int = 45,
):
    if y_str not in df.columns:
        raise ValueError(f"Column '{y_str}' not found in DataFrame")

    # Plot ROC AUC scores for each method
    plt.figure(figsize=(8, 6))
    if orient == "v":
        sns.boxplot(
            x="method_name",
            y=y_str,
            data=df,
            palette="pastel",
            linewidth=2,
            orient=orient,
        )
    elif orient == "h":
        sns.boxplot(
            x=y_str,
            y="method_name",
            data=df,
            palette="pastel",
            linewidth=2,
            orient=orient,
        )
    else:
        raise ValueError(f"Orient '{orient}' not supported")
    # plt.title("Performance of Ensemble Methods: " + y_str, fontsize=20)

    if orient == "v":
        plt.ylabel(y_str)
        plt.xlabel("Ensemble Method")
    elif orient == "h":
        plt.xlabel(y_str)
        plt.ylabel("Ensemble Method")

    if log_y_scale:
        plt.yscale("log")
    if log_x_scale:
        plt.xscale("log")
    if flip_y_axis:
        plt.gca().invert_yaxis()
    if flip_y_axis:
        plt.gca().invert_yaxis()
    plt.xticks(rotation=rotation_x_ticks)
    plt.grid(True)
    plt.tight_layout()

    # save to file
    plt.savefig("plots/boxplot_" + y_str + ".png", dpi=300, bbox_inches="tight")
    plt.savefig("plots/boxplot_" + y_str + ".pdf", dpi=300, bbox_inches="tight")


def is_pareto_efficient(costs, return_mask=True):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = not np.any(
            np.all(costs <= c, axis=1) & np.any(costs < c, axis=1)
        )
    return is_efficient if return_mask else costs[is_efficient]


def calculate_average_hypervolumes(df, method_name):
    df_method = df[df["method_name"] == method_name]
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
                        df_fold["negated_normalized_roc_auc"].values,
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
    plt.figure(figsize=(8, 6))

    # Use seaborn's boxplot to plot the DataFrame and create an axis object
    ax = sns.boxplot(
        y="Method", x="Hypervolume", data=df, palette="Set2", orient="h"
    )  # You can choose a palette that fits your taste

    # Set titles and labels using the axis object
    # ax.set_title("Comparison of Hypervolume by Method", fontsize=20, fontweight="bold")
    ax.set_ylabel("Method", fontsize=20)
    ax.set_xlabel("Hypervolume", fontsize=20)

    # Set font size for ticks directly on the axes
    ax.tick_params(
        axis="x", labelrotation=45, labelsize=16
    )  # Rotate x-ticks to avoid overlap
    ax.tick_params(axis="y", labelsize=16)

    plt.tight_layout()  # Adjust the plot to fit into the figure area nicely
    plt.savefig(
        "plots/hypervolume_comparison.png", dpi=300
    )  # Save the figure with high resolution
    plt.savefig(
        "plots/hypervolume_comparison.pdf", dpi=300
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

    # Plot with updated font size
    plt.close("all")
    width = 6
    fig, ax = plt.subplots(figsize=(12, width))
    plt.rcParams.update({"font.size": 20})

    plot_stats(result, ax=ax)
    ax.tick_params(axis="both", labelsize=20)  # Set font size for axis ticks
    labels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontsize=20)  # Adjust fontsize as needed
    plt.tight_layout()

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

    return result


def create_latex_table(df, repo, max_char=15):
    if not os.path.exists("tables"):
        os.makedirs("tables")

    methods = df["method_name"].unique()
    task_ids = df["task_id"].unique()

    with open("tables/table.tex", "w") as f:
        f.write("\\begin{longtable}{l" + "c" * len(methods) + "}\n")
        f.write(
            "\\caption{Test ROC AUC - Binary: The mean and standard deviation of the test score over all folds for each method. The best methods per dataset are shown in bold. All methods close to the best method are considered best (using NumPy’s default \\texttt{isclose} function).}\n"
        )
        f.write("\\label{tab:results} \\\\ \n")
        f.write("\\toprule\n")
        f.write("Dataset & " + " & ".join(map(str, methods)) + " \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endfirsthead\n")
        f.write("\\toprule\n")
        f.write("Dataset & " + " & ".join(map(str, methods)) + " \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endhead\n")
        f.write("\\midrule\n")
        f.write(
            "\\multicolumn{"
            + str(len(methods) + 1)
            + "}{r}{Continued on next page} \\\\\n"
        )
        f.write("\\midrule\n")
        f.write("\\endfoot\n")
        f.write("\\bottomrule\n")
        f.write("\\endlastfoot\n")

        for task_id in task_ids:
            dataset_name = repo.tid_to_dataset(
                task_id
            )  # Convert task_id to dataset name
            truncated_name = (
                (dataset_name[:max_char] + "...")
                if len(dataset_name) > max_char
                else dataset_name
            )
            escaped_name = truncated_name.replace("_", "\\_")  # Escape underscores
            line = [str(escaped_name)]  # Ensure the first item is a string
            method_scores = []

            for method in methods:
                method_data = df[
                    (df["task_id"] == task_id) & (df["method_name"] == method)
                ]
                if not method_data.empty:
                    mean_score = method_data["roc_auc_test"].mean()
                    std_dev = method_data["roc_auc_test"].std()
                    score_str = f"{mean_score:.4f}($\\pm${std_dev:.4f})"
                    method_scores.append((mean_score, score_str))
                else:
                    method_scores.append((None, "-"))

            # Determine the best score
            best_score = max(
                score[0] for score in method_scores if score[0] is not None
            )

            for mean_score, score_str in method_scores:
                if mean_score is not None and np.isclose(mean_score, best_score):
                    line.append(f"\\textbf{{{score_str}}}")
                else:
                    line.append(score_str)

            f.write(" & ".join(line) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{longtable}\n")


def plot_task_scatterplots(df: pd.DataFrame, directory="plots/scatter/"):
    """
    Plots and saves separate high-quality scatterplots for each method within each task from the DataFrame,
    enforcing the X and Y axes to be within the range of 0 to 1. Plots are saved in both PNG and PDF formats
    in their respective folders under the base directory.

    Parameters:
    - df (DataFrame): Pandas DataFrame containing the columns 'task', 'method_name', 'negated_normalized_roc_auc',
      'normalized_time', and 'name'.
    - directory (str, optional): Base directory to save the plots to, with subdirectories for each format.
    """
    print("Creating scatter plots...")
    sns.set(
        style="whitegrid", font_scale=1.5
    )  # Increase the font scale for better readability in papers

    # Define directories for file formats
    png_directory = os.path.join(directory, "png")
    pdf_directory = os.path.join(directory, "pdf")

    # Create directories if they do not exist
    for directory in [png_directory, pdf_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Loop through each task
    for task, group in df.groupby("task"):
        # Calculate the minimum count of any method within the task
        min_count = group["method_name"].value_counts().min()

        # Loop through each method within the task
        for method_name, method_group in group.groupby("method_name"):
            sampled_method_group = method_group.sample(
                n=min_count, random_state=42, replace=False
            )

            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=sampled_method_group,
                x="negated_normalized_roc_auc",
                y="normalized_time",
                s=100,  # Reduced size from 150 to 100
                alpha=0.9,  # Added opacity control at 70%
                label=method_name,  # Make sure label is applied
                edgecolor="k",
                linewidth=1,
            )

            # Set axis limits with a slight buffer to make edge points visible
            plt.xlim(-0.05, 1)
            plt.ylim(-0.05, 1)

            # Highlight the zero lines for better visibility
            plt.axhline(0, color="grey", linewidth=2)  # Horizontal zero line
            plt.axvline(0, color="grey", linewidth=2)  # Vertical zero line

            # Adding titles and labels with enhanced font sizes
            plt.title(f"Scatter Plot for Task: {task} - Method: {method_name}")
            plt.xlabel("Negated Normalized ROC AUC")
            plt.ylabel("Normalized Time")

            # Constructing file paths
            png_filename = os.path.join(
                png_directory, f"scatter_plot_{task}_{method_name}.png"
            )
            pdf_filename = os.path.join(
                pdf_directory, f"scatter_plot_{task}_{method_name}.pdf"
            )

            # Save the plot in both formats
            plt.savefig(png_filename, bbox_inches="tight", format="png")
            plt.savefig(pdf_filename, bbox_inches="tight", format="pdf")
            plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    reload = False
    if reload:
        repo = load_repository("D244_F3_C1530_100", cache=True)
        df = parse_dataframes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], repo=repo)
        print(df["method"].unique())
        df.reset_index(drop=True, inplace=True)

        if not os.path.exists("data"):
            os.makedirs("data")
        df.to_json("data/full.json")
        df.to_csv("data/full.csv")
    else:
        print("Loading data. This might take a while...")
        df = pd.read_json("data/full.json")
        # Map method IDs to names
        if "method" in df.columns:
            df["method_name"] = df["method"].map(method_id_name_dict)
        else:
            raise ValueError("Column 'method' not found in DataFrame")
        normalize_per_task(df)
        print(df.head())
        print(df.columns)
        print(df["method"].unique())
        context_name = "D244_F3_C1530_100"
        repo = load_repository(context_name, cache=True)
        # Assume avg_over_seeds DataFrame is available from your existing script
        create_latex_table(df, repo)
        plot_task_scatterplots(df, directory="plots/scatter")

        # Hypervolume
        methods = ["GES", "QO-ES", "QDO-ES", "Size-QDO-ES", "Infer-QDO-ES"]
        all_hypervolumes = {}

        for method in methods:
            all_hypervolumes[method] = calculate_average_hypervolumes(df, method)
        print(all_hypervolumes)

        print("Plotting hypervolumes...")
        plot_hypervolumes(all_hypervolumes)
        exit()
        hypervolumes_df = pd.DataFrame(all_hypervolumes)
        hypervolumes_df.to_csv("data/hypervolumes.csv", index=False)

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
        print(pivot_hypervolumes.head())
        print(pivot_hypervolumes.columns)
        print(pivot_hypervolumes.shape)

        # Now you can use the modified cd_evaluation function
        if not os.path.exists("plots"):
            os.makedirs("plots")
        result = cd_evaluation(
            pivot_hypervolumes,
            maximize_metric=True,
            plt_title="Hypervolume Critical Difference Plot",
            filename="plots/CDPHypervolumes.pdf",
        )
        result = cd_evaluation(
            pivot_hypervolumes,
            maximize_metric=True,
            plt_title="Hypervolume Critical Difference Plot",
            filename="plots/CDPHypervolumes.png",
        )
        print(df.columns)

        # DF with the best solution per task_id, fold, seed and method
        print("Picking best solutions...")
        best_val_scores = df.loc[
            df.groupby(["task_id", "method_name", "fold", "seed"])[
                "roc_auc_val"
            ].idxmax()
        ]
        print("Averaging over folds...")
        avg_over_folds = (
            best_val_scores.groupby(["task_id", "method_name", "seed"])
            .agg(
                {
                    "roc_auc_val": "mean",
                    "roc_auc_test": "mean",
                    "inference_time": "mean",
                }
            )
            .reset_index()
        )
        print("Averaging over seeds...")
        avg_over_seeds = (
            avg_over_folds.groupby(["task_id", "method_name"])
            .agg(
                {
                    "roc_auc_val": "mean",
                    "roc_auc_test": "mean",
                    "inference_time": "mean",
                }
            )
            .reset_index()
        )

        # Plot boxplot for inference time and performance
        print(f"Shape after averaging: {avg_over_seeds.shape}")
        boxplot(
            avg_over_seeds,
            "inference_time",
            log_x_scale=True,
            orient="h",
            rotation_x_ticks=0,
        )

        # Rank data within each task based on 'roc_auc_test' and add as a new column
        avg_over_seeds["rank"] = avg_over_seeds.groupby("task_id")["roc_auc_test"].rank(
            "dense", ascending=False
        )
        boxplot(avg_over_seeds, "rank", flip_y_axis=True)
        avg_over_seeds["negated_roc_auc_test"] = 1 - avg_over_seeds["roc_auc_test"]
        pivot_ranks = avg_over_seeds.pivot(
            index="task_id", columns="method_name", values="negated_roc_auc_test"
        )
        cd_evaluation(
            pivot_ranks,
            maximize_metric=False,
            plt_title="Rankings Critical Difference Plot",
            filename="plots/CDPRankings.pdf",
        )
        cd_evaluation(
            pivot_ranks,
            maximize_metric=False,
            plt_title="Rankings Critical Difference Plot",
            filename="plots/CDPRankings.png",
        )

    # main()
