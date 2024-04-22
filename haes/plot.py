import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tabrepo import load_repository, EvaluationRepository


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
    plt.savefig("performance_" + y_str + ".png", dpi=300)


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
    plt.savefig("performance_" + "dataset_rank" + ".png", dpi=300)


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
    df_merged["roc_auc_normalized"] = df_merged.groupby("task_id")["roc_auc"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    return df_merged


def get_inference_time(entry, repo: EvaluationRepository) -> pd.DataFrame:
    # Get the inference time for each task and method
    # The inference time of the ensemble is made up of the inference time of the unique base models
    models_used = entry["models_used"]
    fold = int(entry["task"].split("_")[1])

    dataset = repo.task_to_dataset(entry["task"])
    metrics = repo.metrics(datasets=[dataset], configs=models_used)

    metrics_fold = metrics.xs(fold, level="fold")
    total_inference_time = sum(metrics_fold["time_infer_s"])
    return total_inference_time


def is_pareto_efficient(costs, return_mask=True):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        # None of the other points should have both lower inference time and higher ROC AUC.
        is_efficient[i] = np.all(np.any(costs[:i] >= c, axis=1)) and np.all(
            np.any(costs[i + 1 :] >= c, axis=1)
        )
    return is_efficient if return_mask else costs[is_efficient]


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

    print(df_merged.head(12))

    boxplot(df_merged, "roc_auc_normalized")
    boxplot(df, "normalized_improvement")
    df_merged_infertime_no_outliers = df_merged[
        (df_merged["inference_time"] < 1.5) & (df_merged["roc_auc_normalized"] > 0.6)
    ]
    boxplot(df_merged_infertime_no_outliers, "inference_time")
    boxplot_ranking(df_merged)

    # Pareto efficiency
    # Exclude single best
    df = df[df["method"] != "Single Best"]
    costs = np.stack((df["inference_time"], -df["roc_auc"]), axis=1)
    pareto_mask = is_pareto_efficient(costs)
    pareto_df = df[pareto_mask]

    # Get the pastel color palette with as many colors as there are methods
    methods = df['method'].unique()
    palette = sns.color_palette("pastel", len(methods))

    # Create a color dictionary to map each method to a color
    color_dict = dict(zip(methods, palette))

    # Plot all configurations with colors based on the method
    plt.figure(figsize=(10, 6))
    for method in methods:
        subset = df[df['method'] == method]
        plt.scatter(subset['inference_time'], subset['roc_auc'], color=color_dict[method], label=method)

    # Now plot the Pareto front with a distinct color and label
    plt.scatter(
        pareto_df['inference_time'], pareto_df['roc_auc'],
        color='black', edgecolors='k', label='Pareto Front', zorder=10
    )

    # Enhance plot aesthetics and usability
    plt.title("Pareto Front of Inference Time vs ROC AUC")
    plt.xlabel("Inference Time (s)")
    plt.xscale("log")
    plt.ylabel("ROC AUC")
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to accommodate the legend
    plt.savefig("pareto_front.png", dpi=300)


if __name__ == "__main__":
    main()
