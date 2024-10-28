### Histograms

def plot_evaluation_metrics(df, metric_name):
    plt.figure(figsize=(12, 8))

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    plot_data = df.copy()
    # Ensure the metric is between 0 and 1 for consistent plotting
    plot_data[metric_name] = plot_data[metric_name].clip(0, 1)

    # Generate the plot
    ax = sns.histplot(
        data=plot_data,
        x=metric_name,
        hue="time_weight",
        element="step",
        stat="density",
        common_norm=False,
        palette="viridis",  # Use viridis for color gradient
        line_kws={"linewidth": 2},  # Increase line width for better visibility
        kde=True,
    )

    # Create the colorbar with proper referencing of the axes
    norm = plt.Normalize(plot_data["time_weight"].min(), plot_data["time_weight"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Time Weight")

    plt.title(f"Distribution of {metric_name} Across Different Time Weights")
    plt.xlabel(metric_name)
    plt.ylabel("Density (log scale)")
    plt.grid(True, which="both", ls="--")  # Grid for both major and minor ticks
    plt.show()


def plot_time_weight_impact(df):
    # Filter to include only rows where the method is 'MULTI_GES'
    multi_ges_data = df[df["method"] == "MULTI_GES"]

    # Plot for normalized time
    plot_evaluation_metrics(multi_ges_data, "normalized_time")

    # Plot for performance score, assuming 'negated_normalized_roc_auc' is the performance metric
    plot_evaluation_metrics(multi_ges_data, "negated_normalized_roc_auc")
    
plot_time_weight_impact(df)
# Time weight has impact on resutls


def plot_metrics_histogram(df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Adjusting bins to actual range and not bin index
    x_edges = np.linspace(0, 1, 20)  # normalized_time from 0 to 1
    y_edges = np.linspace(0, 1, 20)  # negated_normalized_roc_auc from 0 to 1

    # Density Plot
    ax1 = axes[0]
    sns.histplot(
        df,
        x="normalized_time",
        y="negated_normalized_roc_auc",
        bins=[x_edges, y_edges],
        cbar=True,
        ax=ax1,
        stat="density",
    )
    ax1.set_title("Density per Bin")
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Negated Normalized ROC AUC")
    ax1.set_ylim([0, 1])  # Correcting y-axis scale

    # Average Time Weight Plot
    ax2 = axes[1]
    df["x_bin"] = pd.cut(
        df["normalized_time"], bins=x_edges, labels=np.round(np.linspace(0, 1, 19), 2)
    )
    df["y_bin"] = pd.cut(
        df["negated_normalized_roc_auc"],
        bins=y_edges,
        labels=np.round(np.linspace(0, 1, 19), 2),
    )

    grouped = df.groupby(["x_bin", "y_bin"])["time_weight"].mean().reset_index()
    grouped["x_bin"] = grouped["x_bin"].astype(float)
    grouped["y_bin"] = grouped["y_bin"].astype(float)

    heatmap_data = grouped.pivot_table(
        index="y_bin", columns="x_bin", values="time_weight"
    )
    sns.heatmap(heatmap_data, ax=ax2, cbar=True, cmap="viridis", annot=True)
    ax2.set_title("Average Time Weight per Bin")
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Negated Normalized ROC AUC")
    ax2.set_ylim([0, 19])  # Correcting y-axis scale for display

    plt.tight_layout()
    plt.show()
    
multi_ges_df = df.loc[df["method"] == "MULTI_GES"]
print(multi_ges_df.shape)
plot_metrics_histogram(multi_ges_df)