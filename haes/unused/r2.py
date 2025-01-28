# from scipy.spatial import distance_matrix

# def calculate_r_squared_scores(df, pareto_fronts, methods, objectives):
#     """
#     Calculate R-squared scores per task_id, seed, fold, and method_name.

#     Parameters:
#     - df: pandas DataFrame containing the data.
#     - pareto_fronts: Nested dictionary with Pareto fronts per task_id, seed, and fold.
#     - methods: List of method names to compare.
#     - objectives: List of two objective column names.

#     Returns:
#     - A DataFrame containing R-squared scores with columns: ['task_id', 'seed', 'fold', 'method_name', 'r_squared'].
#     """
#     r_squared_list = []

#     # Iterate over task_ids
#     for task_id in pareto_fronts:
#         for seed in pareto_fronts[task_id]:
#             for fold in pareto_fronts[task_id][seed]:
#                 # Get the true Pareto front for this combination
#                 pareto_front_df = pareto_fronts[task_id][seed][fold]
#                 true_pareto_points = pareto_front_df[objectives].values

#                 # Compute TSS using the true Pareto front
#                 pareto_mean = np.mean(true_pareto_points, axis=0)
#                 tss = np.sum(np.sum((true_pareto_points - pareto_mean) ** 2, axis=1))

#                 # Skip if TSS is zero (degenerate case)
#                 if tss == 0:
#                     continue

#                 # Iterate over methods
#                 for method_name in methods:
#                     # Filter the DataFrame for the current combination and method
#                     df_method = df[
#                         (df['task_id'] == task_id) &
#                         (df['seed'] == seed) &
#                         (df['fold'] == fold) &
#                         (df['method_name'] == method_name)
#                     ]

#                     # Skip if there are no data points for this method
#                     if df_method.empty:
#                         continue

#                     method_points = df_method[objectives].values

#                     # Compute RSS: Sum of squared distances from method points to the true Pareto front
#                     # For each method point, find the closest point on the true Pareto front
#                     distances = distance_matrix(method_points, true_pareto_points)
#                     min_distances = np.min(distances, axis=1)
#                     rss = np.sum(min_distances ** 2)

#                     # Calculate R-squared
#                     r_squared = 1 - (rss / tss)

#                     # Store the result
#                     r_squared_list.append({
#                         'task_id': task_id,
#                         'seed': seed,
#                         'fold': fold,
#                         'method_name': method_name,
#                         'r_squared': r_squared
#                     })

#     # Convert the list to a DataFrame
#     r_squared_df = pd.DataFrame(r_squared_list)
#     return r_squared_df

# def create_r_squared_boxplots(r_squared_df, methods):
#     """
#     Create boxplots of R-squared scores to compare methods.

#     Parameters:
#     - r_squared_df: DataFrame containing R-squared scores.
#     - methods: List of method names to include in the boxplot.
#     """
#     # Filter the DataFrame for the specified methods
#     r_squared_df = r_squared_df[r_squared_df['method_name'].isin(methods)]

#     # Create a boxplot
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x='method_name', y='r_squared', data=r_squared_df, order=methods, showfliers=False)
#     plt.xlabel('Method')
#     plt.ylabel('R-squared Score')
#     plt.title('Comparison of Methods based on R-squared Scores')
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     plt.tight_layout()

#     # Save the plot
#     save_dir = '../plots/boxplots'
#     os.makedirs(save_dir, exist_ok=True)
#     filepath = os.path.join(save_dir, 'r_squared_comparison.png')
#     plt.savefig(filepath, dpi=300)
#     plt.close()
# Assume df is your DataFrame with all data
# Assume pareto_fronts is the nested dictionary with Pareto fronts
# Assume methods is a list of method names to compare
# Define objectives
# objectives = ['negated_normalized_roc_auc', 'normalized_time']

# Step 1: Calculate R-squared scores
# r_squared_df = calculate_r_squared_scores(df, pareto_fronts, methods, objectives)

# Step 2: Create boxplots to compare methods
# create_r_squared_boxplots(r_squared_df, methods)