# TODO: add scatterplots for different metrics
#       negated roc auc is always good
#       but the other could be inference time, ensemble size, disk space used, memory usage
#       ? Do we want to look at ensemble size & memory used ?

# def plot_task_data(df:pd.DataFrame, directory: str = 'plots/scatter', x_column: str = 'normalized_time', y_column: str = 'negated_normalized_roc_auc'):
#     # Create a directory to save the plots if it doesn't exist
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     
#     # Define the unique tasks to create separate plots
#     tasks = df['task'].unique()
#     
#     # Set the overall aesthetics for the plots
#     sns.set(style="whitegrid")
#     
#     # Iterate over each task to create and save a plot
#     for task in tasks:
#         plt.figure(figsize=(10, 8))  # Define a new figure for each task
#         
#         # Filter the dataframe by task
#         task_data = df[df['task'] == task]
#         
#         # Scatter plot using seaborn to easily differentiate between methods
#         scatter = sns.scatterplot(
#             data=task_data, 
#             x=x_column, 
#             y=y_column, 
#             hue='method',  # Differentiate by color
#             size='models_used_length', 
#             sizes=(20, 200),  # Control the range of sizes of markers
#             legend='brief',  # Simplified legend
#             palette='tab10'  # Use a color palette that is visually appealing
#         )
#         scatter.set_title(f'Task: {task}')
#         scatter.set_xlabel(x_column)
#         scatter.set_ylabel('Negated Normalized ROC AUC')
#         
#         # Simplify the legend: show only methods in legend
#         handles, labels = scatter.get_legend_handles_labels()
#         # Only keep the legend handles that correspond to 'method'
#         method_handles = handles[1:len(df['method'].unique())+1]
#         method_labels = labels[1:len(df['method'].unique())+1]
#         scatter.legend(method_handles, method_labels, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
#         
#         # Save the plot to the specified directory
#         plt.savefig(f"{directory}/Task_{task}.png", bbox_inches='tight')
#         plt.close()

# Always include GES and QDO-ES as baseline
# Inference time: Size-QDO-ES, Infer-QDO-ES
# Memory: Memory-QDO-ES
# Disk space usage: Diskspace-QDO-ES

# df_filtered = df[df['method_name'].isin(['GES', 'QDO-ES', 'Size-QDO-ES', 'Infer-QDO-ES'])]
# plot_task_data(df_filtered, directory="../plots/scatter/inference_time")
# df_filtered = df[df['method_name'].isin(['GES', 'QDO-ES', 'Memory-QDO-ES'])]
# plot_task_data(df_filtered, directory="../plots/scatter/memory", x_column='normalized_memory')
# df_filtered = df[df['method_name'].isin(['GES', 'QDO-ES', 'Memory-QDO-ES'])]
# plot_task_data(df_filtered, directory="../plots/scatter/disk_space", x_column='normalized_diskspace')
# print("Done")


# import matplotlib.image as mpimg
# import random
# 
# 
# # Get a list of all files in the directory and filter for PNG files
# dir_path = "../plots/scatter/"
# all_files = [
#     os.path.join("../plots/scatter/", file)
#     for file in os.listdir(dir_path)
#     if file.endswith(".png")
# ]
# 
# # Randomly pick four images
# selected_images = random.sample(all_files, 4)
# 
# # Set up the figure and axes
# fig, axs = plt.subplots(2, 2, figsize=(12, 7))
# axs = axs.flatten()
# for ax, img_path in zip(axs, selected_images):
#     img = mpimg.imread(img_path)
#     ax.imshow(img)
#     ax.axis("off")
# plt.tight_layout()
# plt.show()