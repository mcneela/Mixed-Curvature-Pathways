import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os

# List of dataset names
dataset_names = ["nci", "kegg", "humancyc", "pathbank", "reactome"]

# Initialize a figure
plt.figure(figsize=(12, 6))

# Loop through each dataset
for dataset_name in dataset_names:
    data_dir = f"{dataset_name}/data"
    
    # Check if the 'data' directory exists for the current dataset
    if os.path.exists(data_dir):
        edge_counts = []

        # Loop through graph numbers (subdirectories)
        for graph_num_dir in os.listdir(data_dir):
            graph_num = int(graph_num_dir)
            
            # Define the path to the edge list file
            edge_list_path = os.path.join(data_dir, graph_num_dir, f"{graph_num}.edges")

            # Check if the file exists
            if os.path.exists(edge_list_path):
                # Read the edge list and create a graph
                G = nx.read_edgelist(edge_list_path)

                # Append the number of edges to the list
                edge_counts.append(len(G.edges()))

        # Plot the histogram for the current dataset
        sns.histplot(edge_counts, label=dataset_name, alpha=0.8, kde=False)

# Set plot labels and title
plt.xlim(0, 7500)
plt.xlabel("Number of Edges")
plt.ylabel("Frequency")
plt.title("Edge Distribution Histograms for Multiple Datasets")

# Show a legend
plt.legend()

# Show the plot
plt.show()
