import os
import pandas as pd

base_dir = '../data'

# Iterate over each subdirectory in the base directory
for graph_num in os.listdir(base_dir):
    graph_dir = os.path.join(base_dir, graph_num)
    
    # Check if the item is a directory
    if os.path.isdir(graph_dir):
        file_path = os.path.join(graph_dir, f'{graph_num}.txt')
        output_path = os.path.join(graph_dir, f'{graph_num}_named.edges')

        # Read the tab-separated file, remove the second column, and write to a new file
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep='\t', header=None)
            df.drop(columns=1, inplace=True)  # Drop the second column (columns are 0-indexed)
            df.to_csv(output_path, sep='\t', index=False, header=False)
