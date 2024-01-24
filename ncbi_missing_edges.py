import os
import json

# Replace 'your_json_file.json' with the actual JSON file path
json_file_path = 'ncbi_prots.json'
edge_list_directory = 'ncbi/'

# Load the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Initialize a set to store unique EDGE1 and EDGE2 values
unique_edges = set()

# Iterate over all files in the edge list directory
for root, dirs, files in os.walk(edge_list_directory):
    for file in files:
        if file.endswith('.txt') and file != 'general_stats.txt':
            edge_file_path = os.path.join(root, file)
            with open(edge_file_path, 'r') as edge_file:
                for line in edge_file:
                    edge1, _, edge2 = line.strip().split('\t')
                    unique_edges.add(edge1)
                    unique_edges.add(edge2)

# Create a list of edges that are not present in the JSON file
missing_edges = [edge for edge in unique_edges if edge not in data]

# Write the missing edges to a .txt file
output_file_path = 'ncbi_missing_edges.txt'
with open(output_file_path, 'w') as output_file:
    for edge in missing_edges:
        output_file.write(edge + '\n')

print(f"Missing edges have been written to {output_file_path}")
