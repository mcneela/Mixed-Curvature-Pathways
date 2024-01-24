import os

base_dir = 'data'

# Iterate over each subdirectory in the base directory
for graph_num in os.listdir(base_dir):
    graph_dir = os.path.join(base_dir, graph_num)

    # Check if the item is a directory
    if os.path.isdir(graph_dir):
        input_file = os.path.join(graph_dir, f'{graph_num}_string.edges')
        output_file = os.path.join(graph_dir, f'{graph_num}_string_cleaned.edges')

        # Read the tab-separated file, remove the third column, and write to a new file
        if os.path.exists(input_file):
            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                for line in infile:
                    parts = line.split(' ')
                    # Remove the third column
                    del parts[2]
                    # Write the remaining columns to the output file
                    outfile.write('\t'.join(parts)+'\n')
