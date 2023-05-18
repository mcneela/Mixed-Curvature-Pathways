import os

data_dir = '../data/'

# Iterate through the directory structure
for dirpath, dirnames, filenames in os.walk(data_dir):
    for filename in filenames:
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            # Parse the filename to get the number
            num = filename.split('.')[0]

            # Open the file
            with open(os.path.join(dirpath, f'{num}_string.edges'), 'r') as f:
                # Read the file contents and parse the lines
                lines = f.readlines()
                edges = []
                for line in lines:
                    # Split the line by tabs and remove the third column
                    cols = line.strip().split(' ')
                    cols.pop(2)

                    # Append the remaining columns to the edges list
                    edges.append('\t'.join(cols))

            # Write the edges list to a new file
            with open(os.path.join(dirpath, f'{num}_string_cleaned.edges'), 'w') as f:
                f.write('\n'.join(edges))
