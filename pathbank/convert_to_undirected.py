import os
import networkx as nx

data_dir = '../data/'

# Iterate through the directory structure
for dirpath, dirnames, filenames in os.walk(data_dir):
    for filename in filenames:
        # Check if the file is a _named.edges file
        if filename.endswith('_named.edges'):
            # Parse the filename to get the number
            num = filename.split('_')[0]

            # Load the edge list into a NetworkX graph
            G = nx.read_edgelist(os.path.join(dirpath, filename))

            # Convert the graph to undirected
            G_undirected = G.to_undirected()

            # Write the undirected edge list to a new file
            nx.write_edgelist(G_undirected, os.path.join(dirpath, f'{num}_undirected.edges'), delimiter='\t')
