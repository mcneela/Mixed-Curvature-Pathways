import os
import networkx as nx

def count_nodes_and_edges(data_dir):
    nodes = []
    edges = []
    for subdir in os.listdir(data_dir):
        with open(os.path.join(data_dir, subdir, f"{subdir}.edges"), 'r') as filein:
            max_int = 0
            for i, line in enumerate(filein):
                n1, n2 = line.split('\t')
                n1, n2 = int(n1), int(n2)
                if n1 > max_int:
                    max_int = n1
                if n2 > max_int:
                    max_int = n2
            edges.append(i + 1)
            nodes.append(max_int + 1)
    return nodes, edges

if __name__ == '__main__':
    nodes, edges = count_nodes_and_edges('data')