import networkx as nx
import matplotlib.pyplot as plt

filein = open('best_configs.tsv', 'r')

for i, line in enumerate(filein):
    if i == 0:
        continue
    graph = line.split('\t')[0]
    print(graph)
    el_path = f"data/{graph}/{graph}.edges"
    print(el_path)
    G = nx.read_edgelist(el_path)
    plt.figure()
    nx.draw(G)
    plt.savefig(f"best_plots/{graph}.png")
    plt.clf()
    