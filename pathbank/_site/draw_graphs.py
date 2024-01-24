import networkx as nx
import matplotlib.pyplot as plt

filein = open('unique_best_spaces.tsv', 'r')

for i, line in enumerate(filein):
    if i == 0:
        continue
    graph = line.split('\t')[0]
    print(graph)
    el_path = f"../data/{graph}/{graph}.edges"
    print(el_path)
    G = nx.read_edgelist(el_path)
    plt.figure()
    nx.draw(G)
    plt.savefig(f"graph_imgs/{graph}.png")
    plt.clf()
    