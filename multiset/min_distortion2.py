import csv
import itertools
import tracemalloc

import torch
import argparse
import numpy as np
import networkx as nx

# ncbi     c = 0.15455
# kegg     c = 0.17356
# pathbank c = 0.089915
def calculate_distortion(G, embedded_nodes, c):
    distortion = torch.tensor(0.0, dtype=torch.float32)
    num_nodes = G.order()
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    count = 0
    for n1, n2 in itertools.combinations(range(0, num_nodes), 2):
        if n1 == n2:
            continue
        try:
            original_distance = lengths[n1][n2]
        except:
            try:
                original_distance = lengths[n2][n1]
            except:
                continue
        embedded_distance = np.linalg.norm(embedded_nodes[n1] - embedded_nodes[n2])
        distortion += torch.abs(c * embedded_distance / original_distance - 1)
        count += 1
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)
    del lengths
    return distortion / count

class Constant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Parameter(torch.tensor(1., dtype=torch.float32))

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, help='Path to input file.')
args = parser.parse_args()
dset = args.file.split('_')[0]

tracemalloc.start(10)
graph_nums, dists, embeds = [], [], []
with open(args.file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, line in enumerate(reader):
        if i == 0:
            continue
        graph_nums.append(line[0])
        dists.append(line[1])
        embeddings = torch.load(f"{dset}_node2vec/{line[0]}.pt", map_location=torch.device("cpu"))
        # Assuming the embeddings are saved as [node1_embedding, node2_embedding, ...]
        embedded_nodes = embeddings.weight.cpu().detach().numpy()
        embeds.append(embedded_nodes)
        del embedded_nodes


def avg_distortion(graphs, nodes, c):
    total_dist = torch.tensor(0., dtype=torch.float32) 
    for i, (G, n) in enumerate(zip(graphs, nodes)):
        print(f"graph num: {i}")
        total_dist += calculate_distortion(G, n, c)
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)
    return total_dist / len(graphs)

if dset != "pathbank":
    graphs = [
        nx.read_edgelist(f"{dset}/{gn}/{gn}.edges", nodetype=int) for gn in graph_nums
    ]
else:
    graphs = [
        nx.read_edgelist(f"data/{gn}/{gn}.edges", nodetype=int) for gn in graph_nums
    ]


model = Constant()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
for i in range(100):
    optimizer.zero_grad()
    loss = avg_distortion(graphs, embeds, model.c)
    loss.backward()
    print(f"Epoch: {i}, Loss: {loss}, C: {model.c.detach().numpy()}")
    optimizer.step()
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)