import csv
import time
import pandas as pd
import networkx as nx

import numpy as np
from tqdm import tqdm

df = pd.read_csv('second_all_data.tsv', sep='\t')

def hyperbolicity_sample(G, num_samples=50000):
    curr_time = time.time()
    hyps = []
    for i in tqdm(range(num_samples)):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    print('Time for hyp: ', time.time() - curr_time)
    return max(hyps)

ranges = {}
best_dists = {}
best_dist_counts = {}
best_spaces = {}
best_space_counts = {}
with open('unique_best_spaces_3.tsv', 'w', newline='') as fpath:
    writer = csv.writer(fpath, delimiter='\t')
    writer.writerow(['Graph num', 'Num Nodes', 'Num Edges', 'Density', 'Number of Self Loops', 'Node Connectivity', 'Avg Clustering Coeff', 'Diameter', 'Degree Assortativity', 'Is Bipartite', 'H dim', 'H copies', 'E dim', 'E copies', 'S dim', 'S copies', 'Num Triangles', 'Dist Range', 'Dist Diff from Mean', 'Dist Diff from Next Best', 'Dist Diff from Euclidean', '% Dist Diff From Euclidean', 'Best Euclidean Distortion', 'Best Overall Distortion'])
    for i, graph_num in enumerate(df['Graph num'].unique()):
        graph = df[df['Graph num'] == graph_num]
        ranges[graph_num] = graph['Best dist'].max() - graph['Best dist'].min()
        dist_range = graph['Best dist'].max() - graph['Best dist'].min()
        dist_diff_from_mean = graph['Best dist'].min() - graph['Best dist'].mean()
        dist_diff_from_next_best = sorted(graph['Best dist'])[0] - sorted(graph['Best dist'])[1]
        euclids = graph[(graph['E copies'] > 0) & (graph['H copies'] == 0) & (graph['S copies'] == 0)]['Best dist']
        best_euclidean_dist = euclids.min()
        best_overall_dist = graph['Best dist'].min()
        dist_diff_from_euclidean = euclids.min() - graph['Best dist'].min()
        percent_dist_diff_from_euclidean = dist_diff_from_euclidean / euclids.min() * 100
        best_dists[graph_num] = graph['Best dist'].min()
        best_dist_counts[graph_num] = len(graph[graph['Best dist'] == graph['Best dist'].min()])
        best_space = graph[graph['Best dist'] == graph['Best dist'].min()]
        best_space_counts[graph_num] = len(best_space)
        best_spaces[graph_num] = [(best_space.iloc[0]['H dim'], best_space.iloc[0]['H copies']), (best_space.iloc[0]['E dim'], best_space.iloc[0]['E copies']), (best_space.iloc[0]['S dim'], best_space.iloc[0]['S copies'])]
        graph_file = open(f'../data/{graph_num}/{graph_num}.edges', 'r')
        g = nx.read_edgelist(graph_file)
        from networkx.algorithms import approximation as approx
        # try:
        #     d_h = hyperbolicity_sample(g)
        # except:
        #     d_h = 'N/A'
        #     continue
        triangles = sum(list(nx.triangles(g).values())) // 3
        density = nx.density(g)
        self_loops = nx.number_of_selfloops(g)
        node_connectivity = nx.node_connectivity(g)
        avg_clustering_coeff = nx.average_clustering(g)
        try:
            diameter = nx.diameter(g)
        except:
            diameter = np.inf
        degree_assortativity = nx.degree_pearson_correlation_coefficient(g)
        is_bipartite = int(nx.is_bipartite(g))
        if best_space_counts[graph_num] == 1:
            writer.writerow([
                graph_num,
                int(best_space.iloc[0]['num nodes']),
                int(best_space.iloc[0]['num edges']),
                density,
                self_loops,
                node_connectivity,
                avg_clustering_coeff,
                diameter,
                degree_assortativity,
                is_bipartite,
                best_space.iloc[0]['H dim'], 
                int(best_space.iloc[0]['H copies']), 
                best_space.iloc[0]['E dim'], 
                int(best_space.iloc[0]['E copies']), 
                best_space.iloc[0]['S dim'], 
                int(best_space.iloc[0]['S copies']),
                triangles,
                # d_h
                dist_range,
                dist_diff_from_mean,
                dist_diff_from_next_best,
                dist_diff_from_euclidean,
                percent_dist_diff_from_euclidean,
                best_euclidean_dist,
                best_overall_dist
            ])

# 167 graphs with single best space
unique_best_spaces = {}
for graph_num in best_spaces:
    if best_space_counts[graph_num] == 1:
        unique_best_spaces[graph_num] = best_spaces[graph_num]
        print(f"{graph_num}\tH:{best_spaces[graph_num][0]}\tE:{best_spaces[graph_num][1]}\tS:{best_spaces[graph_num][2]}")

# count delta over pure euclidean
# avg degree
# seq over degrees
# predict edge labels having to do with where interaction takes place in cell

with open('general_stats.txt', 'w') as fpath:
    fpath.write(f"Number of graphs: {len(df['Graph num'].unique())}\n")
    fpath.write(f"Number of graphs with single best space: {len(unique_best_spaces)}\n")
    fpath.write(f"Number of graphs with multiple best spaces: {len(best_spaces) - len(unique_best_spaces)}\n")
    fpath.write(f"Average number of nodes: {df['num nodes'].mean()}\n")
    fpath.write(f"Average number of edges: {df['num edges'].mean()}\n")
    fpath.write(f"Median number of nodes: {df['num nodes'].median()}\n")
    fpath.write(f"Median number of edges: {df['num edges'].median()}\n")

