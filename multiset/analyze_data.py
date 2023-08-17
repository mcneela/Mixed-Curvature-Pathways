from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('all_data.tsv', sep='\t')

# get the unique graph ids in this dataset
unique_graphs = df['Graph num'].unique()
for graph_id in unique_graphs:
    group = df.iloc[df.index[df['Graph num'] == graph_id]]
    dists = group['Best dist']
    plt.hist(dists, bins=10, alpha=0.5, label=graph_id)
plt.legend(loc='upper right')
plt.savefig(f'analyses/all_data_histogram.png')

plt.figure()

nodes, edges = [], []
for graph_id in unique_graphs:
    group = df.iloc[df.index[df['Graph num'] == graph_id]]
    node = group['num nodes'].unique()[0]
    edge = group['num edges'].unique()[0]
    nodes.append(node)
    edges.append(edge)

plt.scatter(nodes, edges, label=unique_graphs)
plt.xlabel('num nodes')
plt.ylabel('num edges')
plt.legend()
# plt.show()

space_dict = defaultdict(list)
for idx, row in df.iterrows():
    key = (
        row['H dim'], row['H copies'],
        row['S dim'], row['S copies'],
        row['E dim'], row['E copies'],
    )
    space_dict[key].append(row['Best dist'])

for i, k in enumerate(space_dict):
    v = space_dict[k]
    title = f"H dim: {k[0]}, H copies: {k[1]}, S dim: {k[2]}, S copies: {k[3]}, E dim: {k[4]}, E copies: {k[5]}"
    plt.figure()
    plt.title(title)
    plt.hist(v, bins=10, alpha=0.5, label=k)
    plt.legend(loc='upper right')
    plt.savefig(f'analyses/space_combo_histogram_{i}.png') 
