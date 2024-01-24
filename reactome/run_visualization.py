import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import approximation as approx

G = nx.read_edgelist("../data/95/95_string.edges")

labelList = list(G.nodes())
print(labelList)
# positions = nx.spring_layout(G, k=1, iterations=20)
positions = nx.kamada_kawai_layout(G, scale=5.0)
# nx.draw(
#     G,
#     pos=pos,
#     with_labels=True,
#     edge_color=["red" if i % 4 == 0 else "blue" for i in range(len(G.edges()))],
# )
print(positions)
print({idx: val for idx, val in enumerate(labelList)})
nodes = nx.draw_networkx_nodes(G, pos=positions, node_color="white", node_size=[len(labelList[i])**2 * 60 for i in range(len(positions))])
nodes.set_edgecolor('black')
colors = []
for i, e in enumerate(G.edges()):
    if i % 4 == 0 and G.degree(e[0]) > 1 and G.degree(e[1]) > 1:
        # G.edges[e]['color'] = 'red'
        colors.append('red')
    else:
        colors.append('blue')
        # G.edges[e]['color'] = 'blue'
nx.draw_networkx_edges(G, pos=positions, edge_color=colors) #edge_color=["red" if i % 4 == 0 else "blue" for i in range(len(G.edges()))])
nx.draw_networkx_labels(G, pos=positions, labels = {x: x for x in labelList})
# nx.draw_networkx(
#     G, 
#     # node_color =['C{}'.format(i) for i in positions], 
#     node_color="white",
#     edge_color=["red" if i % 4 == 0 else "blue" for i in range(len(G.edges()))],
#     pos=positions, 
#     labels={x: x for x in labelList},
#     node_size=[len(labelList[i])**2 * 60 for i in range(len(positions))]
#     )
plt.show()