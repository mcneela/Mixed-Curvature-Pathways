import numpy as np
import networkx as nx
from sklearn.metrics import average_precision_score

# Assuming 'embeddings' is your numpy array of embeddings
# Assuming 'graph' is your NetworkX graph

def calculate_mean_avg_precision(embeddings, graph, k=10):
    mean_avg_precision = 0

    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 0:
            continue

        node_embedding = embeddings[node]
        neighbor_embeddings = embeddings[neighbors]

        # Calculate pairwise cosine similarities
        similarities = np.dot(node_embedding, neighbor_embeddings.T) / (
            np.linalg.norm(node_embedding) * np.linalg.norm(neighbor_embeddings, axis=1)
        )

        # Rank neighbors by similarity
        ranked_indices = np.argsort(similarities)[::-1]

        # Keep only the top-k ranked neighbors
        ranked_indices = ranked_indices[:k]

        # Create binary relevance labels (1 for true neighbors, 0 for non-neighbors)
        labels = np.zeros(len(neighbors))
        labels[np.isin(neighbors, ranked_indices)] = 1

        # Calculate average precision for the node
        avg_precision = average_precision_score(labels, similarities)
        mean_avg_precision += avg_precision

    mean_avg_precision /= len(graph.nodes())
    return mean_avg_precision

# Call the function with your embeddings and graph
mean_avg_precision = calculate_mean_avg_precision(embeddings, graph)
print("Mean Average Precision:", mean_avg_precision)