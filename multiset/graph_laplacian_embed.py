import os
import numpy as np
import networkx as nx

from argparse import ArgumentParser

def embed_graph(fpath):
    G = nx.read_edgelist(fpath)
    L = nx.laplacian_matrix(G).todense()
    evals, M = np.linalg.eig(L)
    return M

def save_embed(outpath, M):
    np.save(outpath, M)

def embed_all(folder):
    for graph_num in os.listdir(folder):
        edge_path = os.path.join(folder, graph_num, f"{graph_num}.edges")
        M = embed_graph(edge_path)
        outpath = os.path.join(f"{folder}_laplacian_models", f"{graph_num}.npy")
        save_embed(outpath, M)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="name of dataset")
    args = parser.parse_args()
    dataset = args.dataset
    embed_all(dataset)
