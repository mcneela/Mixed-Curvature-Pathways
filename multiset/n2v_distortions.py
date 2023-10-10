import os
import argparse
import itertools
import numpy as np
import networkx as nx
import csv

def calculate_distortion(G, embedded_nodes, C):
    distortion = 0.0
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
        distortion += C * embedded_distance / original_distance - 1
        count += 1
    return distortion / count

def main(dataset):
    c_vals = {
        "ncbi" : 0.15455,
        "kegg" : 0.17356,
        "pathbank" : 0.089915,
        "humancyc" : 1.0,
        "reactome" : 1.0,
    }
    embeddings_folder = f"{dataset}_node2vec"
    dataset_folder = f"{dataset}"
    output_csv_file = f"{dataset}_n2v_rescaled_distortions.csv"

    with open(output_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Graph Number", "Distortion"])

        for filename in os.listdir(embeddings_folder):
            if filename.endswith(".npy"):
                graph_num = filename.split('.')[0]
                embedding_file_path = os.path.join(embeddings_folder, filename)
                edges_file_path = os.path.join(dataset_folder, graph_num, f"{graph_num}.edges")

                embeddings = np.load(embedding_file_path)
                graph = nx.read_edgelist(edges_file_path, nodetype=int)

                embedded_nodes = embeddings.weight.cpu().detach().numpy()

                distortion = calculate_distortion(graph, embedded_nodes, c_vals[dataset])
                csvwriter.writerow([graph_num, distortion])
                print(f"Distortion for graph {graph_num}: {distortion}")

    print(f"Distortions saved to {output_csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate distortion of node embeddings.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    args = parser.parse_args()

    main(args.dataset)