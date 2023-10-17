import os
import argparse
import itertools
import numpy as np
import networkx as nx
import csv
import json
import torch
import pickle

def calculate_distortion(G, graph_num, embedded_nodes, C, dataset):
    distortion = 0.0
    num_nodes = G.order()
    if not os.path.exists(f"graph_dists/{dataset}/{graph_num}.json"):
        if not os.path.exists(f"graph_dists/{dataset}"):
            os.makedirs(f"graph_dists/{dataset}")
        with open(f"graph_dists/{dataset}/{graph_num}.json", "w") as fp:
            lengths = dict(nx.all_pairs_shortest_path_length(G))
            json.dump(lengths, fp)
    else:
        lengths = json.load(open(f"graph_dists/{dataset}/{graph_num}.json", "r"))
    lengths = {int(k) : {int(k_v) : int(v_v) for k_v, v_v in v.items()} for k, v in lengths.items()}
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
        distortion += np.abs(C * embedded_distance / original_distance - 1)
        count += 1
    return distortion / count

def main(dataset):
    # c_vals = {
    #     "ncbi" : 0.15455,
    #     "kegg" : 0.17356,
    #     "pathbank" : 0.089915,
    #     "humancyc" : 1.0,
    #     "reactome" : 1.0,
    # }
    consts = pickle.load(open(f"{dataset}_models.pkl", "rb"))
    consts = [x.c.item() for x in consts]
    print(len(consts))
    embeddings_folder = f"{dataset}_node2vec"
    if dataset == "pathbank":
        dataset_folder = "data"
    else:
        dataset_folder = f"{dataset}"
    output_csv_file = f"{dataset}_n2v_rescaled_distortions.csv"

    with open(output_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Graph Number", "Distortion"])

        with open(f"{dataset}_n2v_distortions.csv", "r") as infile:
            reader = csv.reader(infile)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                graph_num = line[0]
                filename = f"{graph_num}.pt"
                embedding_file_path = os.path.join(embeddings_folder, filename)
                edges_file_path = os.path.join(dataset_folder, graph_num, f"{graph_num}.edges")

                embeddings = torch.load(embedding_file_path, map_location=torch.device("cpu"))
                graph = nx.read_edgelist(edges_file_path, nodetype=int)

                embedded_nodes = embeddings.weight.cpu().detach().numpy()

                distortion = calculate_distortion(graph, graph_num, embedded_nodes, consts[i-1], dataset)
                csvwriter.writerow([graph_num, distortion])
                print(f"Distortion for graph {graph_num}: {distortion}")

    print(f"Distortions saved to {output_csv_file}")

if __name__ == "__main__":
    from min_distortion_multiple import Constant
    parser = argparse.ArgumentParser(description="Calculate distortion of node embeddings.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    args = parser.parse_args()

    main(args.dataset)