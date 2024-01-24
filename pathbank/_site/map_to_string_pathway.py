import os
import csv
import itertools
import networkx as nx

all_proteins = open('pathbank_all_proteins.csv', 'r')
reader = csv.reader(all_proteins, delimiter=',')
gene_prot_map = {}
prot_gene_map = {}
for line in reader:
    uniprot, genbank_id, gene_name, locus = line[4], line[8], line[9], line[10]
    gene_prot_map[genbank_id.upper()] = uniprot
    gene_prot_map[gene_name.upper()] = uniprot
    gene_prot_map[locus.upper()] = uniprot
    prot_gene_map[uniprot] = genbank_id.upper()
all_proteins.close()


aliases = {}
alias_file = open('9606.protein.aliases.v11.5.txt', 'r')
for line in alias_file:
    str_name, alias, source = line.strip().split('\t')
    if alias not in aliases:
        aliases[alias] = str_name
alias_file.close()

validated_links = {}
link_file = open('9606.protein.physical.links.v11.5.txt', 'r')
for i, line in enumerate(link_file):
    if i == 0:
        continue
    p1, p2, score = line.strip().split(' ')
    if int(score) < 500:
        continue
    validated_links[(p1, p2)] = score
link_file.close()

error_count = 0
total_vs_missing = {}
for i, dir_name in enumerate(os.listdir('../data')):
    print(dir_name)
    fname = os.path.join('../data', dir_name, f'{dir_name}_undirected.edges')
    missing_edges = 0
    new_edges = []
    all_pathway_aliases = []
    actual_edges = set()
    real_edges = set()
    with open(fname, 'r') as filein:
        for j, line in enumerate(filein):
            edge = line.strip().split('\t')
            # prot1, etype, prot2 = edge
            prot1, prot2, _ = edge
            if prot1 in aliases:
                p1a = aliases[prot1]
            elif prot1 in gene_prot_map:
                try:
                    p1a = aliases[gene_prot_map[prot1]]
                except:
                    actual_edges.add((prot1, prot2))
                    continue
            else:
                error_count += 1
                actual_edges.add((prot1, prot2))
                continue

            if prot2 in aliases:
                p2a = aliases[prot2]
            elif prot2 in gene_prot_map:
                try:
                    p2a = aliases[gene_prot_map[prot2]]
                except:
                    actual_edges.add((prot1, prot2))
                    continue
            else:  
                error_count += 1
                actual_edges.add((prot1, prot2))
                continue
            all_pathway_aliases.append((prot1, p1a))
            all_pathway_aliases.append((prot2, p2a))
            # actual_edges.add((prot1, prot2))
            # real_edges.add((p1a, p2a))
            # real_edges.add((p2a, p1a))
            actual_edges.add((prot1, prot2))
            # actual_edges.add((prot2, p2a, prot1, p1a))
        original_size = len(actual_edges)
        for ((prot1, p1a), (prot2, p2a)) in itertools.combinations(all_pathway_aliases, 2):
            if (p1a, p2a) in validated_links and (p1a, p2a) not in actual_edges:
                missing_edges += 1
                new_edges += [(prot1, prot2)]
            elif (p2a, p1a) in validated_links and (p2a, p1a) not in actual_edges:
                missing_edges += 1
                new_edges += [(prot2, prot1)]
    new_edges += list(actual_edges)
    new_size = len(new_edges)
    G = nx.from_edgelist(new_edges)
    G = G.to_undirected()
    print(original_size, G.size())
    # G = nx.convert_node_labels_to_integers(G)
    nx.write_edgelist(G, f'../data/{dir_name}/{dir_name}_string.edges')
    total_vs_missing[i] = (j, missing_edges)

ratios = []
for tot, missing in total_vs_missing.values():
    if tot == 0:
        continue
    ratios.append(missing/tot)


