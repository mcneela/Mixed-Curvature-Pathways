import os
import csv

all_proteins = open('pathbank_all_proteins.csv', 'r')
reader = csv.reader(all_proteins, delimiter=',')
gene_prot_map = {}
for line in reader:
    uniprot, genbank_id, gene_name, locus = line[4], line[8], line[9], line[10]
    gene_prot_map[genbank_id.upper()] = uniprot
    gene_prot_map[gene_name.upper()] = uniprot
    gene_prot_map[locus.upper()] = uniprot
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
for line in link_file:
    p1, p2, score = line.strip().split(' ')
    validated_links[(p1, p2)] = score
link_file.close()

global_edge_file = open('pathbank_string_edges.txt', 'w')
error_count = 0
for dir_name in os.listdir('../data'):
    fname = os.path.join('../data', dir_name, f'{dir_name}.txt')
    orig_edges = open('../data', dir_name, f'{dir_name}.edges', 'r')
    local_edge_file = open('../data', dir_name, f'{dir_name}_string.txt', 'w')
    all_pathway_nodes = set()
    with open(fname, 'r') as filein:
        for line in filein:
            edge = line.strip().split('\t')
            prot1, etype, prot2 = edge
            if prot1 in aliases:
                p1a = aliases[prot1]
            elif prot1 in gene_prot_map:
                try:
                    p1a = aliases[gene_prot_map[prot1]]
                except:
                    continue
            else:
                error_count += 1
                continue
            all_pathway_nodes.add(p1a)

            if prot2 in aliases:
                p2a = aliases[prot2]
            elif prot2 in gene_prot_map:
                try:
                    p2a = aliases[gene_prot_map[prot2]]
                except:
                    continue
            else:  
                error_count += 1
                continue
            all_pathway_nodes.add(p2a)
        

            # if (p1a, p2a) in validated_links:
            #     global_edge_file.write(f"{p1a}\t{p2a}\n")
            # elif (p2a, p1a) in validated_links:
            #     global_edge_file.write(f"{p2a}\t{p1a}\n")
            # else:
            #     error_count += 1
            #     continue
global_edge_file.close()



