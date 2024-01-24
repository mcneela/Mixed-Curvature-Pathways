import os
import csv
import json

gene_prot_map = json.load(open('humancyc_prots.json', 'r'))

aliases = {}
alias_file = open('../kegg/9606.protein.aliases.v11.5.txt', 'r')
for line in alias_file:
    str_name, alias, source = line.strip().split('\t')
    if alias not in aliases:
        aliases[alias] = str_name
alias_file.close()

validated_links = {}
link_file = open('../kegg/9606.protein.physical.links.v11.5.txt', 'r')
for line in link_file:
    p1, p2, score = line.strip().split(' ')
    validated_links[(p1, p2)] = score
link_file.close()

global_edge_file = open('humancyc_string_edges.txt', 'w')
error_count = 0
for dir_name in os.listdir('data'):
    dir_name = str(dir_name)
    fname = os.path.join('data', dir_name, f'{dir_name}.txt')
    orig_edges = open(os.path.join('data', dir_name, f'{dir_name}.edges'), 'r')
    local_edge_file = open(os.path.join('data', dir_name, f'{dir_name}_string.txt'), 'w')
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
        

global_edge_file.close()



