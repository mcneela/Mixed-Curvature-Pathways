import os
import csv
import subprocess
from collections import defaultdict

def analyze_pathway_edge_types(fpath):
    with open(fpath, 'r') as pathfile:
        reader = csv.reader(pathfile, delimiter='\t')
        edge_types = defaultdict(int)
        error_count = 0
        for i, line in enumerate(reader):
            if i == 0:
                continue
            if i % 10000 == 0:
                print(f"Processing line {i}")
            try:
                n1, e_type, n2, data_source, pubmed_id, pathway_names, med_ids = line
            except:
                error_count += 1
                continue
            edge_types[e_type] += 1
    return edge_types

def generate_pathways(fpath):
    with open(fpath, 'r') as pathfile:
        pathways = defaultdict(list)
        pathway_to_int = {}
        reader = csv.reader(pathfile, delimiter='\t')
        error_count = 0
        for i, line in enumerate(reader):
            if i == 0:
                continue
            if i % 10000 == 0:
                print(f"Processing line {i}")
            try:
                n1, e_type, n2, data_source, pubmed_id, pathway_names, med_ids = line
            except:
                error_count += 1
                continue
            for name in pathway_names.split(';'):
                if name not in pathway_to_int:
                    int_id = len(pathway_to_int)
                    pathway_to_int[name] = int_id
                else:
                    int_id = pathway_to_int[name]
                pathways[int_id].append(f"{n1}\t{e_type}\t{n2}\n")
    return error_count, pathways, pathway_to_int

def write_pathways_to_files(pathways):
    for pid in pathways:
        edges = pathways[pid]
        id_path = f"data/{pid}"
        if not os.path.exists(id_path):
            os.makedirs(id_path)
        with open(f"data/{pid}/{pid}.txt", "w") as datafile:
            for e in edges:
                datafile.write(e)
        

def gen_all_pathway_files(fpath):
    pathway_txt = open(fpath, 'r')
    files_by_id = {}
    for i, line in enumerate(csv.reader(pathway_txt, delimiter='\t')):
        if i == 0:
            continue
        if i % 10000 == 0:
            print(f"Processing line {i}")
        e1, e_type,	e2,	source,	pid, names, med_ids = line
        all_pids = pid.split(';')
        for p in all_pids:
            id_path = f'pathway/{p}'
            if not os.path.exists(id_path):
                os.makedirs(id_path)
            if p not in files_by_id:
                files_by_id[p] = open(os.path.join(id_path, f'{p}.txt'), 'w')
            files_by_id[p].write(f'{e1}\t{e_type}\t{e2}\n')

    for p in files_by_id:
        files_by_id[p].close()

def efficient_gen_all_pathway_files(fpath):
    pathway_txt = open(fpath, 'r')
    files_by_id = {}
    for i, line in enumerate(csv.reader(pathway_txt, delimiter='\t')):
        if i == 0:
            continue
        try:
            e1, e_type,	e2,	source,	pid, names, med_ids = line
        except:
            continue
        if 'Pathbank' not in source:
            continue
        if ";" in pid:
            all_pids = pid.split(';')
        else:
            all_pids = [pid]
        for p in all_pids:
            if p == '.txt':
                continue
            id_path = f'data/{p}'
            if not os.path.exists(id_path):
                os.makedirs(id_path)
            if p not in files_by_id:
                files_by_id[p] = os.path.join(id_path, f'{p}.txt')
                infile = open(files_by_id[p], 'w')
            else:
                infile = open(files_by_id[p], 'a')
            infile.write(f'{e1}\t{e_type}\t{e2}\n')
            infile.close()

def convert_sif_to_edge_list(fname):
    filein = open(fname, 'r')
    node_dict = {}
    i = 0
    edges = []
    fileout = open(f"{fname[:-4]}.edges", "w")
    for line in filein:
        n1, interact, n2 = line.strip().split('\t')
        if n1 not in node_dict:
            node_dict[n1] = i
            i += 1
        if n2 not in node_dict:
            node_dict[n2] = i
            i += 1
        edges.append((node_dict[n1], node_dict[n2]))
    filein.close()
    for e in edges:
        fileout.write(f"{e[0]}\t{e[1]}\n")
    fileout.close()
    
def convert_all_sif_to_edge(data_dir):
    for subdir in os.listdir(data_dir):
        if subdir == '.txt':
            continue
        path = os.path.join(data_dir, subdir, f"{subdir}.txt")
        convert_sif_to_edge_list(path)

if __name__ == "__main__":
    e_count, pathways, pathway_to_int = generate_pathways('pathbank/PathwayCommons12.pathbank.hgnc.txt')
    # write_pathways_to_files(pathways)
    # convert_all_sif_to_edge('data')
    # edge_types = analyze_pathway_edge_types('pathbank/PathwayCommons12.pathbank.hgnc.txt')
