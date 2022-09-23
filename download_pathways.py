import os
import csv
import subprocess

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
            id_path = f'data/{p}'
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
        e1, e_type,	e2,	source,	pid, names, med_ids = line
        all_pids = pid.split(';')
        for p in all_pids:
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
        path = os.path.join(data_dir, subdir, f"{subdir}.txt")
        convert_sif_to_edge_list(path)

if __name__ == "__main__":
    # gen_all_pathway_files('PathwayCommons12.All.hgnc.txt')
    convert_all_sif_to_edge('data')