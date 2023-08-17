import argparse 

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='file to convert')
    args = parser.parse_args()
    convert_sif_to_edge_list(args.path)
