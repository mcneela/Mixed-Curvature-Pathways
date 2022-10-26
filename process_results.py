import os
import networkx as nx

all_data = {}
seen = {}
fout = open('all_data.stat', 'w')
fout.write('Graph num\tnum nodes\tnum edges\tH dim\tH copies\tS dim\tS copies\tE dim\tE copies\tBest loss\tBest MAP\tBest dist\tBest wc\tFinal loss\tFinal MAP\tFinal dist\tFinal wc\tme\tmc\n')
for fname in os.listdir('log'):
    # we only want to process the stat files
    if fname.endswith('stat'):
        filein = open(os.path.join('log', fname), 'r')
        parts = fname.split('.')
        graph_num = parts[0]
        h_dim = parts[1].split('-')[0][1:]
        h_copies = parts[1].split('-')[1]
        e_dim = parts[2].split('-')[0][1:]
        e_copies = parts[2].split('-')[1]
        s_dim = parts[3].split('-')[0][1:]
        s_copies = parts[3].split('-')[1]
        for i, line in enumerate(filein):
            if i == 0:
                # this is the header
                continue
            l = line.strip()
            best_loss, bmap, dist, wc, final_loss, fmap, dist, wc, me, mc = l.split('\t')
            # start processing graph attrs
            if graph_num not in seen:
                G = nx.read_edgelist(os.path.join('data', graph_num, f'{graph_num}.edges'))
                num_nodes = G.number_of_nodes()
                num_edges = G.number_of_edges()
                seen[graph_num] = {
                    'num_nodes': num_nodes,
                    'num_edges': num_edges,
                }
            else:
                num_nodes = seen[graph_num]['num_nodes']
                num_edges = seen[graph_num]['num_edges'] 
            fout.write(f'{graph_num}\t{num_nodes}\t{num_edges}\t{h_dim}\t{h_copies}\t{s_dim}\t{s_copies}\t{e_dim}\t{e_copies}\t' + l + '\n')
