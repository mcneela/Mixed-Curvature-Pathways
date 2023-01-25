import sys
from collections import defaultdict

filein = open('all_data.stat', 'r')
fileout = open('best_configs.tsv', 'w')

best = defaultdict(lambda: sys.float_info.max)
line_d = defaultdict(str)
for i, line in enumerate(filein):
    if i == 0:
        header = line
        continue
    parts = line.split('\t')
    graph_num, best_loss = parts[0], float(parts[11])
    existing = best[graph_num]
    if best_loss < existing:
        best[graph_num] = best_loss
        line_d[graph_num] = line

fileout.write(header)
for k, v in line_d.items():
    fileout.write(v)



    