import os

#learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, .05, .1, 1]
learning_rates = [1e-3]
hyperbolic_copies = [0, 1, 2, 3]
spherical_copies = [0, 1, 2, 3]
euclidean_copies = [0, 1, 2, 3]

dims = [33, 100]

hyperparam_file = open('../args.tsv', 'w')
input_dir = "data"

combo_template = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n"

for subdir in os.listdir(input_dir)[:1000]:
    fpath = os.path.join(input_dir, subdir, f"{subdir}.edges")
    print(fpath)
    for l in learning_rates:
        for h in hyperbolic_copies:
            for s in spherical_copies:
                for e in euclidean_copies:
                    if e == 0 and h == 0 and s == 0:
                            continue
                    edim = 100 // (e + s + h)
                    hdim = 100 // (e + s + h)
                    sdim = 100 // (e + s + h)
                    hparam_str = combo_template.format(fpath, l, e, edim, h, hdim, s, sdim)
                    hyperparam_file.write(hparam_str)
hyperparam_file.close()