import os
import json
import mygene

mg = mygene.MyGeneInfo()
dataset = "reactome"
prot_dict = dict()
# prot_dict = dict(json.load(open('ncbi_prots.json')))

tried = set()
# tried = set(prot_dict.keys())
i = 0
for dir_name in os.listdir("data"):
    if not os.path.isdir(os.path.join("data", dir_name)):
        continue
    i += 1
    if i % 10 == 0:
        print(f"{i / 1763}% finished.")
    txt_file = open(os.path.join("data", dir_name, f"{dir_name}.txt"), "r")
    # print(dir_name)
    for line in txt_file:
        gene_a, interact, gene_b = line.strip().split("\t")
        for gene in [gene_a, gene_b]:
            if gene not in tried:
                tried.add(gene)
                try:
                    # _id = mg.query(gene)['hits'][0]['_id']
                    # uniprot = mg.getgene(_id)['uniprot']
                    # prot_dict[gene] = uniprot
                    uniprot = mg.query(f'symbol:{gene}',fields="uniprot", species="human")['hits'][0]['uniprot']['Swiss-Prot']
                    prot_dict[gene] = uniprot
                    # print(f"success: {gene}")
                except:
                    continue

json.dump(prot_dict, open("pathbank_prots.json", "w"))
