import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

mixed_paths = [
    'nci/ncbi_unique_best_spaces.tsv',
    'pathbank/pathbank_unique_best_spaces.tsv',
    'reactome/reactome_unique_best_spaces.tsv',
    'humancyc/humancyc_unique_best_spaces.tsv',
    'kegg/kegg_unique_best_spaces.tsv'
]
laplace_paths = [
    'nci/ncbi_laplacian_rescaled_distortions.csv',
    'pathbank_laplacian_rescaled_distortions.csv',
    'reactome_laplacian_rescaled_distortions.csv',
    'humancyc_laplacian_rescaled_distortions.csv',
    'kegg_laplacian_rescaled_distortions.csv'
]
n2v_paths = [
    'nci/ncbi_n2v_rescaled_distortions.csv',
    'pathbank/pathbank_n2v_rescaled_distortions.csv',
    'reactome_n2v_rescaled_distortions.csv',
    'humancyc_n2v_rescaled_distortions.csv',
    'kegg/kegg_n2v_rescaled_distortions.csv'
]

mixed_dist, laplace_dist, n2v_dist, euc_dist = [], [], [], []
for fname in mixed_paths:
    df = pd.read_csv(fname, sep='\t')
    dist = df['Best Overall Distortion'].tolist()
    mixed_dist += dist
    dist = df['Best Euclidean Distortion'].tolist()
    euc_dist += dist
for fname in laplace_paths:
    df = pd.read_csv(fname, sep=',')
    dist = df['Distortion'].tolist()
    laplace_dist += dist
for fname in n2v_paths:
    df = pd.read_csv(fname, sep=',')
    dist = df['Distortion'].tolist()
    n2v_dist += dist

df_dict = {
    'Embedding Type': ['Laplacian'] * len(laplace_dist) + ['Node2Vec'] * len(n2v_dist) +  ['Mixed Curvature'] * len(mixed_dist) + ['Euclidean'] * len(euc_dist),
    'Distortion': laplace_dist + n2v_dist + mixed_dist + euc_dist
}

my_pal = {"Laplacian": "g", "Node2Vec": "b", "Mixed Curvature": "m", "Euclidean": "e"}

df = pd.DataFrame.from_dict(df_dict)
sns.violinplot(data=df, x="Embedding Type", y="Distortion",)# palette=my_pal)
plt.savefig('multiset/plots/violin_plot.png')
plt.show()