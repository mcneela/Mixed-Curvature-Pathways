import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

mixed_paths = [
    'ncbi/ncbi_unique_best_spaces.tsv',
    'pathbank/pathbank_unique_best_spaces.tsv',
    'reactome/reactome_unique_best_spaces.tsv',
    'humancyc/humancyc_unique_best_spaces.tsv',
    'kegg/kegg_unique_best_spaces.tsv'
]
laplace_paths = [
    'ncbi/ncbi_laplacian_distortions.csv',
    'pathbank/pathbank_laplacian_distortions.csv',
    'reactome/reactome_laplacian_distortions.csv',
    'humancyc/humancyc_laplacian_distortions.csv',
    'kegg/kegg_laplacian_distortions.csv'
]
n2v_paths = [
    'ncbi/ncbi_n2v_distortions.csv',
    'pathbank/pathbank_n2v_distortions.csv',
    # 'reactome/reactome_n2v_distortions.csv',
    'humancyc/humancyc_n2v_distortions.csv',
    'kegg/kegg_n2v_distortions.csv'
]

mixed_dist, laplace_dist, n2v_dist = [], [], []
for fname in mixed_paths:
    df = pd.read_csv(fname, sep='\t')
    dist = df['Best Overall Distortion'].tolist()
    mixed_dist += dist
for fname in laplace_paths:
    df = pd.read_csv(fname, sep=',')
    dist = df['Distortion'].tolist()
    laplace_dist += dist
for fname in n2v_paths:
    df = pd.read_csv(fname, sep=',')
    dist = df['Distortion'].tolist()
    n2v_dist += dist

df_dict = {
    'Embedding Type': ['Laplacian'] * len(laplace_dist) + ['Node2Vec'] * len(n2v_dist) +  ['Mixed Curvature'] * len(mixed_dist),
    'Distortion': laplace_dist + n2v_dist + mixed_dist
}
df = pd.DataFrame.from_dict(df_dict)
sns.violinplot(data=df, x="Embedding Type", y="Distortion")
plt.show()