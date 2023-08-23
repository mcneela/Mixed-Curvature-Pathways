import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate multiple datasets (for demonstration purposes)
datasets = [
    {'x_column': 'Best Euclidean Distortion', 'y_column': 'Best Overall Distortion', 'label': 'Reactome'},
    {'x_column': 'Best Euclidean Distortion', 'y_column': 'Best Overall Distortion', 'label': 'Pathbank'},
    {'x_column': 'Best Euclidean Distortion', 'y_column': 'Best Overall Distortion', 'label': 'HumanCyc'},
    {'x_column': 'Best Euclidean Distortion', 'y_column': 'Best Overall Distortion', 'label': 'KEGG'},
    {'x_column': 'Best Euclidean Distortion', 'y_column': 'Best Overall Distortion', 'label': 'NCBI'}
]

dfs = [
    pd.read_csv('reactome/reactome_unique_best_spaces.tsv', sep='\t'),
    pd.read_csv('pathbank/pathbank_unique_best_spaces.tsv', sep='\t'),
    pd.read_csv('humancyc/humancyc_unique_best_spaces.tsv', sep='\t'),
    pd.read_csv('kegg/kegg_unique_best_spaces.tsv', sep='\t'),
    pd.read_csv('ncbi/ncbi_unique_best_spaces.tsv', sep='\t')
]

# Create a color map for each dataset
color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

# Plot each dataset with a different color map
plt.xlabel('Best Euclidean Distortion')
plt.ylabel('Best Overall Distortion')
plt.title('Distortion Across Pathway Datasets')

for i, (dataset, df) in enumerate(zip(datasets, dfs)):
    x = np.array(df[dataset['x_column']])
    y = np.array(df[dataset['y_column']])
    
    # Remove NaN and bad values
    idx = np.where(~np.isnan(x))
    x = x[idx]
    y = y[idx]
    bad_idx = np.where(x == 10000000000.0)[0]
    x = np.delete(x, bad_idx)
    y = np.delete(y, bad_idx)
    
    xy = np.vstack([x.copy(), y.copy()])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    # Plot the scatter plot with the current color map
    xpoints = np.linspace(0, 0.4, 50)

    plt.scatter(x, y, c=z, cmap=color_maps[i], s=3, label=dataset['label'])
    plt.plot(xpoints, xpoints, linestyle='-', color='r', lw=2)

# Add color bar
plt.colorbar(label='Density')

# Add legend
plt.legend()

# Save the plot
plt.savefig('multiset/plots/multi_dataset_scatterplot_density.png', dpi=300)

plt.xlim((0, 0.1))
plt.ylim((0, 0.1))
plt.savefig('multiset/plots/multi_dataset_scatterplot_density_zoomed.png', dpi=300)
