import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

df = pd.read_csv('reactome_unique_best_spaces.tsv', sep='\t')

x = np.array(df['Best Euclidean Distortion'])
idx = np.where(~np.isnan(x))
x = x[idx]
y = np.array(df['Best Overall Distortion'])
y = y[idx]
bad_idx = np.where(x == 10000000000.0)[0]
x = np.delete(x, bad_idx)
y = np.delete(y, bad_idx)

xy = np.vstack([x.copy(), y.copy()])
x_before = x.copy()
y_before = y.copy()
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

xpoints = np.linspace(0, np.max(x))
ypoints = np.linspace(0, np.max(x))
plt.xlabel('Best Euclidean Distortion')
plt.ylabel('Best Overall Distortion')
plt.title('Euclidean vs. Best Distortions')
plt.scatter(x, y, c=z, s=3)
plt.plot(xpoints, ypoints, linestyle='-', color='r', lw=2)
plt.savefig('plots/scatterplot_density.png')
