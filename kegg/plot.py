import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('kegg_unique_best_spaces.tsv', sep='\t')

# sns.histplot(data=df, x='% Dist Diff From Euclidean')
# plt.title('Percentage Improvement in Distortion over Euclidean')
# plt.savefig('plots/histogram_percent_improvement.png')
# plt.clf()

# sns.histplot(data=df, x='Dist Diff from Euclidean')
# plt.title('Raw Improvement in Distortion over Euclidean')
# plt.savefig('plots/histogram_raw_improvement.png')
# plt.clf()

# x = np.array(df['Best Euclidean Distortion'])
# x = x[np.where(~np.isnan(x))]
# y = np.array(df['Best Overall Distortion'])
# y = y[np.where(~np.isnan(x))]
# xpoints = ypoints = np.linspace(0, np.max(x))
# plt.plot(xpoints, ypoints, linestyle='-', color='r', lw=2)
# sns.scatterplot(data=df, x='Best Euclidean Distortion', y='Best Overall Distortion')
# plt.title('Scatterplot of Best Euclidean vs. Best Overall Distortion')
# plt.savefig('plots/scatterplot_dist_euclidean_vs_overall.png')
# plt.clf()

from scipy.stats import gaussian_kde

# not shown: matplotlib code to set up the figure, axis etc
# x and y are numpy arrays of the x and y values of the points
# fig, ax = plt.subplots()
x = np.array(df['Best Euclidean Distortion'])
idx = np.where(~np.isnan(x))
x = x[idx]
y = np.array(df['Best Overall Distortion'])
y = y[idx]
# midx = np.where(x == x.max())
# x = np.delete(x, midx)
# y = np.delete(y, midx)
# x = np.concatenate((x[:x.argmax()], x[x.argmax():]))
# y = np.concatenate((y[:x.argmax()], y[x.argmax():]))

xy = np.vstack([x.copy(), y.copy()])
x_before = x.copy()
y_before = y.copy()
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

xpoints = ypoints = np.linspace(0, np.max(x))
plt.xlabel('Best Euclidean Distortion')
plt.ylabel('Best Overall Distortion')
plt.title('Density Plot of Euclidean vs. Best Distortions (KEGG)')
plt.scatter(x, y, c=z, s=3)
plt.plot(xpoints, ypoints, linestyle='-', color='r', lw=2)
plt.savefig('scatterplot_density_kegg.png')
plt.show()


# create a figure and axis
# fig, ax = plt.subplots(1)
# # plot the line of equivalence
# # sns.lmplot(xpoints, ypoints, linestyle='-', color='r', lw=2)
# ax.set(xlabel='Best Euclidean Distortion', 
#        ylabel='Best Overall Distortion',
#        title='Density Plot of Euclidean vs. Best Distortions')
# sns.scatterplot(x=x, y=y, c=z)
# plt.show()
# plt.savefig('plots/scatterplot_density.png')
# plt.close(fig)