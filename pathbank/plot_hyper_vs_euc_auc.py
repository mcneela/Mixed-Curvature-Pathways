import numpy as np
import matplotlib.pyplot as plt

hyp_aucs = [0.90, 0.90, 0.92, 0.92, 0.91, 0.83, 0.92, 0.50, 0.87, 0.87, 0.92, 0.98, 0.92, 0.92, 0.90, 0.92, 0.67, 0.91, 0.95, 0.92, 0.92, 0.92, 0.92, 0.92]
euc_aucs = [0.69, 0.69, 0.79, 0.79, 0.70, 0.55, 0.94, 0.58, 0.66, 0.66, 0.64, 0.87, 0.79, 0.64, 0.69, 0.79, 0.59, 0.61, 0.78, 0.79, 0.64, 0.64, 0.64, 0.64]

xpoints = ypoints = np.linspace(np.min(euc_aucs), np.max(hyp_aucs))
plt.ylabel('Hyperbolic AUC')
plt.xlabel('Euclidean AUC')
plt.title('Euclidean vs. Hyperbolic Edge Prediction AUC')
plt.scatter(euc_aucs, hyp_aucs)
plt.plot(xpoints, ypoints, linestyle='-', color='r', lw=2)
plt.savefig('plots/scatterplot_hyp_vs_euc_auc.png')
plt.show()