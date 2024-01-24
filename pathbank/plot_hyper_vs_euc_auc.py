import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pb_hyp_aucs = [0.90, 0.90, 0.92, 0.92, 0.91, 0.83, 0.92, 0.50, 0.87, 0.87, 0.92, 0.98, 0.92, 0.92, 0.90, 0.92, 0.67, 0.91, 0.95, 0.92, 0.92, 0.92, 0.92, 0.92]
pb_euc_aucs = [0.69, 0.69, 0.79, 0.79, 0.70, 0.55, 0.94, 0.58, 0.66, 0.66, 0.64, 0.87, 0.79, 0.64, 0.69, 0.79, 0.59, 0.61, 0.78, 0.79, 0.64, 0.64, 0.64, 0.64]
string_hyp_aucs = list(map(lambda x: float(x.split('\t')[1]), open('../../hgcn/initial_lp_results_new.tsv').readlines()[1:]))
string_euc_aucs = list(map(lambda x: float(x.split('\t')[2]), open('../../hgcn/initial_lp_results_new.tsv').readlines()[1:])) 

data = {
    "x": pb_euc_aucs + string_euc_aucs,
    "y": pb_hyp_aucs + string_hyp_aucs,
    "Dataset": ["PathBank (val, in-distribution)"] * len(pb_euc_aucs) + ["STRING (test, out-of-distribution)"] * len(string_euc_aucs)
}
df = pd.DataFrame(data)
xpoints = ypoints = np.linspace(np.min(pb_euc_aucs + string_euc_aucs), np.max(pb_hyp_aucs + string_hyp_aucs))

plt.ylabel('Hyperbolic AUC')
plt.xlabel('Euclidean AUC')
plt.title('Euclidean vs. Hyperbolic Edge Prediction AUC')
sns.scatterplot(data=df, x="x", y="y", hue="Dataset")
# plt.scatter(euc_aucs, hyp_aucs)
plt.plot(xpoints, ypoints, linestyle='-', color='r', lw=2)
plt.savefig('plots/scatterplot_hyp_vs_euc_both_ds.png')
plt.show()