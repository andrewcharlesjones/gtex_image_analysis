import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


GTEX_COLORS_PATH = "../../data/colors/tissue_gtex_colors.tsv"

NUM_GENES = 20000
N_COMPONENTS = 2

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True


expression_reduced = pd.read_csv("./out/expression_reduced.csv", index_col=0)
expression_reduced = expression_reduced.values
# import ipdb; ipdb.set_trace()

# Load GTEx colors
gtex_colors = pd.read_table(GTEX_COLORS_PATH)

color_lists = gtex_colors.tissue_color_rgb.str.split(",").values
reds = [int(x[0]) / 255. for x in color_lists]
greens = [int(x[1]) / 255. for x in color_lists]
blues = [int(x[2]) / 255. for x in color_lists]
zipped_colors = [np.array([reds[ii], greens[ii], blues[ii]]) for ii in range(color_lists.shape[0])]
gtex_colors['tissue_color_rgb_normalized'] = zipped_colors
color_df = pd.DataFrame(tissues, columns=['tissue'])
color_df = pd.merge(color_df, gtex_colors[['tissue_name', 'tissue_color_rgb_normalized']], left_on='tissue', right_on='tissue_name', how='left')

plt.figure(figsize=(7, 6))
plt.scatter(expression_reduced[:, 0], expression_reduced[:, 1], c=color_df.tissue_color_rgb_normalized.values)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP, expression")
plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig("./out/dimreduction_expression.png")
plt.show()