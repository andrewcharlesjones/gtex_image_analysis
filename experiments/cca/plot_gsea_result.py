import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as pjoin
import socket
import numpy as np
import os

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

# GSEA_RESULT_FILE = "./out/gsea_comp23_shared.csv"
# NUM_GENE_SETS_TO_PLOT = 5


# results = pd.read_csv(GSEA_RESULT_FILE, index_col=0)

# results = results.sort_values(["adj_pval", "pathway"]).iloc[:NUM_GENE_SETS_TO_PLOT, :]
# pathways = results.pathway.values
# pathways = [' '.join(x.split("_")) for x in pathways]

# plt.figure(figsize=(12, 10))
# plt.bar(np.arange(NUM_GENE_SETS_TO_PLOT), results.adj_pval.values)
# plt.xticks(np.arange(NUM_GENE_SETS_TO_PLOT), labels=pathways, rotation=-45, size=20, ha="left")
# plt.ylabel("p-value (adjusted)")
# plt.title("Gene set enrichment")
# plt.tight_layout()
# plt.savefig("./out/gsea_barplot.png")
# plt.show()



##### GSEA from lung ventilator analysis #####

GSEA_RESULT_FILE = "./out/lung_ventilator/gsea_comp38_shared.csv"
NUM_GENE_SETS_TO_PLOT = 5


results = pd.read_csv(GSEA_RESULT_FILE, index_col=0)

pathways = results.pathway.values
pathways = [' '.join(x.split("_")[1:]) for x in pathways]

plt.figure(figsize=(12, 10))

plt.scatter(results.NES.values, -np.log10(results.padj.values), s=50)

n_gene_sets_label = 3
results_sorted = results.sort_values('padj')
for ii in range(n_gene_sets_label):

	x = results_sorted.NES.values[ii]
	y = -np.log10(results_sorted.padj.values[ii])
	gs_name = results_sorted.pathway.values[ii]
	gs_name = ' '.join(gs_name.split("_")[1:])

	if ii == 0:
		plt.text(x - 0.4, y - 1, gs_name, fontsize=15, rotation=45)
	elif ii == 1:
		plt.text(x - 1, y - 3, gs_name, fontsize=15, rotation=45)


plt.title("Gene set enrichment")
plt.xlabel("Enrichment score")
plt.ylabel(r'-log$_{10}$(adj. p-value)')
plt.savefig("./out/lung_ventilator/gsea_barplot.png")

plt.show()


