import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import pickle
from os.path import join as pjoin
import socket
import matplotlib.pyplot as plt
import seaborn as sns
import umap

if socket.gethostname() == "andyjones":
    EXPRESSION_PATH = "/Users/andrewjones/Documents/beehive/multimodal_bio/organize_unnormalized_data/data/small_gene_tpm.gct"
    METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    GTEX_COLORS_PATH = "../../data/colors/tissue_gtex_colors.tsv"
else:
    EXPRESSION_PATH = "/tigress/aj13/gtexv8/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_tpm.gct"
    METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    GTEX_COLORS_PATH = "/projects/BEE/RNAseq/RNAseq_dev/Analysis/tissue_gtex_colors.tsv"

NUM_GENES = 20000
N_COMPONENTS = 2

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True


def main():

    # Metadata
    v8_metadata = pd.read_table(
        METADATA_PATH)

    # ------- Load expression ---------
    # Read in expression file
    expression_data = pd.read_table(EXPRESSION_PATH, skiprows=2, index_col=0)

    # Ignore transcript ID column
    expression_data = expression_data.drop(labels='transcript_id(s)', axis=1)

    # Make samples on rows
    expression_data = expression_data.transpose()

    # Drop duplicate sample names
    expression_data = expression_data[~expression_data.index.duplicated(keep='first')]
    v8_metadata = v8_metadata[~v8_metadata.SAMPID.duplicated(keep='first')]

    # Get samples' tissue labels
    samples_with_metadata = np.intersect1d(expression_data.index.values, v8_metadata.SAMPID.values)
    v8_metadata = v8_metadata.set_index("SAMPID")
    v8_metadata = v8_metadata.transpose()[samples_with_metadata].transpose()

    expression_data = expression_data.transpose()[samples_with_metadata].transpose()

    tissues = v8_metadata.SMTSD.values

    # import ipdb; ipdb.set_trace()
    expression_data = expression_data[tissues != "Kidney - Medulla"]
    v8_metadata = v8_metadata[tissues != "Kidney - Medulla"]
    tissues = tissues[tissues != "Kidney - Medulla"]
    

    assert expression_data.shape[0] == v8_metadata.shape[0]
    assert np.array_equal(expression_data.index.values, v8_metadata.index.values)



    # Replace NaNs
    expression_data = expression_data.fillna(0.0)

    # Remove any genes that show no expression
    expression_data = expression_data.iloc[:,
                                           (expression_data.sum(axis=0) > 0).values]

    # Get variances of each gene
    gene_vars = expression_data.var()

    # Get genes with highest variances
    gene_vars = gene_vars.sort_values(ascending=False)  # sort

    top_genes = gene_vars.index.values
    if NUM_GENES is not None:
        top_genes = top_genes[:NUM_GENES]  # subset

    # Subset data to highest variance genes
    expression_data = expression_data[top_genes]

    gene_names = expression_data.columns.values

    # Shorted sample IDs to match IDs from images
    expression_data.index = ["-".join(x.split("-")[:3])
                             for x in expression_data.index.values]

    expression_data = np.log(expression_data + 1)
    expression_data = expression_data - expression_data.mean()


    # ------- Run PCA/UMAP ---------

    # pca = PCA(n_components=N_COMPONENTS)
    # expression_reduced = pca.fit_transform(expression_data)

    pca = PCA(n_components=min(expression_data.shape[1], 50))
    umap_data = pca.fit_transform(expression_data)
    reducer = umap.UMAP()
    expression_reduced = reducer.fit_transform(umap_data)
    
    expression_reduced_df = pd.DataFrame(expression_reduced)
    expression_reduced_df['tissues'] = tissues
    expression_reduced_df.to_csv("./out/expression_reduced.csv")

    # ------- Plot ---------

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

    plt.figure(figsize=(20, 20))
    plt.scatter(expression_reduced[:, 0], expression_reduced[:, 1], c=color_df.tissue_color_rgb_normalized.values)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP, expression")
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig("./out/dimreduction_expression.png")
    plt.show()

    # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
