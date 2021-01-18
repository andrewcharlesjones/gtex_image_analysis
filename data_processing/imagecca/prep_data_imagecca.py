import socket
import pandas as pd
import numpy as np
from os.path import join as pjoin
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

if socket.gethostname() == "andyjones":
    EXPRESSION_PATH = "/Users/andrewjones/Documents/beehive/multimodal_bio/organize_unnormalized_data/data/medium_gene_tpm.gct"
    IMG_DATA_DIR = "/Users/andrewjones/Documents/beehive/gtex_image_analysis/experiments/autoencoder/out"
    SAVE_DIR = "/Users/andrewjones/Documents/beehive/gtex_image_analysis/data/imagecca"
else:
    EXPRESSION_PATH = "/tigress/aj13/gtexv8/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_tpm.gct"
    IMG_DATA_DIR = "/tigress/aj13/gtex_image_analysis/autoencoder/out"
    SAVE_DIR = "/tigress/aj13/gtex_image_analysis/imagecca/data"


NUM_GENES = 20000

def main():

    # ----- Load data -------
    print("Loading data...")

    # Load expression
    expression_data = pd.read_table(EXPRESSION_PATH, skiprows=2, index_col=0)

    # Ignore transcript ID column
    expression_data = expression_data.drop(labels='transcript_id(s)', axis=1)

    # Make samples on rows
    expression_data = expression_data.transpose()

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

    # Make samples on rows
    # expression_data = expression_data.transpose()
    gene_names = expression_data.columns.values

    # Shorted sample IDs to match IDs from images
    expression_data.index = ["-".join(x.split("-")[:3])
                             for x in expression_data.index.values]

    expression_data = np.log(expression_data + 1)

    # expression_data = (expression_data - expression_data.mean()
    #                    ) / expression_data.std()
    expression_data = expression_data - expression_data.mean()



    # ----- Load image embeddings -----
    img_embeddings = np.load(
        pjoin(IMG_DATA_DIR, "latent_z.npy"), allow_pickle=True)

    img_fnames = np.load(
        pjoin(IMG_DATA_DIR, "im_fnames.npy"), allow_pickle=True)
    img_sample_ids = [os.path.basename(x).split('.')[0] for x in img_fnames]

    # Add 1 so that they match expression IDs
    img_sample_ids = [x[:-1] + str(int(x[-1]) + 1) for x in img_sample_ids]

    img_data = pd.DataFrame(img_embeddings)
    img_data.index = img_sample_ids

    # ------- Find shared samples ------
    shared_samples = np.intersect1d(
        expression_data.index.values, img_data.index.values)
    expression_data = expression_data.transpose()[
        shared_samples].transpose()
    img_data = img_data.transpose()[shared_samples].transpose()

    # Whiten image embeddings
    pca_im = PCA(n_components=min(img_data.shape))
    img_data_whitened = pca_im.fit_transform(img_data.values)
    img_data_whitened = pd.DataFrame(img_data_whitened, index=img_data.index)

    img_fname_df = pd.DataFrame({'fname': img_fnames}, index=img_sample_ids)
    img_fname_df = img_fname_df.transpose()[shared_samples].transpose()

    assert expression_data.shape[0] == img_data_whitened.shape[0]
    assert np.array_equal(
        expression_data.index.values, img_data_whitened.index.values)
    assert np.all(np.unique(
        expression_data.index.values, return_counts=True)[1] == 1)
    assert np.all(np.unique(img_data_whitened.index.values, return_counts=True)[1] == 1)

    print("\nimg shape: {}".format(img_data_whitened.shape))
    print("exp shape: {}".format(expression_data.shape))

    # ------- Save ------
    expression_data.to_csv(pjoin(SAVE_DIR, "expression_data_for_cca.csv"))
    img_data_whitened.to_csv(pjoin(SAVE_DIR, "img_data_for_cca.csv"))
    img_fname_df.to_csv(pjoin(SAVE_DIR, "img_fnames.csv"))

    

    


if __name__ == "__main__":
    main()
