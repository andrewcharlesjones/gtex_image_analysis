import socket
import pandas as pd
import numpy as np
from os.path import join as pjoin
import os
# from pcca import PCCA
# from pcca_pytorch import PCCA
# from pcca_shared_and_exclusive import PCCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

import pcca_tfp

if socket.gethostname() == "andyjones":
    EXPRESSION_PATH = "/Users/andrewjones/Documents/beehive/multimodal_bio/organize_unnormalized_data/data/small_gene_tpm.gct"
    IMG_DATA_DIR = "/Users/andrewjones/Documents/beehive/gtex_image_analysis/experiments/autoencoder/out"
    SAVE_DIR = "./out"
else:
    EXPRESSION_PATH = "/tigress/aj13/gtexv8/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_tpm.gct"
    IMG_DATA_DIR = "/tigress/aj13/gtex_image_analysis/autoencoder/out"
    SAVE_DIR = "/tigress/aj13/gtex_image_analysis/cca/pcca_out"


NUM_GENES = 10000
NUM_ITERS = 1000
NUM_PCA_COMPONENTS = 1024
NUM_CCA_COMPONENTS = 50


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

    # ------ Run PCA on expression data to reduce dimensionality ------
    pca = PCA(n_components=min(NUM_PCA_COMPONENTS, expression_data.shape[1]))
    pca.fit(expression_data.values)
    expression_data_transformed = pca.transform(expression_data.values)
    expression_data_transformed = pd.DataFrame(
        expression_data_transformed, index=expression_data.index.values)

    # Save PCA components so we can recover the gene CCA coefficients later
    # pca_comps = pd.DataFrame(pca.components_, columns=gene_names)
    # pca_comps.to_csv(pjoin(SAVE_DIR, "pca_comps.csv"))
    # np.save(pjoin(SAVE_DIR, "gene_pca_components.npy"), pca.components_)
    # import ipdb; ipdb.set_trace()

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
    expression_data_transformed = expression_data_transformed.transpose()[
        shared_samples].transpose()
    img_data = img_data.transpose()[shared_samples].transpose()

    # Whiten image embeddings
    pca_im = PCA(n_components=min(img_data.shape))
    img_data_whitened = pca_im.fit_transform(img_data.values)
    img_data_whitened = pd.DataFrame(img_data_whitened, index=img_data.index)

    img_fname_df = pd.DataFrame({'fname': img_fnames}, index=img_sample_ids)
    img_fname_df = img_fname_df.transpose()[shared_samples].transpose()

    # expression_data = pd.read_csv(pjoin(DATA_DIR, "expression_data_for_cca.csv"), index_col=0)
    # img_data = pd.read_csv(pjoin(DATA_DIR, "img_data_for_cca.csv"), index_col=0)

    assert expression_data_transformed.shape[0] == img_data_whitened.shape[0]
    assert np.array_equal(
        expression_data_transformed.index.values, img_data_whitened.index.values)
    assert np.all(np.unique(
        expression_data_transformed.index.values, return_counts=True)[1] == 1)
    assert np.all(np.unique(img_data_whitened.index.values, return_counts=True)[1] == 1)

    print("\nimg shape: {}".format(img_data_whitened.shape))
    print("exp shape: {}".format(expression_data_transformed.shape))

    # ------ Run PCCA ---------
    print("\nRunning PCCA...")


    data_concat = np.concatenate([img_data_whitened.values, expression_data_transformed.values], axis=1)

    #  # Square features (this was a test for didong)
    # img_data_whitened = np.concatenate([img_data_whitened, img_data_whitened**2], axis=1)
    # expression_data_transformed = np.concatenate([expression_data_transformed, expression_data_transformed**2], axis=1)


    n = img_data_whitened.shape[0]
    p1, p2 = img_data_whitened.shape[1], expression_data_transformed.shape[1]

    pcca_model = pcca_tfp.pcca_model(data_dim_x1=p1, data_dim_x2=p2, latent_dim=NUM_CCA_COMPONENTS, num_datapoints=n, stddv_datapoints=1)
    model_dict = pcca_tfp.fit_pcca_model(model=pcca_model, x1_train=img_data_whitened.T, x2_train=expression_data_transformed.T, latent_dim=NUM_CCA_COMPONENTS, stddv_datapoints=1)


    plt.plot(model_dict['loss_trace'])
    plt.savefig(pjoin(SAVE_DIR, "pcca_loss_trace.png"))
    # plt.show()
    

    # LVs and cefficients for shared space
    lvs_shared = model_dict['zshared'].numpy()

    img_coeffs_shared = model_dict['lambda1'].numpy()
    exp_coeffs_shared = model_dict['lambda2'].numpy()

    # LVs and coefficients for exclusive spaces
    img_lvs_exlusive = model_dict['z1'].numpy()
    exp_lvs_exlusive = model_dict['z2'].numpy()

    img_coeffs_exclusive = model_dict['b1'].numpy()
    exp_coeffs_exclusive = model_dict['b2'].numpy()

    

    # ----- Convert PCA-projected coefficients to gene coefficients -----
    full_pcca_exp_coeffs_shared = pca.components_.T @ exp_coeffs_shared
    full_pcca_exp_coeffs_exclusive = pca.components_.T @ exp_coeffs_exclusive

    # --------- Save outputs ------
    print("Saving outputs...")


    # import ipdb; ipdb.set_trace()

    pd.DataFrame(lvs_shared).to_csv(pjoin(SAVE_DIR, "pcca_lvs_shared.csv"))

    # Image
    pd.DataFrame(img_lvs_exlusive).to_csv(pjoin(SAVE_DIR, "pcca_img_lvs_exclusive.csv"))

    pd.DataFrame(img_coeffs_shared).to_csv(pjoin(SAVE_DIR, "pcca_img_coeffs_shared.csv"))
    pd.DataFrame(img_coeffs_exclusive).to_csv(pjoin(SAVE_DIR, "pcca_img_coeffs_exclusive.csv"))

    # Expression
    pd.DataFrame(exp_lvs_exlusive).to_csv(pjoin(SAVE_DIR, "pcca_exp_lvs_exclusive.csv"))

    pd.DataFrame(full_pcca_exp_coeffs_shared, index=gene_names).to_csv(pjoin(SAVE_DIR, "pcca_exp_coeffs_shared.csv"))
    pd.DataFrame(full_pcca_exp_coeffs_exclusive, index=gene_names).to_csv(pjoin(SAVE_DIR, "pcca_exp_coeffs_exclusive.csv"))


    img_fname_df.to_csv(pjoin(SAVE_DIR, "img_fnames_pcca.csv"))

    # import ipdb; ipdb.set_trace()

    print("Done!")

    


if __name__ == "__main__":
    main()
