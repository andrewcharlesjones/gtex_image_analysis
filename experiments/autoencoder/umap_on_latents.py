import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import matplotlib.image as mpimg
from os.path import join as pjoin
import socket
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import umap

if socket.gethostname() == "andyjones":
    SAVE_DIR = "./out"
    GTEX_COLORS_PATH = "../../data/colors/tissue_gtex_colors.tsv"
    METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
else:
    SAVE_DIR = "/tigress/aj13/gtex_image_analysis/autoencoder/out/"
    GTEX_COLORS_PATH = "/projects/BEE/RNAseq/RNAseq_dev/Analysis/tissue_gtex_colors.tsv"
    METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"

N_COMPONENTS = 2
PERPLEXITY = 100


import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

def main():

    # ---- Load latent representations and image filenames ----
    print("Loading data...")
    latent_z = np.load(pjoin(SAVE_DIR, "latent_z.npy"), allow_pickle=True)
    # tissues = np.load(pjoin(SAVE_DIR, "tissue_labels.npy"), allow_pickle=True)
    im_fnames = np.load(pjoin(SAVE_DIR, "im_fnames.npy"), allow_pickle=True)

    sample_ids = [os.path.basename(x).split(".")[0] for x in im_fnames]
    sample_ids = [x[:-1] + str(int(x[-1]) + 1) for x in sample_ids]
    sample_id_df = pd.DataFrame(sample_ids, columns=['sample_id'])

    # Metadata
    v8_metadata = pd.read_table(METADATA_PATH)
    v8_metadata['sample_id'] = v8_metadata.SAMPID.str.split("-").str[:3].str.join("-")
    v8_metadata = v8_metadata[~v8_metadata.sample_id.duplicated(keep='first')]

    

    shared_samples = np.intersect1d(v8_metadata.sample_id.values, sample_id_df.sample_id.values)

    shared_idx1 = np.where(np.isin(sample_id_df.sample_id.values, shared_samples))[0]
    sample_id_df = sample_id_df.iloc[shared_idx1, :]
    latent_z = latent_z[shared_idx1]
    im_fnames = im_fnames[shared_idx1]

    shared_idx2 = np.where(np.isin(v8_metadata.sample_id.values, shared_samples))[0]
    v8_metadata = v8_metadata.iloc[shared_idx2, :]

    sample_id_df = pd.merge(sample_id_df, v8_metadata[["sample_id", "SMTSD"]], on="sample_id", how='left')
    latent_z = latent_z[sample_id_df.SMTSD != "Kidney - Medulla"]
    im_fnames = im_fnames[sample_id_df.SMTSD != "Kidney - Medulla"]
    sample_id_df = sample_id_df[sample_id_df.SMTSD != "Kidney - Medulla"]
    tissues = sample_id_df.SMTSD.values

    print("Doing PCA...")
    pca = PCA(n_components=min(min(latent_z.shape), 50))
    pca.fit(latent_z)
    transformed_data_pca = pca.transform(latent_z)

    # ---- Do tSNE ----
    print("Fitting UMAP...")

    reducer = umap.UMAP()
    transformed_data = reducer.fit_transform(transformed_data_pca)
    transformed_df = pd.DataFrame(transformed_data, columns=["umap1", "umap2"])

    # ----- Make plot of all tissues ------
    # print("Plotting...")
    # plt.figure(figsize=(13, 10))
    # sns.scatterplot(data=transformed_df, x="umap1", y="umap2", hue="tissue")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.tight_layout()
    # plt.savefig("./out/umap_on_latents.png", bbox_inches='tight')
    # plt.close()


    # # ----- Make grid of plots, each one with a different tissue colored -------
    # tissues_unique = np.unique(tissues)
    # n_tissues = len(tissues_unique)
    # grid_size = np.ceil(np.sqrt(n_tissues))
    # plt.figure(figsize=(20, 20))
    # for ii, curr_tissue in enumerate(tissues_unique):

    #     plt.subplot(grid_size, grid_size, ii + 1)
    #     curr_tissue_idx = np.where(tissues == curr_tissue)[0]
    #     other_idx = np.where(tissues != curr_tissue)[0]
    #     plt.scatter(transformed_df.umap1.values[other_idx], transformed_df.umap2.values[other_idx], label="Other", color="gray", alpha=0.2)
    #     plt.scatter(transformed_df.umap1.values[curr_tissue_idx], transformed_df.umap2.values[curr_tissue_idx], label=curr_tissue)
    #     plt.title(curr_tissue)


    
    # plt.savefig("./out/umap_on_latents_by_tissue.png")



    # ----------- Plot images over umap points -----------
    n = latent_z.shape[0]
    num_ims_to_plot = 2000
    num_ims_to_plot = min(n, num_ims_to_plot)
    if num_ims_to_plot is None:
        rand_idx = np.arange(n)
    else:
        rand_idx = np.random.choice(
            np.arange(n), size=num_ims_to_plot, replace=False)

    fnames_to_plot = im_fnames[rand_idx]
    tissues_to_plot = tissues[rand_idx]


    umap_x, umap_y = transformed_df.umap1.values[rand_idx], transformed_df.umap2.values[rand_idx]

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(umap_x, umap_y)

    # Get a unique color for each tissue. We'll color the border of each image with this color.
    tissues_unique = np.unique(tissues_to_plot)
    NUM_COLORS = len(tissues_unique)


    # Load GTEx colors
    gtex_colors = pd.read_table(GTEX_COLORS_PATH)
    
    color_lists = gtex_colors.tissue_color_rgb.str.split(",").values
    reds = [int(x[0]) / 255. for x in color_lists]
    greens = [int(x[1]) / 255. for x in color_lists]
    blues = [int(x[2]) / 255. for x in color_lists]
    zipped_colors = [np.array([reds[ii], greens[ii], blues[ii]]) for ii in range(color_lists.shape[0])]
    gtex_colors['tissue_color_rgb_normalized'] = zipped_colors
    

    # Look over images, plotting each one at the correct location
    
    for x0, y0, path, tissue in zip(umap_x, umap_y, fnames_to_plot, tissues_to_plot):

        # Get tissue's color.
        curr_color = gtex_colors.tissue_color_rgb_normalized.values[gtex_colors.tissue_name == tissue][0]

        # Plot image on plot
        ab = AnnotationBbox(getImage(path, zoom=0.05), (x0, y0), frameon=True, bboxprops=dict(
            facecolor=curr_color, boxstyle='round', color=curr_color))
        ax.add_artist(ab)

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP, images")
    plt.tight_layout()
    # Save figure.
    plt.savefig("./out/umap_on_latents_imgs.png")
    plt.clf()
    plt.close()

    print("Done")


def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)


if __name__ == '__main__':
    main()
