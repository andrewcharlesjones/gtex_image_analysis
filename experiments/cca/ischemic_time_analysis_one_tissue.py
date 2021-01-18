import numpy as np
import pandas as pd
from os.path import join as pjoin
import socket
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


if socket.gethostname() == "andyjones":
    EXPRESSION_PATH = "/Users/andrewjones/Documents/beehive/multimodal_bio/organize_unnormalized_data/data/small_gene_tpm.gct"
    METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
    SAMPLE_METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    DATA_DIR = "./out"
else:
    EXPRESSION_PATH = "/tigress/aj13/gtexv8/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_tpm.gct"
    METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
    SAMPLE_METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    DATA_DIR = "/tigress/aj13/gtex_image_analysis/cca/pcca_out"


TISSUE_NAME = "Heart - Left Ventricle"
NUM_EXTREME_IMGS = 5

import matplotlib
font = {'size': 60}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)

def load_image(file, normalize=False):
    im = mpimg.imread(file)

    # Fourth channel is blank for some reason - remove it
    im = im[:, :, :3]

    # Mean subtract each channel
    if normalize:
        for curr_channel_num in range(im.shape[2]):
            curr_chan = im[:, :, curr_channel_num]
            im[:, :, curr_channel_num] = (
                curr_chan - np.mean(curr_chan)) / np.std(curr_chan)
    return im

def main():

    # Metadata
    v8_metadata = pd.read_table(METADATA_PATH)

    v8_sample_metadata = pd.read_table(SAMPLE_METADATA_PATH)
    v8_sample_metadata['sample_id'] = v8_sample_metadata.SAMPID.str.split("-").str[:3].str.join("-")
    v8_sample_metadata = v8_sample_metadata[~v8_sample_metadata.sample_id.duplicated(keep='first')]

    all_tissues = v8_sample_metadata.SMTSD.unique()
    all_tissues = all_tissues[all_tissues != "Kidney - Medulla"]


    # ---------------- Load data ----------------

    lv_names = pd.read_csv(pjoin(DATA_DIR, "img_fnames_pcca.csv"), index_col=0)
    lv_fnames = lv_names.fname.values
    lv_sample_ids = lv_names.index.values
    lv_subject_ids = np.array(["-".join(x.split("-")[:2]) for x in lv_sample_ids])
    subject_id_df = pd.DataFrame(lv_subject_ids, columns=['subject_id'])

    # Latent variables
    lvs_shared = pd.read_csv(pjoin(DATA_DIR, "pcca_lvs_shared.csv"), index_col=0).values.T
    n_lvs = lvs_shared.shape[1]


    # Sample IDs
    sample_ids = [os.path.basename(x).split(".")[0] for x in lv_fnames]
    sample_ids = [x[:-1] + str(int(x[-1]) + 1) for x in sample_ids]
    sample_id_df = pd.DataFrame(sample_ids, columns=['sample_id'])


    # Subset to this tissue's data
    curr_sample_metadata = v8_sample_metadata[v8_sample_metadata.SMTSD == TISSUE_NAME]
    tissue_sample_ids = curr_sample_metadata.sample_id.values

    shared_samples = np.intersect1d(tissue_sample_ids, sample_id_df.sample_id.values)

    shared_idx = np.where(np.isin(sample_id_df.sample_id.values, shared_samples))[0]
    curr_lvs_shared = lvs_shared[shared_idx]
    curr_fnames = lv_fnames[shared_idx]
    curr_sample_ids = sample_id_df.sample_id.values[shared_idx]
    curr_subject_ids = ['-'.join(x.split("-")[:2]) for x in curr_sample_ids]
    curr_metadata = v8_metadata[v8_metadata.SUBJID.isin(curr_subject_ids)]
    curr_metadata = curr_metadata[~curr_metadata.SUBJID.duplicated(keep='first')]

    # Make DF
    curr_lvs_shared_df = pd.DataFrame(curr_lvs_shared)
    curr_lvs_shared_df['subject_id'] = curr_subject_ids

    # Collapse any subjects that have multiple samples
    nondup_idx = ~curr_lvs_shared_df.subject_id.duplicated(keep='first')
    curr_lvs_shared_df = curr_lvs_shared_df.groupby("subject_id").mean()
    curr_fnames = curr_fnames[nondup_idx]
    assert curr_fnames.shape[0] == curr_lvs_shared_df.shape[0]

    # Merge metadata and LVs
    curr_data = pd.merge(curr_lvs_shared_df, curr_metadata[["SUBJID", "TRISCHD"]], left_index=True, right_on="SUBJID")
    curr_data = curr_data.set_index("SUBJID")

    # Get correlation between isch. time and each LV
    corr_mat = curr_data.corr()
    it_corrs = corr_mat["TRISCHD"].values[:-1]

    # Plot top-correlated LV
    top_idx = np.argmax(it_corrs)

    fig, ax = plt.subplots(figsize=(23, 20))
    ax.scatter(curr_data.iloc[:, top_idx], curr_data.TRISCHD.values)


    ## Scatter plot with points as images

    # Look over images, plotting each one at the correct location
    for ii in range(curr_data.shape[0]):

        x, y = curr_data.iloc[ii, top_idx], curr_data.TRISCHD.values[ii]
        path = curr_fnames[ii]

        # Plot image on plot
        ab = AnnotationBbox(getImage(path, zoom=0.05), (x, y), frameon=False)
        ax.add_artist(ab)

    sns.regplot(curr_data.iloc[:, top_idx], curr_data.TRISCHD.values)


    plt.xlabel("LV {}".format(top_idx + 1))
    plt.ylabel("Ischemic time (mins.)")
    plt.title(TISSUE_NAME + r', $\rho={}$'.format(round(it_corrs[top_idx], 2)))
    plt.tight_layout()
    # Save figure.
    plt.savefig("./out/ischemic_time/img_plots/{}_ischemic_time.png".format(TISSUE_NAME))
    plt.close()



    ## Plot images with highest and lowest ischemic times
    # sorted_idx = np.argsort(curr_data.TRISCHD.values)
    sorted_idx = np.argsort(curr_data.iloc[:, top_idx].values)
    high_it_fnames = curr_fnames[sorted_idx[-NUM_EXTREME_IMGS:]]
    low_it_fnames = curr_fnames[sorted_idx[:NUM_EXTREME_IMGS]]

    fig = plt.figure(figsize=(20, 10))

    for jj in range(NUM_EXTREME_IMGS):

        plt.subplot(2, NUM_EXTREME_IMGS, jj + 1)

        # Plot high IT image
        plt.imshow(load_image(high_it_fnames[jj]))
        plt.xticks([])
        plt.yticks([])

        if jj == 0:
            plt.ylabel("High value", rotation=0, labelpad=150)
            plt.tight_layout()
        plt.subplot(2, NUM_EXTREME_IMGS, jj + NUM_EXTREME_IMGS + 1)
        if jj == 0:
            plt.ylabel("Low value", rotation=0, labelpad=150)
            plt.tight_layout()

        # Plot low IT image
        plt.imshow(load_image(low_it_fnames[jj]))
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(TISSUE_NAME)
    plt.tight_layout()
    plt.savefig("./out/ischemic_time/img_plots/{}_ischemic_time_extreme_imgs.png".format(TISSUE_NAME))
    plt.close()
    
    

if __name__ == "__main__":
    main()
