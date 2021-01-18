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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

if socket.gethostname() == "andyjones":
    EXPRESSION_PATH = "/Users/andrewjones/Documents/beehive/multimodal_bio/organize_unnormalized_data/data/small_gene_tpm.gct"
    METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
    SAMPLE_METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    DATA_DIR = "./out"
    GTEX_COLORS_PATH = "../../data/colors/tissue_gtex_colors.tsv"
else:
    EXPRESSION_PATH = "/tigress/aj13/gtexv8/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_tpm.gct"
    METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
    SAMPLE_METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    DATA_DIR = "/tigress/aj13/gtex_image_analysis/cca/pcca_out"
    GTEX_COLORS_PATH = "/projects/BEE/RNAseq/RNAseq_dev/Analysis/tissue_gtex_colors.tsv"


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

    r2_scores = []
    tissue_names_r2 = []
    for curr_tissue in all_tissues[1:]:

        curr_sample_metadata = v8_sample_metadata[v8_sample_metadata.SMTSD == curr_tissue]
        tissue_sample_ids = curr_sample_metadata.sample_id.values

        shared_samples = np.intersect1d(tissue_sample_ids, sample_id_df.sample_id.values)
        if shared_samples.shape[0] == 0:
            continue

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
        plt.scatter(curr_data.iloc[:, top_idx], curr_data.TRISCHD.values)
        plt.title(curr_tissue + ", {}".format(round(it_corrs[top_idx], 3)))
        plt.savefig(pjoin("out/ischemic_time/tissue_scatters", curr_tissue + ".png"))
        plt.close()

        # Get R2 for association with ischemic time
        X = curr_data.values[:, :n_lvs]
        y = curr_data.values[:, -1]

        if X.shape[0] < n_lvs or np.var(y) == 0:
            continue

        if curr_tissue == "Bladder":
            import ipdb; ipdb.set_trace()

        reg = LinearRegression().fit(X, y)

        # Compute R^2
        preds = reg.predict(X)
        curr_r2 = r2_score(y, preds)
        r2_scores.append(curr_r2)
        tissue_names_r2.append(curr_tissue)

    r2_df = pd.DataFrame({"tissue": tissue_names_r2, "r2": r2_scores})
    

    # Load GTEx colors
    gtex_colors = pd.read_table(GTEX_COLORS_PATH)

    color_lists = gtex_colors.tissue_color_rgb.str.split(",").values
    reds = [int(x[0]) / 255. for x in color_lists]
    greens = [int(x[1]) / 255. for x in color_lists]
    blues = [int(x[2]) / 255. for x in color_lists]
    zipped_colors = [np.array([reds[ii], greens[ii], blues[ii]]) for ii in range(color_lists.shape[0])]
    gtex_colors['tissue_color_rgb_normalized'] = zipped_colors


    r2_df = pd.merge(r2_df, gtex_colors[['tissue_name', 'tissue_color_rgb_normalized']], left_on='tissue', right_on='tissue_name', how='left')
    r2_df = r2_df.sort_values("r2", ascending=False)

    import matplotlib
    font = {'size': 60}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['text.usetex'] = True
    plt.figure(figsize=(35, 20))
    plt.bar(np.arange(r2_df.shape[0]), r2_df.r2.values, color=r2_df.tissue_color_rgb_normalized.values)
    plt.xticks(np.arange(r2_df.shape[0]), r2_df.tissue.values, size=50, rotation=90)
    plt.ylabel("$R^2$")
    plt.xlabel("")
    plt.title("Ischemic time association")
    plt.tight_layout()
    plt.savefig(pjoin("out/tissue_ischemic_time_barplot.png"))
    # plt.show()
    
    

if __name__ == "__main__":
    main()
