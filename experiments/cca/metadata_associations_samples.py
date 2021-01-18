import numpy as np
import pandas as pd
from os.path import join as pjoin
import socket
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.patches as mpatches

if socket.gethostname() == "andyjones":
    EXPRESSION_PATH = "/Users/andrewjones/Documents/beehive/multimodal_bio/organize_unnormalized_data/data/small_gene_tpm.gct"
    METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    METADATA_ANNOT_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2015-01-12_Annotations_SampleAttributesDD.xlsx"
    DATA_DIR = "./out"
else:
    EXPRESSION_PATH = "/tigress/aj13/gtexv8/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_tpm.gct"
    METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    METADATA_ANNOT_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2015-01-12_Annotations_SampleAttributesDD.xlsx"
    DATA_DIR = "/tigress/aj13/gtex_image_analysis/cca/pcca_out"

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

MIN_NUM_SAMPLES = 20
NUM_VARS_TO_PLOT = 10

def main():

    # Metadata
    v8_metadata = pd.read_table(METADATA_PATH)
    v8_metadata['sample_id'] = ['-'.join(x.split("-")[:3]) for x in v8_metadata.SAMPID.values]

    metadata_col_annots = pd.read_excel(METADATA_ANNOT_PATH)

    # ---------------- Load data ----------------

    lv_names = pd.read_csv(pjoin(DATA_DIR, "img_fnames_pcca.csv"), index_col=0)
    lv_fnames = lv_names.fname.values
    lv_sample_ids = lv_names.index.values
    lv_subject_ids = np.array(["-".join(x.split("-")[:2]) for x in lv_sample_ids])

    # tissues = np.array([x.split("/")[-2] for x in lv_fnames])


    ##### Shared latent variables #####

    lvs_shared = pd.read_csv(pjoin(DATA_DIR, "pcca_img_lvs_exclusive.csv"), index_col=0).transpose()
    lvs_shared.index = lv_sample_ids

    n_cc_vars = lvs_shared.shape[1]
    lvs_shared.columns = ["CC{}".format(x+1) for x in np.arange(n_cc_vars)]
    all_data = pd.merge(v8_metadata, lvs_shared, left_on="sample_id", right_index=True)

    cc_varnames = ["CC" + str(ii + 1) for ii in range(n_cc_vars)]
    
    metadata_varnames = np.setdiff1d(all_data.columns, cc_varnames)
    is_numeric_cols = np.array([np.issubdtype(all_data[x].dtype, np.number) for x in metadata_varnames])
    metadata_varnames = metadata_varnames[is_numeric_cols]

    # Fit regression mdodel for each metadata variable, using CCs as covariates
    metadata_r2_scores = []
    metadata_varnames_regression = []
    col_annots = []
    for ii, curr_metadata_varname in enumerate(metadata_varnames):
        

        curr_metadata = all_data[curr_metadata_varname].values

        # Get rid of nans and numbers that code for missing data
        reported_idx = np.where(~np.isin(curr_metadata, [96, 97, 98, 99]))[0]
        not_na_idx = np.where(~np.isnan(curr_metadata))[0]
        curr_valid_idx = np.intersect1d(reported_idx, not_na_idx)


        X = all_data.iloc[curr_valid_idx][cc_varnames].values
        y = all_data.iloc[curr_valid_idx][curr_metadata_varname].values
        
        # Only run regression if there are a sufficient number of samples
        if X.shape[0] <= MIN_NUM_SAMPLES or len(np.unique(y)) == 1:
            continue

        # Fit model
        reg = LinearRegression().fit(X, y)

        # Compute R^2
        preds = reg.predict(X)
        curr_r2 = r2_score(y, preds)

        metadata_r2_scores.append(curr_r2)
        metadata_varnames_regression.append(curr_metadata_varname)

        curr_annot = metadata_col_annots.DOCFILE.values[metadata_col_annots.VARNAME == curr_metadata_varname][0]
        col_annots.append(curr_annot)

    metadata_r2_scores = np.array(metadata_r2_scores)
    metadata_varnames_regression = np.array(metadata_varnames_regression)

    r2_df = pd.DataFrame({"varname": metadata_varnames_regression, "r2": metadata_r2_scores, "category": col_annots})
    r2_df_top = r2_df.sort_values("r2", ascending=False).iloc[:NUM_VARS_TO_PLOT, :]

    # Get color of each bar
    cats_unique = np.unique(r2_df_top.category.values)
    n_colors = len(cats_unique)
    clrs = sns.color_palette('husl', n_colors=n_colors)    

    bar_colors = []
    
    for cat in r2_df_top.category.values:
        cat_idx = np.argwhere(cats_unique == cat)[0][0]
        curr_color = clrs[cat_idx]
        bar_colors.append(curr_color)

    legend_patches = []
    for cat in np.unique(r2_df_top.category.values):
        cat_idx = np.argwhere(cats_unique == cat)[0][0]
        curr_color = clrs[cat_idx]
        patch = mpatches.Patch(color=curr_color, label=cat)
        legend_patches.append(patch)
        

    plt.figure(figsize=(10, 8))
    plt.bar(np.arange(r2_df_top.shape[0]), r2_df_top.r2.values, color=bar_colors)
    plt.legend(handles=legend_patches, prop={'size': 20})
    plt.xticks(np.arange(r2_df_top.shape[0]), labels=r2_df_top.varname.values, rotation=-45, size=20, ha="left")
    plt.ylabel("$R^2$")
    plt.xlabel("")
    plt.title("Sample-level metadata")
    plt.tight_layout()
    plt.savefig("./out/metadata_associations_samples.png")
    plt.show()



if __name__ == "__main__":
    main()
