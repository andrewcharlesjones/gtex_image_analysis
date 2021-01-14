import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join as pjoin
import socket
import pandas as pd
import seaborn as sns


if socket.gethostname() == "andyjones":
    DATA_DIR = "/Users/andrewjones/Documents/beehive/gtex_data_sample"
    METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    EXPRESSION_PATH = "/Users/andrewjones/Documents/beehive/multimodal_bio/organize_unnormalized_data/data/small_gene_tpm.gct"
    GTEX_COLORS_PATH = "../../data/colors/tissue_gtex_colors.tsv"
else:
    DATA_DIR = "/projects/BEE/GTExV8_dpcca"
    METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    EXPRESSION_PATH = "/tigress/aj13/gtexv8/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_tpm.gct"
    GTEX_COLORS_PATH = "/projects/BEE/RNAseq/RNAseq_dev/Analysis/tissue_gtex_colors.tsv"


import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

# ------ Get number of paired samples in v8 --------

# Metadata
v8_metadata = pd.read_table(METADATA_PATH)
v8_metadata['donor_id'] = ['-'.join(x.split("-")[:2])
                           for x in v8_metadata.SAMPID.values]
v8_metadata['sample_id'] = ['-'.join(x.split("-")[:3])
                            for x in v8_metadata.SAMPID.values]


# Load expression sample names
exp_data_header = pd.read_table(EXPRESSION_PATH, skiprows=2, nrows=1)
exp_full_ids = exp_data_header.columns.values[2:]
sample_ids_exp = ['-'.join(x.split("-")[:3]) for x in exp_full_ids]
donor_ids_exp = ['-'.join(x.split("-")[:2]) for x in exp_full_ids]


# Load image sample names
im_dir = pjoin(DATA_DIR, "images")
tissues = os.listdir(im_dir)
tissues = [x for x in tissues if (x != "README") and (x != ".DS_Store")]
sample_ids_imgs = []
donor_ids_imgs = []
tissue_counts = {}
for tiss in tissues:
    curr_files = os.listdir(os.path.join(im_dir, tiss))
    curr_samp_ids = [x.split(".")[0] for x in curr_files]
    # Need to add one to align with expression samples
    samps = []
    for samp in curr_samp_ids:
        try:
            new_endnum = str(int(samp[-1]) + 1)
        except:
            continue
        newsamp = samp[:-1] + new_endnum

        samps.append(newsamp)
    sample_ids_imgs.append(samps)
    donor_ids_imgs.append(['-'.join(x.split("-")[:2]) for x in samps])
    tissue_counts[tiss] = len(np.unique(samps))

sample_ids_imgs = np.concatenate(sample_ids_imgs)
donor_ids_imgs = np.concatenate(donor_ids_imgs)


# Get shared sample and donor IDs
sample_ids_shared = np.unique(np.intersect1d(sample_ids_exp, sample_ids_imgs))
donor_ids_shared = np.unique(np.intersect1d(donor_ids_exp, donor_ids_imgs))


# Get number of tissues
tissues_shared = v8_metadata[v8_metadata.sample_id.isin(
    sample_ids_shared)].SMTSD.unique()


# These numbers were manually taken from v6 image paper
v6_num_samples = 2221
v6_num_tissues = 29

v8_num_samples = len(sample_ids_shared)
v8_num_tissues = len(tissues_shared)



plt.figure(figsize=(16, 5))

plt.subplot(121)
plt.bar(np.arange(2), [v6_num_samples, v8_num_samples])
plt.xticks(np.arange(2), ["v6", "v8"])
plt.ylabel("Sample count")

plt.subplot(122)
plt.bar(np.arange(2), [v6_num_tissues, v8_num_tissues])
plt.xticks(np.arange(2), ["v6", "v8"])
plt.ylabel("Tissue count")

plt.suptitle("Paired expression/image samples")

plt.savefig("./out/versions_comparison.png")
plt.close()


# ------- Get breakdown of samples by tissue -------

shared_metadata = v8_metadata[v8_metadata.sample_id.isin(sample_ids_shared)]
shared_metadata = shared_metadata.drop_duplicates(subset=["sample_id"])

shared_metadata = shared_metadata[shared_metadata.SMTSD != "Kidney - Medulla"]
print("Num shared samples: {}".format(len(sample_ids_shared)))

unique_tissues, tissue_counts = np.unique(shared_metadata.SMTSD, return_counts=True)


# Load GTEx colors
gtex_colors = pd.read_table(GTEX_COLORS_PATH)

color_lists = gtex_colors.tissue_color_rgb.str.split(",").values
reds = [int(x[0]) / 255. for x in color_lists]
greens = [int(x[1]) / 255. for x in color_lists]
blues = [int(x[2]) / 255. for x in color_lists]
zipped_colors = [np.array([reds[ii], greens[ii], blues[ii]]) for ii in range(color_lists.shape[0])]
gtex_colors['tissue_color_rgb_normalized'] = zipped_colors

color_df = pd.DataFrame(unique_tissues, columns=['tissue'])
color_df['counts'] = tissue_counts
color_df = pd.merge(color_df, gtex_colors[['tissue_name', 'tissue_color_rgb_normalized']], left_on='tissue', right_on='tissue_name', how='left')
color_df = color_df.sort_values("counts", ascending=False)
# import ipdb; ipdb.set_trace()

# color_palette = dict(zip(color_df.tissue, color_df.tissue_color_rgb_normalized))

plt.figure(figsize=(14, 9))
plt.bar(np.arange(color_df.shape[0]), color_df.counts.values, color=color_df.tissue_color_rgb_normalized.values)
plt.xticks(np.arange(color_df.shape[0]), color_df.tissue.values, size=20)
# sns.countplot(data=color_df, x="tissue", order=color_df['tissue'].value_counts().index, color=color_df.tissue_color_rgb_normalized.values) #hue="tissue", palette=color_palette)
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("Count")
plt.title("Paired expression/image samples in GTEx v8")
plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig("./out/tissue_counts_shared_v8.png")

plt.close()

