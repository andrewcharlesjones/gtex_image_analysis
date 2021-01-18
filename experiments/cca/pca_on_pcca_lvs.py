import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import socket
from os.path import join as pjoin
import os

import matplotlib
font = {'size': 60}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

NUM_IMS_TO_PLOT = 2000

def getImage(path, zoom=1):
	return OffsetImage(plt.imread(path), zoom=zoom)


if socket.gethostname() == "andyjones":
	DATA_DIR = "./out"
	METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
	GTEX_COLORS_PATH = "../../data/colors/tissue_gtex_colors.tsv"
else:
	DATA_DIR = "/tigress/aj13/gtex_image_analysis/cca/pcca_out"
	METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
	GTEX_COLORS_PATH = "/projects/BEE/RNAseq/RNAseq_dev/Analysis/tissue_gtex_colors.tsv"


lv_names = pd.read_csv(pjoin(DATA_DIR, "img_fnames_pcca.csv"), index_col=0)
lv_fnames = lv_names.fname.values
lv_sample_ids = lv_names.index.values

lvs_shared = pd.read_csv(pjoin(DATA_DIR, "pcca_lvs_shared.csv"), index_col=0).values.T
lvs_exclusive_img = pd.read_csv(pjoin(DATA_DIR, "pcca_img_lvs_exclusive.csv"), index_col=0).values.T
lvs_exclusive_exp = pd.read_csv(pjoin(DATA_DIR, "pcca_exp_lvs_exclusive.csv"), index_col=0).values.T

# tissues = np.array([x.split("/")[-2] for x in lv_fnames])


# Get corresponding tissues
sample_ids = [os.path.basename(x).split(".")[0] for x in lv_fnames]
sample_ids = [x[:-1] + str(int(x[-1]) + 1) for x in sample_ids]
sample_id_df = pd.DataFrame(sample_ids, columns=['sample_id'])

# Metadata
v8_metadata = pd.read_table(METADATA_PATH)
v8_metadata['sample_id'] = v8_metadata.SAMPID.str.split("-").str[:3].str.join("-")
v8_metadata = v8_metadata[~v8_metadata.sample_id.duplicated(keep='first')]


shared_samples = np.intersect1d(v8_metadata.sample_id.values, sample_id_df.sample_id.values)

shared_idx1 = np.where(np.isin(sample_id_df.sample_id.values, shared_samples))[0]
sample_id_df = sample_id_df.iloc[shared_idx1, :]
# latent_z = latent_z[shared_idx1]
lvs_shared = lvs_shared[shared_idx1]
lvs_exclusive_img = lvs_exclusive_img[shared_idx1]
lvs_exclusive_exp = lvs_exclusive_exp[shared_idx1]
lv_fnames = lv_fnames[shared_idx1]

shared_idx2 = np.where(np.isin(v8_metadata.sample_id.values, shared_samples))[0]
v8_metadata = v8_metadata.iloc[shared_idx2, :]

sample_id_df = pd.merge(sample_id_df, v8_metadata[["sample_id", "SMTSD"]], on="sample_id", how='left')
# latent_z = latent_z[sample_id_df.SMTSD != "Kidney - Medulla"]
lvs_shared = lvs_shared[sample_id_df.SMTSD != "Kidney - Medulla"]
lvs_exclusive_img = lvs_exclusive_img[sample_id_df.SMTSD != "Kidney - Medulla"]
lvs_exclusive_exp = lvs_exclusive_exp[sample_id_df.SMTSD != "Kidney - Medulla"]

lv_fnames = lv_fnames[sample_id_df.SMTSD != "Kidney - Medulla"]
sample_id_df = sample_id_df[sample_id_df.SMTSD != "Kidney - Medulla"]
tissues = sample_id_df.SMTSD.values

assert np.all(np.array([lvs_shared.shape[0], lvs_exclusive_img.shape[0], lvs_exclusive_exp.shape[0], lv_fnames.shape[0]]) == tissues.shape[0])


n = lv_fnames.shape[0]

num_ims_to_plot = min(n, NUM_IMS_TO_PLOT)
if num_ims_to_plot is None:
	rand_idx = np.arange(n)
else:
	rand_idx = np.random.choice(
		np.arange(n), size=num_ims_to_plot, replace=False)

fnames_to_plot = lv_fnames[rand_idx]
tissues_to_plot = tissues[rand_idx]

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

##### Shared latent variables #####

lvs_shared_reduced = PCA().fit_transform(lvs_shared)
pca_x, pca_y = lvs_shared_reduced[rand_idx, 0], lvs_shared_reduced[rand_idx, 1]

fig, ax = plt.subplots(figsize=(23, 20))
ax.scatter(pca_x, pca_y)

font = {'size': 15}
matplotlib.rc('font', **font)

# Look over images, plotting each one at the correct location
for x0, y0, path, tissue in zip(pca_x, pca_y, fnames_to_plot, tissues_to_plot):

	# Get tissue's color.
	curr_color = gtex_colors.tissue_color_rgb_normalized.values[gtex_colors.tissue_name == tissue][0]

	# Plot image on plot
	ab = AnnotationBbox(getImage(path, zoom=0.05), (x0, y0), frameon=True, bboxprops=dict(facecolor=curr_color, boxstyle='round', color=curr_color))
	ax.add_artist(ab)

font = {'size': 60}
matplotlib.rc('font', **font)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Shared variation")
plt.tight_layout()
plt.savefig("./out/pcca_pca_lvs_shared.png")
plt.close()

# import ipdb; ipdb.set_trace()


##### Image-specific latent variables #####


lvs_shared_reduced = PCA().fit_transform(lvs_exclusive_img)
pca_x, pca_y = lvs_shared_reduced[rand_idx, 0], lvs_shared_reduced[rand_idx, 1]

fig, ax = plt.subplots(figsize=(23, 20))
ax.scatter(pca_x, pca_y)

font = {'size': 15}
matplotlib.rc('font', **font)

# Look over images, plotting each one at the correct location
for x0, y0, path, tissue in zip(pca_x, pca_y, fnames_to_plot, tissues_to_plot):

	# Get tissue's color.
	curr_color = gtex_colors.tissue_color_rgb_normalized.values[gtex_colors.tissue_name == tissue][0]

	# Plot image on plot
	ab = AnnotationBbox(getImage(path, zoom=0.05), (x0, y0), frameon=True, bboxprops=dict(facecolor=curr_color, boxstyle='round', color=curr_color))
	ax.add_artist(ab)


font = {'size': 60}
matplotlib.rc('font', **font)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Image-specific variation")
plt.tight_layout()
plt.savefig("./out/pcca_pca_lvs_exclusive_img.png")
plt.close()


##### Gene-specific latent variables #####



lvs_shared_reduced = PCA().fit_transform(lvs_exclusive_exp)
pca_x, pca_y = lvs_shared_reduced[rand_idx, 0], lvs_shared_reduced[rand_idx, 1]

fig, ax = plt.subplots(figsize=(23, 20))
ax.scatter(pca_x, pca_y)

font = {'size': 15}
matplotlib.rc('font', **font)

# Look over images, plotting each one at the correct location
for x0, y0, path, tissue in zip(pca_x, pca_y, fnames_to_plot, tissues_to_plot):

	# Get tissue's color.
	curr_color = gtex_colors.tissue_color_rgb_normalized.values[gtex_colors.tissue_name == tissue][0]

	# Plot image on plot
	ab = AnnotationBbox(getImage(path, zoom=0.05), (x0, y0), frameon=True, bboxprops=dict(facecolor=curr_color, boxstyle='round', color=curr_color))
	ax.add_artist(ab)


font = {'size': 60}
matplotlib.rc('font', **font)


plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Gene-specific variation")
plt.tight_layout()
plt.savefig("./out/pcca_pca_lvs_exclusive_gene.png")
plt.close()

print("Done.")
# import ipdb; ipdb.set_trace()

