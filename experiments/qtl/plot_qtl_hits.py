import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as pjoin
import os
import socket 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from decimal import Decimal



if socket.gethostname() == "andyjones":
	QTL_DIR = "./out"
	DATA_DIR = "../../data_processing/qtl/data"
	PLOT_DIR = "./plots"
	IMG_DIR = "/Users/andrewjones/Documents/beehive/gtex_data_sample/images"
	PVAL_THRESHOLD = 0.5
else:
	QTL_DIR = "/tigress/aj13/gtex_image_analysis/qtl/out"
	DATA_DIR = "/tigress/aj13/gtex_image_analysis/qtl/data"
	PLOT_DIR = "/tigress/aj13/gtex_image_analysis/qtl/plots"
	IMG_DIR = "/tigress/BEE/GTExV8_dpcca/images"
	PVAL_THRESHOLD = 1e-8


import matplotlib
font = {'size': 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

def getImage(path, zoom=1):
	return OffsetImage(plt.imread(path), zoom=zoom)

def get_fname_from_subjid(img_data_dir, subjid):
	img_files = os.listdir(img_data_dir)
	curr_file = [x for x in img_files if subjid in x][0]
	return pjoin(img_data_dir, curr_file)


# Maps from qtl directory names to image directory names
tissue_name_map = {
	# "Adipose_Subcutaneous": "",
	# "Artery_Coronary": "CoronaryArtery",
	# "Artery_Tibial": "TibialArtery",
	# "Heart_Atrial_Appendage": "AtrialAppendage",
	# "Muscle_Skeletal": "SkeletalMuscle",
	# "Thyroid": "ThyroidGland",
	# "Adipose_Visceral_Omentum": "",
	# "Adrenal_Gland": "AdrenalGlands",
	# "Brain_Cortex": "BrainCortex",
	# "Breast_Mammary_Tissue": "MammaryTissueBreast",
	# "Minor_Salivary_Gland": "MinorSalivaryGlands"

	"Adipose_Subcutaneous": "AdiposeTissue",
	"Adipose_Visceral_Omentum": "Omentum",
	"Adrenal_Gland": "AdrenalGlands",
	"Artery_Aorta": "Aorta",
	"Artery_Coronary": "CoronaryArtery",
	"Artery_Tibial": "TibialArtery",
	"Brain_Cerebellum": "BrainCerebellum",
	"Brain_Cortex": "BrainCortex",
	"Breast_Mammary_Tissue": "MammaryTissueBreast",
	"Colon_Sigmoid": "SigmoidColon",
	"Colon_Transverse": "Colon",
	"Esophagus_Gastroesophageal_Junction": "GastroesophagealJunction",
	"Esophagus_Mucosa": "EsophagusMucosa",
	"Esophagus_Muscularis": "EsophagusMuscularis",
	"Heart_Atrial_Appendage": "Heart",
	"Heart_Left_Ventricle": "",
	"Kidney_Cortex": "KidneyCortex",
	"Liver": "Liver",
	"Lung": "Lung",
	"Minor_Salivary_Gland": "MinorSalivaryGlands",
	"Muscle_Skeletal": "SkeletalMuscle",
	"Nerve_Tibial": "TibialNerve",
	"Ovary": "Ovary",
	"Pancreas": "Pancreas",
	"Pituitary": "PituitaryGland",
	"Prostate": "Prostate",
	"Skin_Not_Sun_Exposed_Suprapubic": "SuprapubicSkin",
	"Skin_Sun_Exposed_Lower_leg": "Skin",
	"Small_Intestine_Terminal_Ileum": "Ileum",
	"Spleen": "Spleen",
	"Stomach": "Stomach",
	"Testis": "Testis",
	"Thyroid": "ThyroidGland",
	"Uterus": "Uterus",
	"Vagina": "Vagina",
}


# Get list of folders with QTLs
tissue_dirs = os.listdir(QTL_DIR)


# Plot significant QTLs for each tissue
for curr_tissue in tissue_dirs:

	print("Plotting {}...".format(curr_tissue))

	# All matrixEQTL output files for this tissue
	files = os.listdir(pjoin(QTL_DIR, curr_tissue))

	# Each file corresponds to one imageCCA component
	for curr_f in files:

		# Read QTL file
		qtl_output = pd.read_table(pjoin(QTL_DIR, curr_tissue, curr_f))

		# Filter by p-value
		qtl_output = qtl_output[qtl_output['p-value'] < PVAL_THRESHOLD]
		qtl_output = qtl_output.sort_values("p-value")

		# No hits for this component
		if qtl_output.shape[0] == 0:
			continue

		# Get component number as integer
		comp_num = int(curr_f.split("_")[-1].split(".")[0])

		# Get genotype and phenotype for each significant QTL
		for ii in range(qtl_output.shape[0])[:3]:
			
			curr_snp = qtl_output.SNP.values[ii]
			curr_pc = qtl_output.gene.values[ii]
			curr_pval = qtl_output['p-value'].values[ii]
			curr_pval = '%.2E' % Decimal(curr_pval)

			# Load original data files used to find this QTL
			curr_genotype_file = pjoin(DATA_DIR, "tissues", curr_tissue, "genotype_comp_{}.csv".format(comp_num))
			curr_phenotype_file = pjoin(DATA_DIR, "tissues", curr_tissue, "phenotype_comp_{}.csv".format(comp_num))

			curr_genotype_snps = pd.read_csv(curr_genotype_file, usecols=[0]).iloc[:, 0].values

			curr_phenotypes = pd.read_csv(curr_phenotype_file, index_col=0)
			# curr_phenotypes = curr_phenotypes.loc[curr_pc].values
			curr_phenotypes = pd.read_csv(curr_phenotype_file, index_col=0).transpose()[curr_pc]

			#  Just look at rows for this SNP
			idx_to_keep = np.where(curr_genotype_snps == curr_snp)[0] + 1
			idx_to_skip = np.setdiff1d(np.arange(curr_genotype_snps.shape[0]), idx_to_keep)
			rows_to_skip = np.delete(idx_to_skip, np.argwhere(idx_to_skip == 0))
			
			# Read in genotypes just for this SNP
			curr_genotypes = pd.read_csv(curr_genotype_file, skiprows=rows_to_skip, index_col=0)

			assert curr_genotypes.index.values[0] == curr_snp

			curr_genotypes = curr_genotypes.iloc[0, :]
			

			#### Boxplot of genotype and phenotype

			# Plot
			plt.figure(figsize=(10, 9))
			sns.boxplot(x=curr_genotypes, y=curr_phenotypes)
			plt.xlabel("Genotype\n(Minor allele frequency)")
			plt.ylabel("Image component {}".format(curr_pc))
			plt.title(curr_tissue.replace("_", " ") + "\n" + curr_snp.replace("_", "\_") + "\npval: {}".format(curr_pval), fontsize=20)
			plt.tight_layout()

			# Save
			save_dir = pjoin(PLOT_DIR, curr_tissue)
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			plt.savefig(pjoin(save_dir, "{}qtl_boxplot_{}_{}_comp{}.png".format(ii+1, curr_snp, curr_pc, comp_num)))
			plt.close()
			# import ipdb; ipdb.set_trace()


			### Overlay images on this type of plot

			subject_ids = pd.read_csv(curr_genotype_file).columns.values[1:]

			if curr_tissue not in tissue_name_map.keys():
				continue

			curr_img_tissue_name = tissue_name_map[curr_tissue]
			if curr_img_tissue_name == "":
				continue
			tissue_img_dir = pjoin(IMG_DIR, curr_img_tissue_name)

			# Get subject IDs that have both image and QTL data
			img_files = os.listdir(tissue_img_dir)
			subjs_with_data = np.intersect1d(['-'.join(x.split("-")[:2]) for x in img_files], subject_ids)

			# NUM_IMGS = min(subjs_with_data.shape[0], 200)
			NUM_IMGS = subjs_with_data.shape[0]

			rand_idx = np.random.choice(np.arange(subjs_with_data.shape[0]), replace=False, size=NUM_IMGS)
			subject_ids_to_plot = subjs_with_data[rand_idx]
			
			# Get filenames for these subjects
			fnames = np.array([get_fname_from_subjid(tissue_img_dir, x) for x in subject_ids_to_plot]) #[rand_idx]

			x, y = curr_genotypes[subject_ids_to_plot], curr_phenotypes[subject_ids_to_plot]

			# add jitter to x
			x += np.random.normal(0, 0.08, size=x.shape[0])

			fig, ax = plt.subplots(figsize=(10, 9))
			ax.scatter(x, y)

			# Look over images, plotting each one at the correct location
			font = {'size': 5}
			matplotlib.rc('font', **font)
			for x0, y0, path in zip(x, y, fnames):

				# Plot image on plot
				ab = AnnotationBbox(getImage(path, zoom=0.05), (x0, y0), frameon=True, bboxprops=dict(
					facecolor='black', boxstyle='round'))
				ax.add_artist(ab)

			font = {'size': 20}
			matplotlib.rc('font', **font)
			plt.xlabel("Genotype\n(Minor allele frequency)")
			plt.xticks([0, 1, 2], labels=[0, 1, 2])
			plt.ylabel("Image component {}".format(curr_pc))
			plt.title(curr_tissue.replace("_", " ") + "\n" + curr_snp.replace("_", "\_") + "\npval: {}".format(curr_pval), fontsize=20)
			plt.tight_layout()

			# Save figure.
			plt.savefig(pjoin(save_dir, "{}qtl_imageplot_{}_comp{}.png".format(ii+1, curr_snp, comp_num)))
			# plt.show()
			plt.close()
			# import ipdb; ipdb.set_trace()

