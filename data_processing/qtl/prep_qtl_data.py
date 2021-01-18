import numpy as np
import pandas as pd
from os.path import join as pjoin
import socket
import os
from sklearn.decomposition import PCA


if socket.gethostname() == "andyjones":
    CCA_DIR = "/Users/andrewjones/Documents/beehive/gtex_image_analysis/experiments/imagecca/out"
    IMG_DATA_DIR = "/Users/andrewjones/Documents/beehive/gtex_image_analysis/data/imagecca"
    METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    GENOTYPE_PATH = "/Users/andrewjones/Documents/beehive/gtex_data_sample/genotypes/thyroid_genotype_small.tsv"
    GENOTYPE_DIR = "/Users/andrewjones/Documents/beehive/gtex_data_sample/genotypes"
    GENE_NAME_DIR = "/Users/andrewjones/Documents/beehive/multimodal_bio/qtl/data/gene_snp_names"
    EQTL_DIR = "./data"
    SAVE_DIR = "./data"
else:
    CCA_DIR = "/tigress/aj13/gtex_image_analysis/imagecca/out"
    IMG_DATA_DIR = "/tigress/aj13/gtex_image_analysis/imagecca/data"
    METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    GENOTYPE_DIR = "/tigress/BEE/RNAseq/RNAseq_dev/Data/Genotype/gtex_v8/dosage"
    GENE_NAME_DIR = "/scratch/gpfs/aj13/qtl/data/gene_snp_names"
    EQTL_DIR = "/tigress/aj13/gtexv8/eqtls/GTEx_Analysis_v8_eQTL"
    SAVE_DIR = "/tigress/aj13/gtex_image_analysis/qtl/data"

# Metadata
v8_metadata = pd.read_table(METADATA_PATH)
v8_metadata['sample_id'] = ['-'.join(x.split("-")[:3]) for x in v8_metadata.SAMPID.values]
# import ipdb; ipdb.set_trace()

# Load ImageCCA coefficients
imagecca_coeffs = pd.read_table(pjoin(CCA_DIR, "imagecca_coeffs.txt"))

# Load ImageCCA latent variables
imagecca_img_lvs = pd.read_csv(pjoin(CCA_DIR, "imagecca_image_vars.txt"), index_col=0)


img_data = pd.read_csv(pjoin(IMG_DATA_DIR, "img_data_for_cca.csv"), index_col=0)


# Separate gene and image coefficients from ImageCCA
imagecca_coeffs_gene = imagecca_coeffs[imagecca_coeffs.type == "gene"]
imagecca_coeffs_image = imagecca_coeffs[imagecca_coeffs.type == "image"]
component_nums = np.sort(imagecca_coeffs_gene.CCA_var.unique())

# Get all unique tissues
metadata_tissues = v8_metadata.SMTSD.unique()
metadata_tissues = metadata_tissues[~pd.isna(metadata_tissues)]

# Get list of genotype files available
genotype_files = os.listdir(GENOTYPE_DIR)
genotype_tissues = np.array([x.split(".")[1] for x in genotype_files if x.endswith(".tsv")])

# Get list of files containing QTL hits
eqtl_files = os.listdir(EQTL_DIR)

# For each tissue, put together data
for curr_tissue in metadata_tissues:
# for curr_tissue in ['Thyroid']:

    print("Loading {}...".format(curr_tissue))

    # Make tissue name match the QTL filenames
    curr_genotype_tissue = '_'.join(curr_tissue.replace(' - ', '_').replace('(', '').replace(')', '').split(" "))

    save_dir_tissue = pjoin(SAVE_DIR, "tissues", curr_genotype_tissue.strip())

    # Skip if we've already processed this tissue
    if os.path.isdir(save_dir_tissue):
        continue

    # Get sample IDs for this tissue
    curr_tissue_sample_ids = v8_metadata.sample_id.values[v8_metadata.SMTSD.values == curr_tissue]
    imagecca_lvs_curr_tissue = imagecca_img_lvs[imagecca_img_lvs.index.isin(
        curr_tissue_sample_ids)]

    # Get the image data for this tissue
    img_data_curr_tissue = img_data[img_data.index.isin(curr_tissue_sample_ids)]
    imagecca_lvs_curr_tissue = (imagecca_lvs_curr_tissue - imagecca_lvs_curr_tissue.mean()) / imagecca_lvs_curr_tissue.std()

    if imagecca_lvs_curr_tissue.shape[0] == 0:
        print("No latent variables")
        continue



    # Get list 
    continue_flag = False
    if socket.gethostname() == "andyjones":
        genotype_file_path = "/Users/andrewjones/Documents/beehive/gtex_data_sample/genotypes/thyroid_genotype_small.tsv"
    else:
        # genotype_file_path = "/tigress/aj13/gtexv8/genotypes/small/thyroid_genotype_small.tsv"
        genotype_file_path = [x for x in genotype_files if (curr_genotype_tissue in x) & ('final' in x)]

        
        # if len(genotype_file_path) == 0:
        #     continue_flag = True

    print(genotype_file_path)
    if len(genotype_file_path) == 0:
        continue
    if socket.gethostname() != "andyjones":
        genotype_file_path = pjoin(GENOTYPE_DIR, genotype_file_path[0])

    # if (~continue_flag) and (socket.gethostname() != "andyjones"):
    #     genotype_file_path = pjoin(GENOTYPE_DIR, genotype_file_path[0])
    # elif socket.gethostname() != "andyjones":
    #     continue
    
    
    
    eqtl_file_path = [x for x in eqtl_files if (curr_genotype_tissue in x) & ("signif_variant_gene_pairs" in x)][0]
    eqtl_file_path = pjoin(EQTL_DIR, eqtl_file_path)

    # print(genotype_file_path)
    # print(eqtl_file_path)

    # Load significant QTLs for this tissue
    curr_tissue_eqtls = pd.read_table(eqtl_file_path)

    # Load SNPs with genotype data for this tissue
    genotype_snp_df = pd.read_table(genotype_file_path, usecols=["constVarID"])

    # Only take samples that are in the imageCCA output
    imagecca_subject_ids = img_data_curr_tissue.index.values
    imagecca_subject_ids = np.array(['-'.join(x.split("-")[:2]) for x in imagecca_subject_ids])

    genotype_subject_ids = pd.read_table(genotype_file_path, nrows=0).columns.values

    # Subjects with both ImageCCA variables and genotype data
    shared_subject_ids = np.intersect1d(imagecca_subject_ids, genotype_subject_ids)



    # For each ImageCCA component:
    # 1. Find the intersection between nonzero genes in that ImageCCA component and QTL hits in that gene
    phenotype_list = []
    for component_ii in component_nums:

        # Get genes with nonzero ImageCCA coefficients
        curr_nonzero_genes = imagecca_coeffs_gene[imagecca_coeffs_gene.CCA_var == component_ii].name.values

        # Get image features with nonzero ImageCCA coefficients
        curr_nonzero_img_features = imagecca_coeffs_image[imagecca_coeffs_image.CCA_var == component_ii].name.values

        ########## Load genotype data ##########

        # Get significant eQTL SNPs in this tissue
        inset_idx = np.where(curr_tissue_eqtls.gene_id.isin(curr_nonzero_genes) == True)[0]
        curr_snp_ids = np.unique(curr_tissue_eqtls.variant_id.values[inset_idx])
        qtl_snp_df = pd.DataFrame({'qtl_snps': curr_snp_ids})

        # Join QTL hit data and subject-specific genotype data
        shared_snps_df = genotype_snp_df.merge(qtl_snp_df, right_on="qtl_snps", left_on="constVarID", how='left')
        # import ipdb; ipdb.set_trace()
        assert shared_snps_df.shape[0] == genotype_snp_df.shape[0]
        assert np.array_equal(genotype_snp_df.constVarID.values, shared_snps_df.constVarID.values)

        # Get indices where QTL snps had a match
        idx_to_keep = np.where(~shared_snps_df.qtl_snps.isna())[0] + 1

        # If no QTLs, skip this component
        if idx_to_keep.shape[0] == 0:
            continue

        # Indices of SNPs to skip when reading the genotype data
        idx_to_skip = np.setdiff1d(np.arange(genotype_snp_df.shape[0]), idx_to_keep)

        # Which subjects to get genotype data for
        cols_to_read = np.append(shared_subject_ids, "constVarID")

        # Load the genotype data
        rows_to_skip = np.delete(idx_to_skip, np.argwhere(idx_to_skip == 0))
        curr_genotypes = pd.read_table(genotype_file_path, index_col="constVarID", skiprows=rows_to_skip, usecols=cols_to_read)

        # Drop last row (weird pandas thing)
        curr_genotypes = curr_genotypes.iloc[:-1, :]
        assert np.all(curr_genotypes.index.isin(qtl_snp_df.qtl_snps) == True)

        
        ########## Load phenotype data ##########

        # Drop subjects with multiple samples
        img_data_curr_tissue.index = imagecca_subject_ids
        img_data_curr_tissue = img_data_curr_tissue[~img_data_curr_tissue.index.duplicated(keep='first')]

        # Take image data for the subjects with genotype data
        # import ipdb; ipdb.set_trace()
        curr_nonzero_img_features_idx = [int(x[1:]) for x in curr_nonzero_img_features]
        curr_phenotypes = img_data_curr_tissue.iloc[:, curr_nonzero_img_features_idx].transpose()[shared_subject_ids]

        # Make sure subject IDs match for genotype and phenotype data
        assert np.array_equal(curr_phenotypes.columns.values, curr_genotypes.columns.values)
        
        # import ipdb; ipdb.set_trace()
        ########## Save ##########

        if not os.path.exists(save_dir_tissue):
            os.makedirs(save_dir_tissue)
        curr_phenotypes.to_csv(pjoin(save_dir_tissue, "phenotype_comp_{}.csv".format(component_ii)))
        curr_genotypes.to_csv(pjoin(save_dir_tissue, "genotype_comp_{}.csv".format(component_ii)))


        save_dir_eqtls = pjoin(SAVE_DIR, "eqtl_data", curr_genotype_tissue.strip())
        if not os.path.exists(save_dir_eqtls):
            os.makedirs(save_dir_eqtls)
        curr_eqtl_data = curr_tissue_eqtls[curr_tissue_eqtls.variant_id.isin(curr_genotypes.index.values)]
        curr_eqtl_data.to_csv(pjoin(save_dir_eqtls, "eqtls_comp_{}.csv".format(component_ii)))



