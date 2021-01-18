import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as pjoin
import socket
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import SVC

if socket.gethostname() == "andyjones":
	DATA_DIR = "../cca/out"
	METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
else:
	DATA_DIR = "/tigress/aj13/gtex_image_analysis/cca/pcca_out/"
	METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"


NUM_CV_FOLDS = 5

## ------ shared LVS -----
lv_names = pd.read_csv(pjoin(DATA_DIR, "img_fnames_pcca.csv"), index_col=0)
lv_fnames = lv_names.fname.values
lv_sample_ids = lv_names.index.values

tissues = np.array([x.split("/")[-2] for x in lv_fnames])

lvs_shared = pd.read_csv(pjoin(DATA_DIR, "pcca_lvs_shared.csv"), index_col=0).values.T
lvs_exclusive_img = pd.read_csv(pjoin(DATA_DIR, "pcca_img_lvs_exclusive.csv"), index_col=0).values.T
lvs_exclusive_exp = pd.read_csv(pjoin(DATA_DIR, "pcca_exp_lvs_exclusive.csv"), index_col=0).values.T


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

lvs_shared = lvs_shared[shared_idx1]
lvs_exclusive_img = lvs_exclusive_img[shared_idx1]
lvs_exclusive_exp = lvs_exclusive_exp[shared_idx1]
lv_fnames = lv_fnames[shared_idx1]

shared_idx2 = np.where(np.isin(v8_metadata.sample_id.values, shared_samples))[0]
v8_metadata = v8_metadata.iloc[shared_idx2, :]

sample_id_df = pd.merge(sample_id_df, v8_metadata[["sample_id", "SMTSD"]], on="sample_id", how='left')

lvs_shared = lvs_shared[sample_id_df.SMTSD != "Kidney - Medulla"]
lvs_exclusive_img = lvs_exclusive_img[sample_id_df.SMTSD != "Kidney - Medulla"]
lvs_exclusive_exp = lvs_exclusive_exp[sample_id_df.SMTSD != "Kidney - Medulla"]

lv_fnames = lv_fnames[sample_id_df.SMTSD != "Kidney - Medulla"]
sample_id_df = sample_id_df[sample_id_df.SMTSD != "Kidney - Medulla"]
tissues = sample_id_df.SMTSD.values

assert np.all(np.array([lvs_shared.shape[0], lvs_exclusive_img.shape[0], lvs_exclusive_exp.shape[0], lv_fnames.shape[0]]) == tissues.shape[0])

# Subset to tissues with a sufficient number of samples
tissue_counts = pd.Series(tissues).value_counts()
common_tissues = tissue_counts.index.values[tissue_counts >= 10]
common_tissue_idx = np.where(np.isin(tissues, common_tissues))[0]
# import ipdb; ipdb.set_trace()
lvs_shared = lvs_shared[common_tissue_idx, :]
lvs_exclusive_img = lvs_exclusive_img[common_tissue_idx, :]
lvs_exclusive_exp = lvs_exclusive_exp[common_tissue_idx, :]
tissues = tissues[common_tissue_idx]


##### Shared latent variables #####

# Fit classifier
clf = LogisticRegression(random_state=1, max_iter=1000, class_weight="balanced")
X_scaled = preprocessing.scale(lvs_shared)
cv_results = cross_validate(clf, X_scaled, tissues, cv=NUM_CV_FOLDS, verbose=0)
np.savetxt("./out/pcca_shared_vars_test_acc.csv", cv_results['test_score'], delimiter=",")
print(cv_results)


##### exclusive image LVS #####

# Fit classifier
clf = LogisticRegression(random_state=1, max_iter=1000, class_weight="balanced")

X_scaled = preprocessing.scale(lvs_exclusive_img)
cv_results = cross_validate(clf, X_scaled, tissues, cv=NUM_CV_FOLDS, verbose=0)
np.savetxt("./out/pcca_img_vars_test_acc.csv", cv_results['test_score'], delimiter=",")
print(cv_results)



##### exclusive expression LVS #####

# Fit classifier
clf = LogisticRegression(random_state=1, max_iter=1000, class_weight="balanced")

X_scaled = preprocessing.scale(lvs_exclusive_exp)
cv_results = cross_validate(clf, X_scaled, tissues, cv=NUM_CV_FOLDS, verbose=0)
np.savetxt("./out/pcca_exp_vars_test_acc.csv", cv_results['test_score'], delimiter=",")
print(cv_results)

import ipdb; ipdb.set_trace()


