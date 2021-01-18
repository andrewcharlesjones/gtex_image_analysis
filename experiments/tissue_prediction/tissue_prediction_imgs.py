import matplotlib.pyplot as plt
import socket
import pandas as pd
import numpy as np
from os.path import join as pjoin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os

if socket.gethostname() == "andyjones":
	DATA_DIR = "../autoencoder/out"
	METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
else:
	DATA_DIR = "/tigress/aj13/gtex_image_analysis/autoencoder/out/"
	METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

def main():

	# --- Load autoencoder data ---
	print("Loading data...")
	ae_latents = np.load(pjoin(DATA_DIR, "latent_z.npy"), allow_pickle=True)
	# tissues = np.load(pjoin(DATA_DIR, "tissue_labels.npy"), allow_pickle=True)
	im_fnames = np.load(pjoin(DATA_DIR, "im_fnames.npy"), allow_pickle=True)

	# Get corresponding tissues
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
	ae_latents = ae_latents[shared_idx1]
	im_fnames = im_fnames[shared_idx1]

	shared_idx2 = np.where(np.isin(v8_metadata.sample_id.values, shared_samples))[0]
	v8_metadata = v8_metadata.iloc[shared_idx2, :]

	sample_id_df = pd.merge(sample_id_df, v8_metadata[["sample_id", "SMTSD"]], on="sample_id", how='left')
	ae_latents = ae_latents[sample_id_df.SMTSD != "Kidney - Medulla"]
	im_fnames = im_fnames[sample_id_df.SMTSD != "Kidney - Medulla"]
	sample_id_df = sample_id_df[sample_id_df.SMTSD != "Kidney - Medulla"]
	tissues = sample_id_df.SMTSD.values

	X_scaled = preprocessing.scale(ae_latents)
	print("Loaded {} samples".format(X_scaled.shape[0]))
	


	# Fit to get confusion matrix
	clf = MLPClassifier(random_state=1, max_iter=300)
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, tissues, test_size=0.25, random_state=42) #, stratify=tissues)
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)


	# confusion matrix
	tissues_unique = np.unique(y_test)
	confusion_mat = confusion_matrix(y_test, preds, labels=tissues_unique)
	row_sums = confusion_mat.sum(axis=1)
	confusion_mat = confusion_mat / row_sums[:, np.newaxis]
	
	plt.figure(figsize=(16, 14))
	sns.heatmap(confusion_mat, 
		xticklabels=tissues_unique, 
		yticklabels=tissues_unique)
	plt.xticks(size=15)
	plt.yticks(size=15)
	plt.title("Tissue prediction, images")
	plt.xlabel("Predicted tissue")
	plt.ylabel("True tissue")
	plt.tight_layout()
	plt.savefig("./out/confusion_matrix_images.png")

	# import ipdb; ipdb.set_trace()
	
	# --- Fit linear model ---
	print("Fitting model...")
	clf = LogisticRegression(random_state=0)
	cv_results = cross_validate(clf, X_scaled, tissues, cv=5, verbose=0)
	print(cv_results)
	np.savetxt("./out/im_ae_latents_test_acc.csv", cv_results['test_score'], delimiter=",")

if __name__ == "__main__":
	main()