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

if socket.gethostname() == "andyjones":
	EXPRESSION_PATH = "/Users/andrewjones/Documents/beehive/multimodal_bio/organize_unnormalized_data/data/small_gene_tpm.gct"
	METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
else:
	EXPRESSION_PATH = "/tigress/aj13/gtexv8/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_tpm.gct"
	METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"

NUM_GENES = 2000

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True


def main():

	# ---------------- Load data ----------------

	# Metadata
	v8_metadata = pd.read_table(
		METADATA_PATH)
	v8_metadata['sample_id'] = [
		'-'.join(x.split("-")[:3]) for x in v8_metadata.SAMPID.values]
	v8_metadata = v8_metadata.drop_duplicates("sample_id")

	# ------- Load expression ---------
	# Read in expression file
	expression_data = pd.read_table(EXPRESSION_PATH, skiprows=2, index_col=0)

	# Ignore transcript ID column
	expression_data = expression_data.drop(labels='transcript_id(s)', axis=1)

	# Make samples on rows
	expression_data = expression_data.transpose()

	# Replace NaNs
	expression_data = expression_data.fillna(0.0)

	# Remove any genes that show no expression
	expression_data = expression_data.iloc[:, (expression_data.sum(axis=0) > 0).values]

	# Get variances of each gene
	gene_vars = expression_data.var()

	# Get genes with highest variances
	gene_vars = gene_vars.sort_values(ascending=False)  # sort

	top_genes = gene_vars.index.values
	if NUM_GENES is not None:
		top_genes = top_genes[:NUM_GENES]  # subset

	# Subset data to highest variance genes
	expression_data = expression_data[top_genes]

	# Make samples on rows
	gene_names = expression_data.columns.values

	# Shorted sample IDs to match IDs from images
	expression_data.index = ["-".join(x.split("-")[:3]) for x in expression_data.index.values]

	expression_data = expression_data.loc[~expression_data.index.duplicated(keep='first')]


	shared_data = expression_data.merge(v8_metadata[["sample_id", "SMTSD"]], left_index=True, right_on="sample_id")

	assert np.all(np.unique(shared_data.index.values, return_counts=True)[1] == 1)
	assert shared_data.shape[0] == expression_data.shape[0]

	# shared_data.to_csv("./out/expression_data.csv")

	# shared_data = pd.read_csv("./out/expression_data.csv", index_col=0)

	# Scale data
	data = np.log(expression_data[gene_names].values + 1)
	X_scaled = preprocessing.scale(data)
	tissues = shared_data.SMTSD.values

	tissue_names, tissue_counts = np.unique(tissues, return_counts=True)
	abundant_tissues = tissue_names[tissue_counts >= 10]
	abundant_tissues_idx = np.where(np.isin(tissues, abundant_tissues))[0]
	X_scaled, tissues = X_scaled[abundant_tissues_idx, :], tissues[abundant_tissues_idx]

	print("Loaded {} samples".format(X_scaled.shape[0]))

	# Fit MLP for confusion matrix
	clf = MLPClassifier(random_state=1, max_iter=300)
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, tissues, test_size=0.25, random_state=42, stratify=tissues)
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)

	# confusion matrix
	tissues_unique = np.unique(y_test)
	confusion_mat = confusion_matrix(y_test, preds, labels=tissues_unique)
	row_sums = confusion_mat.sum(axis=1)
	confusion_mat = confusion_mat / row_sums[:, np.newaxis]

	# Plot confusion matrix
	plt.figure(figsize=(16, 14))
	sns.heatmap(confusion_mat, 
		xticklabels=tissues_unique, 
		yticklabels=tissues_unique)
	plt.xticks(size=15)
	plt.yticks(size=15)
	plt.title("Tissue prediction, expression")
	plt.xlabel("Predicted tissue")
	plt.ylabel("True tissue")
	plt.tight_layout()
	plt.savefig("./out/confusion_matrix_expression.png")
	plt.show()

	# --- Fit linear model ---
	print("Fitting model...")
	clf = LogisticRegression(random_state=0)
	cv_results = cross_validate(clf, X_scaled, tissues, cv=5, verbose=0)

	np.savetxt("./out/exp_latents_test_acc.csv", cv_results['test_score'], delimiter=",")
	print(cv_results)

if __name__ == "__main__":
	main()