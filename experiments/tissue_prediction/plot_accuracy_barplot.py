import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

########## Prediction from raw data ##########

# Load results

exp_results = pd.read_csv("./out/exp_latents_test_acc.csv", header=None)
exp_results.columns = ['test_acc']
exp_results['type'] = ['Expression' for _ in range(exp_results.shape[0])]

img_results = pd.read_csv("./out/im_ae_latents_test_acc.csv", header=None)
img_results.columns = ['test_acc']
img_results['type'] = ['Images' for _ in range(img_results.shape[0])]

all_results = pd.concat([exp_results, img_results], axis=0)


plt.figure(figsize=(12, 10))
sns.barplot(data=all_results, x="type", y="test_acc")
plt.xlabel("")
plt.ylabel("Test accuracy")
plt.title("Tissue prediction")
plt.tight_layout()
plt.savefig("./out/tissue_pred_barplot.png")
plt.show()


########## Prediction from PCCA latent variables ##########

# Load results

shared_results = pd.read_csv("./out/pcca_shared_vars_test_acc.csv", header=None)
shared_results.columns = ['test_acc']
shared_results['type'] = ['Shared' for _ in range(shared_results.shape[0])]

exp_results = pd.read_csv("./out/pcca_exp_vars_test_acc.csv", header=None)
exp_results.columns = ['test_acc']
exp_results['type'] = ['Expression' for _ in range(exp_results.shape[0])]

img_results = pd.read_csv("./out/pcca_img_vars_test_acc.csv", header=None)
img_results.columns = ['test_acc']
img_results['type'] = ['Images' for _ in range(img_results.shape[0])]

all_results = pd.concat([shared_results, exp_results, img_results], axis=0)


font = {'size': 60}
matplotlib.rc('font', **font)

plt.figure(figsize=(22, 20))
sns.barplot(data=all_results, x="type", y="test_acc")
plt.xlabel("")
plt.ylabel("Test accuracy")
plt.title("Tissue prediction")
plt.tight_layout()
plt.savefig("./out/tissue_pred_barplot_pcca.png")
plt.show()


# import ipdb; ipdb.set_trace()
