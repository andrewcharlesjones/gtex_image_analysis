import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as pjoin
import socket
import matplotlib.image as mpimg
import numpy as np
import os
from scipy.stats import pearsonr


if socket.gethostname() == "andyjones":
    DATA_DIR = "./out"
    SAVE_DIR = "./out/extreme_imgs"
else:
    DATA_DIR = "/tigress/aj13/gtex_image_analysis/cca/pcca_out"
    SAVE_DIR = "/tigress/aj13/gtex_image_analysis/cca/pcca_extreme_imgs"

NUM_VARS_TO_PLOT = 10
NUM_IMS_TO_PLOT = 5

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True


def main():


    ### Shared LVs

    # Load latent variables
    latent_vars_pcca_shared = pd.read_csv(pjoin(DATA_DIR, "pcca_lvs_shared.csv"), index_col=0).transpose()
    img_fnames = pd.read_csv(pjoin(DATA_DIR, "img_fnames_pcca.csv"), index_col=0)

    ## Loop through latent dims, sort, and save
    for ii in range(latent_vars_pcca_shared.shape[1]):
        curr_lv = latent_vars_pcca_shared.iloc[:, ii].values

        # Sort
        sorted_idx = np.argsort(curr_lv)
        curr_lv_sorted = curr_lv[sorted_idx]
        curr_fnames_sorted = img_fnames.fname.values[sorted_idx]

        # Get high and low
        low_fnames = curr_fnames_sorted[:NUM_IMS_TO_PLOT]
        high_fnames = curr_fnames_sorted[-NUM_IMS_TO_PLOT:]

        print("Var {}".format(ii))
        print("High filenames: {}".format(curr_fnames_sorted[:10]))
        print("Low filenames: {}".format(curr_fnames_sorted[-10:]))
        print("\n")

        plt.close()
        fig = plt.figure(figsize=(20, 10))

        for jj in range(NUM_IMS_TO_PLOT):

            plt.subplot(2, NUM_IMS_TO_PLOT, jj + 1)
            tiss = high_fnames[jj].split("/")[-2]
            plt.imshow(load_image(high_fnames[jj]))
            plt.xticks([])
            plt.yticks([])
            plt.title(tiss)
            if jj == 0:
                plt.ylabel("High values", rotation=0, labelpad=100)
                plt.tight_layout()
            plt.subplot(2, NUM_IMS_TO_PLOT, jj + NUM_IMS_TO_PLOT + 1)
            if jj == 0:
                plt.ylabel("Low values", rotation=0, labelpad=100)
                plt.tight_layout()
            tiss = low_fnames[jj].split("/")[-2]
            plt.imshow(load_image(low_fnames[jj]))
            plt.xticks([])
            plt.yticks([])
            plt.title(tiss)
        fig.suptitle('IBFA latent variable {}'.format(ii + 1))
        plt.tight_layout()
        plt.savefig(
            pjoin(SAVE_DIR, "pcca_shared_vars_extreme_imgs_var{}.png".format(ii + 1)))


    ### Exclusive image LVs
    
    # Load latent variables
    latent_vars_pcca = pd.read_csv(pjoin(DATA_DIR, "pcca_img_lvs_exclusive.csv"), index_col=0).transpose()
    img_fnames = pd.read_csv(pjoin(DATA_DIR, "img_fnames_pcca.csv"), index_col=0)

    ## Loop through latent dims, sort, and save
    for ii in range(latent_vars_pcca.shape[1]):
        curr_lv = latent_vars_pcca.iloc[:, ii].values

        # Sort
        sorted_idx = np.argsort(curr_lv)
        curr_lv_sorted = curr_lv[sorted_idx]
        curr_fnames_sorted = img_fnames.fname.values[sorted_idx]

        # Get high and low
        low_fnames = curr_fnames_sorted[:NUM_IMS_TO_PLOT]
        high_fnames = curr_fnames_sorted[-NUM_IMS_TO_PLOT:]

        print("Var {}".format(ii))
        print("High filenames: {}".format(curr_fnames_sorted[:10]))
        print("Low filenames: {}".format(curr_fnames_sorted[-10:]))
        print("\n")

        plt.close()
        fig = plt.figure(figsize=(20, 10))

        for jj in range(NUM_IMS_TO_PLOT):

            plt.subplot(2, NUM_IMS_TO_PLOT, jj + 1)
            tiss = high_fnames[jj].split("/")[-2]
            plt.imshow(load_image(high_fnames[jj]))
            plt.xticks([])
            plt.yticks([])
            plt.title(tiss)
            if jj == 0:
                plt.ylabel("High values", rotation=0, labelpad=100)
                plt.tight_layout()
            plt.subplot(2, NUM_IMS_TO_PLOT, jj + NUM_IMS_TO_PLOT + 1)
            if jj == 0:
                plt.ylabel("Low values", rotation=0, labelpad=100)
                plt.tight_layout()
            tiss = low_fnames[jj].split("/")[-2]
            plt.imshow(load_image(low_fnames[jj]))
            plt.xticks([])
            plt.yticks([])
            plt.title(tiss)
        fig.suptitle('IBFA latent variable {}'.format(ii + 1))
        plt.tight_layout()
        plt.savefig(
            pjoin(SAVE_DIR, "pcca_exclusive_img_vars_extreme_imgs_var{}.png".format(ii + 1)))

    ### Exclusive expression LVs
    
    # Load latent variables
    latent_vars_pcca = pd.read_csv(pjoin(DATA_DIR, "pcca_exp_lvs_exclusive.csv"), index_col=0).transpose()
    img_fnames = pd.read_csv(pjoin(DATA_DIR, "img_fnames_pcca.csv"), index_col=0)

    ## Loop through latent dims, sort, and save
    for ii in range(latent_vars_pcca.shape[1]):
        curr_lv = latent_vars_pcca.iloc[:, ii].values

        # Sort
        sorted_idx = np.argsort(curr_lv)
        curr_lv_sorted = curr_lv[sorted_idx]
        curr_fnames_sorted = img_fnames.fname.values[sorted_idx]

        # Get high and low
        low_fnames = curr_fnames_sorted[:NUM_IMS_TO_PLOT]
        high_fnames = curr_fnames_sorted[-NUM_IMS_TO_PLOT:]

        print("Var {}".format(ii))
        print("High filenames: {}".format(curr_fnames_sorted[:10]))
        print("Low filenames: {}".format(curr_fnames_sorted[-10:]))
        print("\n")

        plt.close()
        fig = plt.figure(figsize=(20, 10))

        for jj in range(NUM_IMS_TO_PLOT):

            plt.subplot(2, NUM_IMS_TO_PLOT, jj + 1)
            tiss = high_fnames[jj].split("/")[-2]
            plt.imshow(load_image(high_fnames[jj]))
            plt.xticks([])
            plt.yticks([])
            plt.title(tiss)
            if jj == 0:
                plt.ylabel("High values", rotation=0, labelpad=100)
                plt.tight_layout()
            plt.subplot(2, NUM_IMS_TO_PLOT, jj + NUM_IMS_TO_PLOT + 1)
            if jj == 0:
                plt.ylabel("Low values", rotation=0, labelpad=100)
                plt.tight_layout()
            tiss = low_fnames[jj].split("/")[-2]
            plt.imshow(load_image(low_fnames[jj]))
            plt.xticks([])
            plt.yticks([])
            plt.title(tiss)
        fig.suptitle('PCCA latent variable {}'.format(ii + 1))
        plt.tight_layout()
        plt.savefig(
            pjoin(SAVE_DIR, "pcca_exclusive_exp_vars_extreme_imgs_var{}.png".format(ii + 1)))




def load_image(file, normalize=False):
    im = mpimg.imread(file)

    # Fourth channel is blank for some reason - remove it
    im = im[:, :, :3]

    # Mean subtract each channel
    if normalize:
        for curr_channel_num in range(im.shape[2]):
            curr_chan = im[:, :, curr_channel_num]
            im[:, :, curr_channel_num] = (
                curr_chan - np.mean(curr_chan)) / np.std(curr_chan)
    return im


if __name__ == "__main__":
    main()
