import numpy as np
import torch
import os
import pandas as pd
import socket
from os.path import join as pjoin
import matplotlib.image as mpimg
from PIL import Image


if socket.gethostname() == "andyjones":
    DATA_DIR = "/Users/andrewjones/Documents/beehive/gtex_data_sample"
    METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    SAVE_PATH = "/Users/andrewjones/Documents/beehive/gtex_image_analysis/data"
    NUM_SAMPLES = 10
else:
    DATA_DIR = "/projects/BEE/GTExV8_dpcca"
    METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    SAVE_PATH = "/tigress/aj13/gtex_image_analysis/autoencoder"
    NUM_SAMPLES = None

SAVE_FNAME = "train.pth"
IMG_DIR = os.path.join(DATA_DIR, "images")
RESIZE_SIZE = 512


def main():

    # ---------------- Load data ----------------

    # Metadata
    v8_metadata = pd.read_table(
        METADATA_PATH)
    v8_metadata['sample_id'] = [
        '-'.join(x.split("-")[:2]) for x in v8_metadata.SAMPID.values]

    # Get all tissue names
    tissue_names_imgs = [x for x in os.listdir(IMG_DIR) if (
        (not x.startswith('.')) and (x != "README"))]

    data_df = pd.DataFrame(
        columns=["im_fname", "im_id", "tissue"])
    for curr_tissue in tissue_names_imgs:

        # Get image filenames
        curr_dir = pjoin(DATA_DIR, "images", curr_tissue)
        curr_fnames = [x for x in os.listdir(curr_dir) if (
            x != ".DS_Store") and (x.endswith(".png") and ("failed" not in x) and (x != "README") and (len(x) >= 10))]

        print("Loading {}, {} samples".format(curr_tissue, len(curr_fnames)))

        curr_img_paths = np.array([pjoin(curr_dir, x) for x in curr_fnames])

        # Get sample names from image filenames
        # img_sample_ids = np.array(
        #     ['-'.join(os.path.splitext(x)[0].split("-")[:3]) for x in curr_fnames])
        img_sample_ids = np.array(
            [os.path.splitext(x)[0] for x in curr_fnames])


        # Make dataframe with both types of sample
        tiss = [curr_tissue for _ in range(len(curr_img_paths))]
        img_df = pd.DataFrame(
            {"im_fname": curr_img_paths, "im_id": img_sample_ids, "tissue": tiss})
        img_df = img_df.drop_duplicates(subset=["im_id", "tissue"])

        if NUM_SAMPLES is not None:
            img_df = img_df.iloc[:NUM_SAMPLES, :]

        # Concatenate to running combined dataframe
        data_df = pd.concat([data_df, img_df], axis=0)

    # Make sure all samples are unique
    # assert np.all(np.unique(data_df.im_id.values, return_counts=True)[
    #     1] == 1)
    # print(np.unique(data_df.im_id.values, return_counts=True))
    assert np.array_equal(np.unique(data_df.im_id.values, return_counts=True)[
        1], np.ones(data_df.shape[0]))

    image_data = [load_image(x, normalize=False)
                  for x in data_df.im_fname.values]
    image_data = np.array(image_data)

    # image_data = np.rollaxis(image_data, 3, 1)

    assert image_data.shape[0] == data_df.shape[0]

    print("Loaded {} samples.".format(data_df.shape[0]))

    torch.save({
        'images': image_data,
        'fnames': data_df.im_fname.values,
        'tissues': data_df.tissue.values,
    }, os.path.join(SAVE_PATH, SAVE_FNAME), pickle_protocol=4)

    print("Done.")


def load_image(file, normalize=False):
    """Load single image

    Args:
                                    file (TYPE): Description

    Returns:
                                    TYPE: Description
    """

    im = Image.open(file)
    if RESIZE_SIZE is not None:
        im.thumbnail((RESIZE_SIZE, RESIZE_SIZE),
                     Image.ANTIALIAS)  # resizes image in-place

    im = np.array(im)

    # im = mpimg.imread(file)

    # Fourth channel is blank for some reason - remove it
    im = im[:, :, :3]

    # Make UINT8
    # im = (im * 255).astype('uint8')

    # Mean subtract each channel
    if normalize:
        for curr_channel_num in range(im.shape[2]):
            curr_chan = im[:, :, curr_channel_num]
            im[:, :, curr_channel_num] = (
                curr_chan - np.mean(curr_chan)) / np.std(curr_chan)
    return im


def random_crop(im, size=50):

    # If size is None, don't crop at all
    if size == None:
        return im

    im_height, im_width, im_channels = im.shape
    assert im_height == im_width

    # Draw random point for x1 and y1
    max_pt = im_height - size
    x1, y1 = np.random.randint(low=0, high=max_pt, size=2)

    # Crop out square
    x2, y2 = x1 + size, y1 + size
    im_cropped = im[x1:x2, y1:y2, :]
    return im_cropped


if __name__ == "__main__":
    main()
