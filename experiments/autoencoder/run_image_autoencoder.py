import pandas as pd
import numpy as np
import os
import matplotlib.image as mpimg
from os.path import join as pjoin
import socket

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

from models import ConvAutoencoder, ConvAutoencoderBig, ConvAutoencoderSmall, DCGANAE128

# from data import loader

import matplotlib.pyplot as plt

torch.manual_seed(1)

if socket.gethostname() == "andyjones":
    DATA_DIR = "/Users/andrewjones/Documents/beehive/gtex_image_analysis/data"
    RANDOM_CROP_SIZE = 128
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8
    NUM_CROPS_TEST = 2
    SAVE_MODEL_EVERY = 5
else:
    DATA_DIR = "/tigress/aj13/gtex_image_analysis/autoencoder/"
    RANDOM_CROP_SIZE = 128
    NUM_EPOCHS = 3 # 12000
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    NUM_CROPS_TEST = 20
    SAVE_MODEL_EVERY = 20

IM_SIZE = (1000, 1000, 3)
DATA_FILE = "train.pth"
LOAD_MODEL = True
LATENT_Z_SIZE = 1024
RESIZE_SIZE = 512


def main():

    # ------ Prepare devices -------

    # Set CUDA device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("\n=== Device: {} ===\n".format(device))

    print("Forming dataset...")
    gtexv8_dataset = Dataset()
    print("Number of samples: {}".format(len(gtexv8_dataset)))

    # prepare data loaders
    print("Preparing data loader...")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        gtexv8_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    # ----- Prepare network ------
    model = DCGANAE128(latent_z_size=LATENT_Z_SIZE).to(device)

    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Load saved model if desired
    if LOAD_MODEL:
        model, optimizer, start_epoch, loss_history = load_checkpoint(
            model, optimizer, filename='./out/model.pt')
    else:
        start_epoch = 1
        loss_history = []

    # ------- Train the network --------
    print("Training...")

    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):

        train_loss, plot_ims, plot_recons = train(
            train_loader, model, criterion, optimizer, epoch, device, return_examples=True)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))
        loss_history.append(train_loss)

        # Save current version of reconstructions
        if epoch % 5 == 0:
            plot_example(plot_ims, plot_recons,
                         "./out/reconstructions_ongoing.png")
            plot_loss(loss_history)
        if epoch % SAVE_MODEL_EVERY == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(
            ), 'optimizer': optimizer.state_dict(), 'loss_history': loss_history}
            torch.save(state, "./out/model.pt")

    # Save final trained model
    state = {'epoch': epoch, 'state_dict': model.state_dict(
    ), 'optimizer': optimizer.state_dict(), 'loss_history': loss_history}
    torch.save(state, "./out/model.pt")

    # ---- Extract latent representation -------

    # Get untransformed data for "testing" (no rotations, flips, etc.)
    gtexv8_dataset = DatasetTest()

    gtexv8_dataset.is_transformed = False
    test_loader = torch.utils.data.DataLoader(
        gtexv8_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    print("Extracting latent variables...")

    # Initialize array to store latent reps
    latent_z_all = np.zeros((len(gtexv8_dataset), 1024))

    model.eval()
    with torch.no_grad():
        for epoch in range(NUM_CROPS_TEST):

            print("Test epoch {} / {}".format(epoch + 1, NUM_CROPS_TEST))

            # Run test loop for one epoch
            if epoch == NUM_CROPS_TEST - 1:
                curr_latent_z, ims_test_examples, recons_test_examples, fnames_all = test(
                    test_loader, model, epoch, device, return_examples=True)
            else:
                curr_latent_z = test(
                    test_loader, model, epoch, device, return_examples=False)

            latent_z_all = latent_z_all + curr_latent_z

    # Take average latent representation across subcrops of each image
    latent_z_all = latent_z_all / NUM_CROPS_TEST

    recons_test_examples = np.concatenate(recons_test_examples)
    ims_test_examples = np.concatenate(ims_test_examples)
    ims_test_examples = np.moveaxis(ims_test_examples, 1, -1)

    assert np.array_equal(np.array(np.concatenate(
        fnames_all)), gtexv8_dataset.im_fnames)

    out_df = pd.DataFrame(latent_z_all)
    out_df['fname'] = gtexv8_dataset.im_fnames
    out_df = out_df.groupby('fname').mean()
    latent_z_all = out_df.values
    fnames_all = out_df.index.values

    # Move channels to last axis
    recons_test_examples = np.moveaxis(recons_test_examples, 1, -1)

    print("Latent shape: {}".format(latent_z_all.shape))
    assert recons_test_examples.shape[0] == ims_test_examples.shape[0]

    # ------- Save plots --------

    print("Saving plots...")

    # Training loss plot
    plot_loss(loss_history)

    # Reconstruction plot
    plot_example(ims_test_examples, recons_test_examples, "./out/reconstructions.png")

    # Save latent representations
    np.save("./out/latent_z.npy", latent_z_all)

    # Save image filenames and tissue labels to be able to recover them later
    np.save("./out/im_fnames.npy", np.array(fnames_all))
    np.save("./out/tissue_labels.npy", gtexv8_dataset.tissue_labels)

    print("Done.")


def train(train_loader, model, criterion, optimizer, epoch, device, return_examples=False):

    train_loss = 0.0
    counter = 0
    for ims, tissues, fnames in train_loader:

        ims = ims.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        outputs, latent_z = model(ims)

        # calculate the loss
        loss = criterion(outputs, ims)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        # update running training loss
        train_loss += loss.item()*ims.size(0)

        if counter == 0:
            plot_ims = np.moveaxis(ims.detach().cpu().numpy()[
                                   :3, :, :, :], 1, -1)
            plot_recons = np.moveaxis(
                outputs.detach().cpu().numpy()[:3, :, :, :], 1, -1)
        counter += 1

    train_loss = train_loss/len(train_loader)

    if return_examples:
        return train_loss, plot_ims, plot_recons
    else:
        return train_loss


def test(test_loader, model, epoch, device, return_examples=False):

    # If this is the last epoch, save the filenames
    if return_examples:
        fnames_all = []

    # Prepare to save the latent reps on this round
    curr_latent_z = []
    ims_examples = []
    recons_examples = []
    counter = 0
    for ims, tissues, fnames in test_loader:

        ims = ims.to(device)

        if return_examples == True and counter == 0:
            reconstructions, latent_z = model(ims)
            reconstructions, latent_z, ims = reconstructions.detach().cpu(
            ).numpy(), latent_z.detach().cpu().numpy(), ims.detach().cpu().numpy()

            # Save some images and reconstructions for plotting
            rand_batch_idx = np.random.choice(
                np.arange(reconstructions.shape[0]), size=5, replace=False)
            recons_examples.append(
                reconstructions[rand_batch_idx, :, :, :])
            ims_examples.append(ims[rand_batch_idx, :, :, :])
        else:
            _, latent_z = model(ims)
            latent_z = latent_z.detach().cpu().numpy()

        curr_latent_z.append(latent_z)

        if return_examples:
            fnames_all.append(fnames)

        counter += 1

    curr_latent_z = np.concatenate(curr_latent_z)

    if return_examples:
        return curr_latent_z, ims_examples, recons_examples, np.array(fnames_all)
    else:
        return curr_latent_z


def load_checkpoint(model, optimizer, filename='./out/model.pt'):
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_history = checkpoint['loss_history']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, loss_history


def plot_example(images, reconstructions, path):
    n = images.shape[0]
    plt.figure(figsize=(n*4, 10))
    for ii in range(n):
        plt.subplot(2, n, ii + 1)
        plt.imshow(images[ii, :, :, :])
        plt.subplot(2, n, ii + n + 1)
        plt.imshow(reconstructions[ii, :, :, :])
    plt.savefig(path)
    plt.close()


def plot_loss(loss_trace):
    plt.plot(loss_trace)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('./out/train_loss.png')
    plt.close()


# Load all images before starting training
class Dataset(Data.Dataset):

    def __init__(self, is_transformed=True):
        'Initialization'

        data = torch.load(pjoin(DATA_DIR, DATA_FILE))
        self.im_fnames = data['fnames']
        self.images = data['images']
        self.tissue_labels = data['tissues']
        self.is_transformed = is_transformed

        if RANDOM_CROP_SIZE is not None:
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                # transforms.RandomRotation((0, 360)),
                transforms.RandomCrop(RANDOM_CROP_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.no_transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.RandomCrop(RANDOM_CROP_SIZE),
                transforms.ToTensor()
            ])
        else:
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                # transforms.RandomRotation((0, 360)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.no_transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_fnames)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        im_fname = self.im_fnames[index]
        im_full = self.images[index]
        tissue_label = self.tissue_labels[index]
        im_fname = self.im_fnames[index]

        bad_crop = True
        while bad_crop:
            if self.is_transformed:
                im = self.transform_image(im_full).numpy()
            else:
                im = self.no_transform_image(im_full).numpy()
            # We want to avoid all black crops because it prevents us from
            # feature normalization.
            if im.min() == im.max():
                continue
            # We want to avoid crops that are majority black.
            if (im == 0).sum() / im.size > 0.5:
                continue

            bad_crop = False

        return im, tissue_label, im_fname


class DatasetTest(Data.Dataset):

    def __init__(self, is_transformed=False):

        data = torch.load(pjoin(DATA_DIR, DATA_FILE))
        self.im_fnames = data['fnames']
        self.images = data['images']
        self.tissue_labels = data['tissues']
        self.is_transformed = is_transformed

        self.no_transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomCrop(RANDOM_CROP_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_fnames)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        im_fname = self.im_fnames[index]
        im_full = self.images[index]  # load_image(im_fname)

        # Get good crop
        bad_crop = True
        while bad_crop:
            im = self.no_transform_image(im_full).numpy()
            # We want to avoid all black crops because it prevents us from
            # feature normalization.
            if im.min() == im.max():
                continue
            # We want to avoid crops that are majority black.
            if (im == 0).sum() / im.size > 0.5:
                continue

            bad_crop = False

        tissue_label = self.tissue_labels[index]
        im_fname = self.im_fnames[index]

        return im, tissue_label, im_fname


if __name__ == '__main__':
    main()
