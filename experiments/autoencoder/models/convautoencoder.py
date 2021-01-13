from torch import nn
import torch.nn.functional as F
from torch import sigmoid


class ConvAutoencoder(nn.Module):
    def __init__(self, n_channels=3):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(n_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(4, 4, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(4, 4, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(4, n_channels, 2, stride=2)

    def forward(self, x):
        # # ------ Encode -------
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        latent_z = self.pool(x)

        # ----- Decode --------
        x = F.relu(self.t_conv1(latent_z))
        x = F.relu(self.t_conv2(x))
        x = sigmoid(self.t_conv3(x))

        # # ------ Encode -------
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # latent_z = self.pool(x)

        # # ----- Decode --------
        # x = F.relu(self.t_conv1(latent_z))
        # x = sigmoid(self.t_conv3(x))


        return x, latent_z
        