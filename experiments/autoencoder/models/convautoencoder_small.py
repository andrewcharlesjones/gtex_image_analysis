from torch import nn
import torch.nn.functional as F
from torch import sigmoid


class ConvAutoencoderSmall(nn.Module):
    def __init__(self, n_channels=3):
        super(ConvAutoencoderSmall, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(n_channels, 8, 5, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Linear layers
        # MNIST
        # self.fc1 = nn.Linear(1352, 64)
        # self.fc2 = nn.Linear(64, 8*13*13)

        # GTex
        self.fc1 = nn.Linear(31752, 1024)
        self.fc2 = nn.Linear(1024, 8*63*63)

        # Upsampling
        self.ups = nn.Upsample(scale_factor=2, mode="nearest")

        ## decoder layers ##
        self.t_conv1 = nn.Conv2d(8, n_channels, 5, padding=3)

    def forward(self, x):
        # # ------ Encode -------
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Flatten
        # import ipdb; ipdb.set_trace()
        x = x.view(x.size(0), -1)

        # Linear layers down to latent
        x = self.fc1(x)
        latent_z = self.fc2(x)

        # Reshape into tensor
        # x = latent_z.view(latent_z.size(0), 8, 13, 13)
        x = latent_z.view(latent_z.size(0), 8, 63, 63)

        # Decode
        x = self.ups(x)
        x = self.t_conv1(x)

        return x, latent_z
