from torch import nn
import torch.nn.functional as F
from torch import sigmoid


class ConvAutoencoderBig(nn.Module):
    def __init__(self, n_channels=3):
        super(ConvAutoencoderBig, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(n_channels, 8, 5, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 5, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 5, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Linear layers
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 128*4*4)

        # Upsampling
        self.ups = nn.Upsample(scale_factor=2, mode="nearest")

        ## decoder layers ##
        # self.t_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # self.t_conv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # self.t_conv3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        # self.t_conv4 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        # self.t_conv5 = nn.ConvTranspose2d(8, n_channels, 2, stride=2)
        self.t_conv1 = nn.Conv2d(128, 64, 5, padding=2)
        self.t_conv2 = nn.Conv2d(64, 32, 5, padding=2)
        self.t_conv3 = nn.Conv2d(32, 16, 5, padding=2)
        self.t_conv4 = nn.Conv2d(16, 8, 5, padding=2)
        self.t_conv5 = nn.Conv2d(8, n_channels, 5, padding=2)

    def forward(self, x):
        # # ------ Encode -------
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Linear layers down to latent
        latent_z = self.fc1(x)
        x = F.relu(latent_z)
        x = F.relu(self.fc2(x))

        # Reshape into tensor
        x = x.view(x.size(0), 128, 4, 4)

        # Decode
        x = self.ups(x)
        x = F.relu(self.t_conv1(x))
        x = self.ups(x)
        x = F.relu(self.t_conv2(x))
        x = self.ups(x)
        x = F.relu(self.t_conv3(x))
        x = self.ups(x)
        x = F.relu(self.t_conv4(x))
        x = self.ups(x)
        x = sigmoid(self.t_conv5(x))

        return x, latent_z
