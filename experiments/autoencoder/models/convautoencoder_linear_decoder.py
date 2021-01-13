from torch import nn
import torch.nn.functional as F
from torch import sigmoid

import torch.nn as nn

# ------------------------------------------------------------------------------

class ConvAutoencoderLinearDec(nn.Module):

    def __init__(self, img_size, z_size):
        """Initialize LeNet5.
        """
        super(ConvAutoencoderLinearDec, self).__init__()

        self.img_size = img_size

        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 4, 5)

        self.fc1 = nn.Linear(4*(self.img_size//4 - 2)**2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2048)

        # self.fc4 = nn.Linear(64, 512)
        self.fc4 = nn.Linear(2048, 5096)
        self.fc5 = nn.Linear(5096, 3 * self.img_size**2)

# ------------------------------------------------------------------------------

    def encode(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        y = F.relu(self.conv1(x))
        y = F.max_pool2d(y, 2)
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, 2)
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        # y = F.relu(self.fc2(y))
        y = self.fc2(y)
        # y = self.fc3(y)
        return y

# ------------------------------------------------------------------------------

    def decode(self, z):
        y = F.relu(self.fc3(z))
        y = F.relu(self.fc4(y))
        y = self.fc5(y)
        # y = F.relu(self.fc6(y))
        y = y.view(-1, 3, self.img_size, self.img_size)
        return y

# ------------------------------------------------------------------------------

    def forward(self, x):
        """Perform forward pass on neural network.
        """
        latent_z = self.encode(x)
        x = self.decode(latent_z)
        return x, latent_z
