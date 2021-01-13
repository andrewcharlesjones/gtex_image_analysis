from torch import nn
import torch.nn.functional as F
from torch import sigmoid


class ConvAutoencoderOneLayer(nn.Module):
    def __init__(self, img_size):
        super(ConvAutoencoderOneLayer, self).__init__()
        self.img_size = img_size
        
        self.fc1 = nn.Linear(self.img_size**2 * 3, 32)
        self.fc2 = nn.Linear(32, self.img_size**2 * 3)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        latent_z = self.fc1(x)
        x = self.fc2(latent_z)
        x = x.view(-1, 3, self.img_size, self.img_size)

        return x, latent_z

