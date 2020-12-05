import torch.nn as nn
import torch.nn.functional as F
from tensorflow import Tensor


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=(4, 4),
                               stride=2,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(4, 4),
                               stride=2,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(4, 4),
                               stride=2,
                               padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels=128,
                                        out_channels=64,
                                        kernel_size=(4, 4),
                                        stride=2,
                                        padding=1)
        self.conv5 = nn.ConvTranspose2d(in_channels=64,
                                        out_channels=32,
                                        kernel_size=(4, 4),
                                        stride=2,
                                        padding=1)
        self.conv6 = nn.ConvTranspose2d(in_channels=32,
                                        out_channels=3,
                                        kernel_size=(4, 4),
                                        stride=2,
                                        padding=1)

    def forward(self, x: Tensor):
        ipt = F.interpolate(x, scale_factor=4)
        x = self.conv1(ipt)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x, True)
        x = self.conv4(x)
        x = F.relu(x, True)
        x = self.conv5(x)
        x = F.relu(x, True)
        x = self.conv6(x)
        return x + ipt
