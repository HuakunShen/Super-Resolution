import torch.nn as nn
import torch.nn.functional as F
from tensorflow import Tensor


class DRRN(nn.Module):
    def __init__(self):
        super(DRRN, self).__init__()
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
        self.conv3 = nn.ConvTranspose2d(in_channels=64,
                                        out_channels=32,
                                        kernel_size=(4, 4),
                                        stride=2,
                                        padding=1)
        self.conv4 = nn.Conv2d(in_channels=32,
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
        opt1 = self.conv1(ipt)
        opt1 = F.relu(opt1)
        opt2 = self.conv2(opt1)
        opt2 = F.relu(opt2)
        opt3 = self.conv3(opt2)
        opt3 = F.relu(opt3)
        opt4 = self.conv4(opt1 + opt3)
        opt4 = F.relu(opt4)
        opt5 = self.conv5(opt4)
        opt5 = F.relu(opt5)
        opt6 = self.conv6(opt1 + opt5)
        return opt6 + ipt
