import math
from torch import nn
import torch
from torch.nn.modules.activation import ReLU
from torchvision import transforms
from PIL import Image


class SRCNN(nn.Module):
    def __init__(self, in_channel: int = 3):
        super(SRCNN, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64,
                      kernel_size=9, padding=9//2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=5, padding=5//2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=in_channel,
                      kernel_size=5, padding=5//2),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        return self.body(inputs)


if __name__ == "__main__":
    model = SRCNN(3)

    img = torch.rand((1, 3, 600, 600))
    print(model(img).shape)
