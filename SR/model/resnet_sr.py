import os
import sys
import torch
from torch import nn
from PIL import Image
import torchvision
from torchvision import transforms
from model.common import double_conv, BasicBlock, ResBlock


class ResNetPretrainedSS(nn.Module):
    """600 to 600, input must be of size 600x600"""

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.body = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.ConvTranspose2d(512, 256, kernel_size=7,
                               stride=4, padding=0, output_padding=1),
            nn.ReLU(True),
            double_conv(256, 5, 1),
            nn.ConvTranspose2d(256, 128, kernel_size=7,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(True),
            double_conv(128, 5, 1),
            nn.ConvTranspose2d(128, 64, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )

    def forward(self, input):
        result = self.body(input)
        result += input
        return result


class ResNetPretrainedDS(nn.Module):
    """150 to 600, input must be of size 150x150"""

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.body = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.ConvTranspose2d(512, 256, kernel_size=7,
                               stride=4, padding=0, output_padding=1),
            nn.ReLU(True),
            double_conv(256, 5, 1),
            nn.ConvTranspose2d(256, 128, kernel_size=7,
                               stride=4, padding=2, output_padding=1),
            nn.ReLU(True),
            double_conv(128, 5, 1),
            nn.ConvTranspose2d(128, 64, kernel_size=5,
                               stride=4, padding=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2,
                               padding=2, output_padding=1),
        )

    def forward(self, input):
        result = self.body(input)
        return result


if __name__ == '__main__':
    img = torch.rand((4, 3, 150, 150))
    model = ResNetPretrainedDS()
    print(model(img).shape)
