# https://arxiv.org/abs/1505.04597
# TODO: rename functions and variables

import torch
import torch.nn as nn
from math import ceil
from torchvision import transforms
from PIL import Image


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
    )


def crop_tensor(src, target):
    target_size = target.size()[2]
    src_size = src.size()[2]
    delta = src_size - target_size
    delta = delta // 2
    tmp_src_size = src_size if src_size % 2 == 0 else src_size - 1
    return src[:, :, delta:tmp_src_size - delta, delta: tmp_src_size-delta]


class UNet(nn.Module):
    def __init__(self, factor=4):
        super(UNet, self).__init__()
        self.factor = factor
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, image):
        desired_size = image.shape[-1] * self.factor
        image = transforms.Compose([
            transforms.Resize(desired_size, Image.BICUBIC)
        ])(image)
        x1 = self.down_conv_1(image)    #
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)       #
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)       #
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)       #
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        print(x9.shape)
        # decoder
        x = self.up_trans_1(x9)
        y = crop_tensor(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_tensor(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))
        print(x.shape)

        x = self.up_trans_3(x)
        y = crop_tensor(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))
        print(x.shape)

        x = self.up_trans_4(x)
        print(x.shape)

        y = crop_tensor(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))
        print(x.shape)

        x = self.out(x)

        return x


if __name__ == "__main__":
    image = torch.rand((1, 3, 150, 150))
    print(image.size())
    model = UNet()
    out = model(image)
    print(out.size())
