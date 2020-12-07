import torch
from torch import nn
from PIL import Image
import torchvision
from torchvision import transforms


class ResBlock(nn.Module):
    def __init__(self, conv, n_c, kernel_size: int, bias=True, bn=False, activation=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        layers = []
        for i in range(2):
            layers.append(conv(n_c, n_c, kernel_size=kernel_size, bias=bias, padding=(kernel_size // 2)))
            if bn:
                layers.append(nn.BatchNorm2d(n_c))
            if i == 0:
                layers.append(activation)
        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        result = self.body(x).mul(self.res_scale)
        result += x
        return result


class BasicBlock(nn.Module):
    def __init__(self, conv, n_c: int, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(
            conv(in_channels=n_c, out_channels=n_c, kernel_size=kernel_size, stride=stride, padding=padding,
                 bias=bias),
            nn.BatchNorm2d(num_features=n_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            conv(in_channels=n_c, out_channels=n_c, kernel_size=kernel_size, stride=stride, padding=padding,
                 bias=bias),
            nn.BatchNorm2d(num_features=n_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

    def forward(self, input):
        return self.body(input)


if __name__ == '__main__':
    # m = ResBlock(nn.Conv2d, 10, 3)
    # m = BasicBlock(nn.Conv2d, 10)
    pass
