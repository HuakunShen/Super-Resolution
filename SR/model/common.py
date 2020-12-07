import torch
from torch import nn
from PIL import Image
import torchvision
from torchvision import transforms


def double_conv(n_c, kernel_size, padding=1):
    return nn.Sequential(
        nn.Conv2d(n_c, n_c, kernel_size=kernel_size, padding=padding),
        nn.ReLU(True),
        nn.Conv2d(n_c, n_c, kernel_size=kernel_size, padding=padding),
        nn.ReLU(True),
    )


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

    def forward(self, inputs: torch.Tensor):
        return self.body(inputs)


class ResBlock(nn.Module):
    def __init__(self, conv, n_c, kernel_size: int = 3, bias=True, bn=False, activation=nn.ReLU(True),
                 res_scale=1):
        super(ResBlock, self).__init__()
        layers = []
        for i in range(2):
            layers.append(conv(n_c, n_c, kernel_size=kernel_size,
                               bias=bias, padding=kernel_size // 2))
            if bn:
                layers.append(nn.BatchNorm2d(n_c))
            if i == 0:
                layers.append(activation)
        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, inputs: torch.Tensor):
        result = self.body(inputs).mul(self.res_scale)
        result += inputs
        return result


if __name__ == '__main__':
    img = torch.rand((1, 3, 150, 150))
    model = ResBlock(nn.Conv2d, 3)
    print(model(img).shape)
