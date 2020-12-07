# from model.common import ResBlock
import os
import sys
import torch
from torch import nn
from PIL import Image
import torchvision
from torchvision import transforms

sys.path.append(os.path.abspath('..'))


# class BaseModel(nn.Module):
#     """
#     Base class for all models
#     """
#     @abstractmethod
#     def forward(self, *inputs):
#         """
#         Forward pass logic

#         :return: Model output
#         """
#         raise NotImplementedError

#     def __str__(self):
#         """
#         Model prints with number of trainable parameters
#         """
#         model_parameters = filter(lambda p: p.requires_grad, self.parameters())
#         params = sum([np.prod(p.size()) for p in model_parameters])
#         return super().__str__() + '\nTrainable parameters: {}'.format(params)


def double_conv(n_c, kernel_size, padding=1):
    return nn.Sequential(
        nn.Conv2d(n_c, n_c, kernel_size=kernel_size, padding=padding),
        nn.ReLU(True),
        nn.Conv2d(n_c, n_c, kernel_size=kernel_size, padding=padding),
        nn.ReLU(True),
    )


# class ResNetSRTransfer(nn.Module):
#     def __init__(self):
#         super(ResNetSRTransfer, self).__init__()
#         self.resnet = torchvision.models.resnet18(pretrained=True)
#         self.resnet_trim = nn.Sequential(
#             *list(self.resnet.children())[:-2],
#             nn.ConvTranspose2d(
#                 in_channels=512, out_channels=256, kernel_size=2, stride=2),
#             double_conv(256, 256)
#         )

#     def forward(self, input):
#         return self.resnet_trim(input)


# class ResNetSR(nn.Module):
#     def __init__(self, factor=4):
#         super(ResNetSR, self).__init__()
#         self.factor = factor
#         self.body = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64,
#                       kernel_size=15, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1,
#                            affine=True, track_running_stats=True),
#             ResBlock(nn.Conv2d, 64, kernel_size=15),
#             ResBlock(nn.Conv2d, 64, kernel_size=15),
#             nn.ConvTranspose2d(in_channels=64, out_channels=3,
#                                kernel_size=15, stride=2)
#         )

#     def forward(self, input):
#         desired_size = input.shape[-1] * self.factor
#         scaled_up_input = transforms.Compose([
#             transforms.Resize(desired_size, Image.BICUBIC)
#         ])(input)
#         output = self.body(scaled_up_input)
#         return transforms.Compose([
#             transforms.CenterCrop(desired_size)
#         ])(output)


class ResBlock(nn.Module):
    def __init__(self, conv, n_c, kernel_size: int, bias=True, bn=False, activation=nn.ReLU(True),
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

    def forward(self, x):
        result = self.body(x).mul(self.res_scale)
        result += x
        return result


class ResNetPretrainedSS(nn.Module):
    """600 to 600"""

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
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
    """150 to 600"""

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


class ResNetPretrainedDSRes(nn.Module):
    """150 to 600"""

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.body = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.ConvTranspose2d(512, 256, kernel_size=7,
                               stride=4, padding=2, output_padding=0),
            nn.ReLU(True),
            ResBlock(nn.Conv2d, 256, 5),
            nn.ConvTranspose2d(256, 128, kernel_size=9,
                               stride=4, padding=3, output_padding=1),
            nn.ReLU(True),
            ResBlock(nn.Conv2d, 128, 5),
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=4, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2,
                               padding=2, output_padding=1),
        )

    def forward(self, input):
        result = self.body(input)
        return result


if __name__ == '__main__':
    # print(sys.path)
    # print("debug")
    # model = ResNetSR()
    # img = torch.rand((1, 3, 150, 150))
    # output = model(img)
    # print(output.shape)

    model = ResNetPretrainedDSRes()
    img = torch.rand((4, 3, 150, 150))
    # print(model(img))
    print(model(img).shape)
