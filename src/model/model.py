from model.common import ResBlock
import os
import sys
import torch
from torch import nn
from PIL import Image
import torchvision
from torchvision import transforms

sys.path.append(os.path.abspath('..'))
print(sys.path)


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
    )


class ResNetSRTransfer(nn.Module):
    def __init__(self):
        super(ResNetSRTransfer, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet_trim = nn.Sequential(
            *list(self.resnet.children())[:-2],
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=2, stride=2),
            double_conv(256, 256)
        )

    def forward(self, input):
        return self.resnet_trim(input)


class ResNetSR(nn.Module):
    def __init__(self, factor=4):
        super(ResNetSR, self).__init__()
        self.factor = factor
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=15, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            ResBlock(nn.Conv2d, 64, kernel_size=15),
            ResBlock(nn.Conv2d, 64, kernel_size=15),
            nn.ConvTranspose2d(in_channels=64, out_channels=3,
                               kernel_size=15, stride=2)
        )

    def forward(self, input):
        desired_size = input.shape[-1] * self.factor
        scaled_up_input = transforms.Compose([
            transforms.Resize(desired_size, Image.BICUBIC)
        ])(input)
        output = self.body(scaled_up_input)
        return transforms.Compose([
            transforms.CenterCrop(desired_size)
        ])(output)


if __name__ == '__main__':
    print(sys.path)
    print("debug")
    model = ResNetSR()
    # img = Image.open('/home/hacker/Documents/Super-Resolution/datasets/DIV2K/custom/DIV2K_train_HR_crop_600/0001.png')
    # img = transforms.ToTensor()(img)
    # img.show()
    # img = img.unsqueeze(0)
    img = torch.rand((1, 3, 150, 150))
    output = model(img)
    print(output.shape)
