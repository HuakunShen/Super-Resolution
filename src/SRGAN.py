# SRGAN:
# https://arxiv.org/abs/1609.04802

# models for SRGAN, where Generator is SR-Resnet
#
# call Generator(num_ResidualBlock, num_UpscaleBlock)
# to initialize the G function, 
# where the output scale is x2^(num_UpscaleBlock)

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(0.3)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.bn1(tmp)
        tmp = self.relu1(tmp)
        tmp = self.conv2(tmp)
        tmp = self.bn2(tmp)
        return torch.add(x, tmp)


# upscale x2
class UpscaleBlock(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=256,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(UpscaleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.deconv = nn.PixelShuffle(2)
    
    def forward(self, x):
        output = self.conv(x)
        return self.deconv(output)


# SR-Resnet SRCNN
class Generator(nn.Module):
    def __init__(self, num_ResidualBlock=3, num_UpscaleBlock=2):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=9,
                               stride=1,
                               padding=4,
                               bias=False)
        self.relu1 = nn.LeakyReLU(0.3)
        
        residual_blocks = []
        for i in range(num_ResidualBlock):
            residual_blocks.append(ResidualBlock())
        self.residual = nn.Sequential(*residual_blocks)
        
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        
        # SubPixel Convolution
        deconv_blocks = []
        for i in range(num_UpscaleBlock):
            deconv_blocks.append(UpscaleBlock())
        self.deconv = nn.Sequential(*deconv_blocks)
        
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=3,
                               kernel_size=9,
                               stride=1,
                               padding=4,
                               bias=False)
    
    def forward(self, x):
        tmp1   = self.conv1(x)
        output = self.relu1(tmp1)
        output = self.residual(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = torch.add(tmp1, output)
        
        # upscale
        output = self.deconv(output)
        return self.conv3(output)


# a binary classification model
# need fixed image size
### this class has not been tested ###
class Discriminator(nn.Module):
    def __init__(self, image_size=600):
        super(Discriminator, self).__init__()
        
        
        self.conv = nn.Sequential(
                nn.Conv2d(3,64,3,1,1),
                nn.LeakyReLU(0.3),
                
                nn.Conv2d(64,64,3,2,1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.3),
                
                nn.Conv2d(64,128,3,1,1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.3),
                
                nn.Conv2d(128,128,3,2,1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.3),
                
                nn.Conv2d(128,256,3,1,1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.3),
                
                nn.Conv2d(256,256,3,2,1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.3),
                
                nn.Conv2d(256,512,3,1,1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.3),
                
                nn.Conv2d(512,512,3,2,1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.3)
                )
        self.fc = nn.Sequential(
                # TODO
                # find out flattened size and change the first input
                nn.Linear(1, 1024),
                nn.LeakyReLU(0.3),
                nn.Linear(1024, 1)
                )
        
    def forward(self, x):
        output = self.conv(x)
        
        # TODO
        # this may need to be modified
        output = output.view(-1)
        
        output = self.fc(output)
        return nn.Sigmoid(output)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        