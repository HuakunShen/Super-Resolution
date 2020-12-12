import torch
from torch import nn
from PIL import Image
from torch.nn.modules import padding
from torchvision import transforms
import torch.nn.functional as F


class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=in_channels, out_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=out_channels, out_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=in_channels, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=mid_channel, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels,
                                     kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=in_channels, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=mid_channel, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel,
                            out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(
            in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        # Decode
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(
            bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(
            cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(
            cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return final_layer


def double_convolution(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=3//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=3//2),
        nn.ReLU(inplace=True),
    )


class UNetSR(nn.Module):
    def __init__(self, in_c: int = 3, out_c: int = 3):
        super(UNetSR, self).__init__()
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # encoder part
        self.encoder_conv_1 = double_convolution(in_c, 64)
        self.encoder_conv_2 = double_convolution(64, 128)
        self.encoder_conv_3 = double_convolution(128, 256)
        self.encoder_conv_4 = double_convolution(256, 512)
        self.encoder_conv_5 = double_convolution(512, 1024)
        # decoder part
        self.ConvT2D_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=1)
        self.decoder_conv_1 = double_convolution(1024, 512)
        self.ConvT2D_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=1)
        self.decoder_conv_2 = double_convolution(512, 256)
        self.ConvT2D_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decoder_conv_3 = double_convolution(256, 128)
        self.ConvT2D_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decoder_conv_4 = double_convolution(128, 64)
        # output layer to 3 channels
        self.final = nn.Conv2d(64, out_c, kernel_size=1)

    def forward(self, image):
        # desired_size = image.shape[-1] * self.factor
        # image = transforms.Compose([
        #     transforms.Resize(desired_size, Image.BICUBIC)
        # ])(image)

        x1 = self.encoder_conv_1(image)    # to be concatenated to decoder
        x2 = self.MaxPool2d(x1)
        # print(x2.shape)

        x3 = self.encoder_conv_2(x2)       # to be concatenated to decoder
        x4 = self.MaxPool2d(x3)
        # print(x4.shape)

        x5 = self.encoder_conv_3(x4)       # to be concatenated to decoder
        x6 = self.MaxPool2d(x5)
        # print(x6.shape)

        x7 = self.encoder_conv_4(x6)       # to be concatenated to decoder
        x8 = self.MaxPool2d(x7)
        # print(x8.shape)

        x9 = self.encoder_conv_5(x8)
        # print(x9.shape)

        x = self.ConvT2D_1(x9)
        # print(x.shape)

        x = self.decoder_conv_1(torch.cat([x, x7], 1))
        # print(x.shape)

        x = self.ConvT2D_2(x)
        # print(x.shape)

        x = self.decoder_conv_2(torch.cat([x, x5], 1))
        # print(x.shape)

        x = self.ConvT2D_3(x)
        x = self.decoder_conv_3(torch.cat([x, x3], 1))
        # print(x.shape)

        x = self.ConvT2D_4(x)
        # print(x.shape)

        x = self.decoder_conv_4(torch.cat([x, x1], 1))
        # print(x.shape)

        x = self.final(x)
        # print(x.shape)

        return x


class UNetNoTop(nn.Module):
    """
    remove top layer skip connection
    """

    def __init__(self, in_c: int = 3, out_c: int = 3):
        super(UNetNoTop, self).__init__()
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # encoder part
        self.encoder_conv_1 = double_convolution(in_c, 64)
        self.encoder_conv_2 = double_convolution(64, 128)
        self.encoder_conv_3 = double_convolution(128, 256)
        self.encoder_conv_4 = double_convolution(256, 512)
        self.encoder_conv_5 = double_convolution(512, 1024)
        # decoder part
        self.ConvT2D_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=1)
        self.decoder_conv_1 = double_convolution(1024, 512)
        self.ConvT2D_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=1)
        self.decoder_conv_2 = double_convolution(512, 256)
        self.ConvT2D_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decoder_conv_3 = double_convolution(256, 128)
        self.ConvT2D_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decoder_conv_4 = double_convolution(64, 64)
        # output layer to 3 channels
        self.final = nn.Conv2d(64, out_c, kernel_size=1)

    def forward(self, image):
        x1 = self.encoder_conv_1(image)    # to be concatenated to decoder
        x2 = self.MaxPool2d(x1)
        x3 = self.encoder_conv_2(x2)       # to be concatenated to decoder
        x4 = self.MaxPool2d(x3)
        x5 = self.encoder_conv_3(x4)       # to be concatenated to decoder
        x6 = self.MaxPool2d(x5)
        x7 = self.encoder_conv_4(x6)       # to be concatenated to decoder
        x8 = self.MaxPool2d(x7)
        x9 = self.encoder_conv_5(x8)
        x = self.ConvT2D_1(x9)
        x = self.decoder_conv_1(torch.cat([x, x7], 1))
        x = self.ConvT2D_2(x)
        x = self.decoder_conv_2(torch.cat([x, x5], 1))
        x = self.ConvT2D_3(x)
        x = self.decoder_conv_3(torch.cat([x, x3], 1))
        x = self.ConvT2D_4(x)
        # x = self.decoder_conv_4(torch.cat([x, x1], 1))
        x = self.decoder_conv_4(x)
        x = self.final(x)
        return x


class UNetD4(nn.Module):
    """
    UNet depth=4, instead of original 5 layers
    """

    def __init__(self, in_c: int = 3, out_c: int = 3):
        super(UNetD4, self).__init__()
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # encoder part
        self.encoder_conv_1 = double_convolution(in_c, 64)
        self.encoder_conv_2 = double_convolution(64, 128)
        self.encoder_conv_3 = double_convolution(128, 256)
        self.encoder_conv_4 = double_convolution(256, 512)
        self.encoder_conv_5 = double_convolution(512, 1024)
        # decoder part
        self.ConvT2D_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=1)
        self.decoder_conv_1 = double_convolution(1024, 512)
        self.ConvT2D_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=1)
        self.decoder_conv_2 = double_convolution(512, 256)
        self.ConvT2D_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decoder_conv_3 = double_convolution(256, 128)
        self.ConvT2D_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decoder_conv_4 = double_convolution(128, 64)
        # output layer to 3 channels
        self.final = nn.Conv2d(64, out_c, kernel_size=1)

    def forward(self, image):
        x1 = self.encoder_conv_1(image)    # to be concatenated to decoder
        x2 = self.MaxPool2d(x1)

        x3 = self.encoder_conv_2(x2)       # to be concatenated to decoder
        x4 = self.MaxPool2d(x3)

        x5 = self.encoder_conv_3(x4)       # to be concatenated to decoder
        x6 = self.MaxPool2d(x5)

        x7 = self.encoder_conv_4(x6)       # to be concatenated to decoder
        x = self.ConvT2D_2(x7)

        x = self.decoder_conv_2(torch.cat([x, x5], 1))

        x = self.ConvT2D_3(x)
        x = self.decoder_conv_3(torch.cat([x, x3], 1))

        x = self.ConvT2D_4(x)

        x = self.decoder_conv_4(torch.cat([x, x1], 1))

        x = self.final(x)
        return x


if __name__ == "__main__":
    image = torch.rand((1, 3, 300, 300))
    # print(image.size())
    # model = UNetSR()
    # model = UNetNoTop()
    model = UNetD4()
    out = model(image)
    print(out.shape)
