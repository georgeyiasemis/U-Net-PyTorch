import torch
import torch.nn as nn


class Conv2d_Block(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, padding=0):
        super(Conv2d_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=3,
                               padding=padding)
        self.batchnorm1 = nn.BatchNorm2d(num_features=hidden_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=padding)

        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.batchnorm1, self.relu1,
                                self.conv2, self.batchnorm2, self.relu2)

    def forward(self, inp):
        return self.net(inp)

class UpConv2d_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv2d_Block, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=in_channels // 2,
                                         kernel_size=2,
                                         stride=2)

        self.conv = Conv2d_Block(in_channels=in_channels,
                                 hidden_channels=out_channels,
                                 out_channels=out_channels)



    def forward(self, inp1, inp2, crop=True):

        # Assumption inp1 has larger width and height than inp2.
        inp2 = self.upconv(inp2)

        _, _, H1, W1 = inp1.shape
        _, _, H2, W2 = inp2.shape
        if not crop:
            if (W1 - W2) % 2 == 0:
                padding_left, padding_right = (W1 - W2) // 2, (W1 - W2) // 2
            else:
                padding_left, padding_right = (W1 - W2) // 2, (W1 - W2) // 2 + 1
            if  (H1 - H2) % 2 == 0:
                padding_top, padding_bottom = (H1 - H2) // 2, (H1 - H2) // 2
            else:
                padding_top, padding_bottom = (H1 - H2) // 2, (H1 - H2) // 2 + 1
            inp2 = nn.functional.pad(input=inp2, pad=(padding_left, padding_right, padding_top, padding_bottom))
        else:
            if (W1 - W2) % 2 == 0:
                crop_left, crop_right = (W1 - W2) // 2, (W1 - W2) // 2
            else:
                crop_left, crop_right = (W1 - W2) // 2, (W1 - W2) // 2 - 1

            if  (H1 - H2) % 2 == 0:
                crop_top, crop_bottom = (H1 - H2) // 2, (H1 - H2) // 2
            else:
                crop_top, crop_bottom = (H1 - H2) // 2, (H1 - H2) // 2 - 1
            inp1 = inp1[:, :, crop_bottom: -crop_top, crop_left: - crop_right]

        inp = torch.cat((inp1, inp2), dim=1)
        inp = self.conv(inp)

        return inp

class MaxPool2d_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MaxPool2d_Block, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2)
        self.conv = Conv2d_Block(in_channels=in_channels,
                                 hidden_channels=out_channels,
                                 out_channels=out_channels)

    def forward(self, inp):

        inp = self.maxpool(inp)
        inp = self.conv(inp)
        return inp

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        '''
        Following implementation of
            "U-Net: Convolutional Networks for Biomedical Image Segmentation".
        '''
        super(UNet, self).__init__()

        self.conv_block_in = Conv2d_Block(in_channels, 64, 64)

        self.maxpool_block1 = MaxPool2d_Block(64, 128)
        self.maxpool_block2 = MaxPool2d_Block(128, 256)
        self.maxpool_block3 = MaxPool2d_Block(256, 512)
        self.maxpool_block4 = MaxPool2d_Block(512, 1024)

        self.upconv_block1 =  UpConv2d_Block(1024, 512)
        self.upconv_block2 =  UpConv2d_Block(512, 256)
        self.upconv_block3 =  UpConv2d_Block(256, 128)
        self.upconv_block4 =  UpConv2d_Block(128, 64)

        self.conv_out = nn.Conv2d(in_channels=64,
                                  out_channels=out_channels,
                                  kernel_size=1)

    def forward(self, inp):

        inp1 = self.conv_block_in(inp)

        inp2 = self.maxpool_block1(inp1)
        inp3 = self.maxpool_block2(inp2)
        inp4 = self.maxpool_block3(inp3)
        inp = self.maxpool_block4(inp4)

        inp = self.upconv_block1(inp4, inp)
        inp = self.upconv_block2(inp3, inp)
        inp = self.upconv_block3(inp2, inp)
        inp = self.upconv_block4(inp1, inp)

        inp = self.conv_out(inp)
        return inp
