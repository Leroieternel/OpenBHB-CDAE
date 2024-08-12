""" Full assembly of the parts to form the complete network """

# from unet_parts import *


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UNet_Encoder_b(nn.Module):
    def __init__(self, n_channels, bilinear=False, n_classes=5):
        super(UNet_Encoder_b, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_classes = n_classes
        # self.inc = (DoubleConv(n_channels, 8))
        # self.down1 = (Down(8, 16))
        # self.down2 = (Down(16, 32))
        # self.down3 = (Down(32, 64))
        # factor = 2 if bilinear else 1   # factor = 1
        # self.down4 = (Down(64, 128 // factor))
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(128*11*13, 512)


        MULT = 2
        self.inc = (DoubleConv(n_channels, 4 * MULT))
        self.down1 = (Down(4 * MULT, 8 * MULT))
        self.down2 = (Down(8 * MULT, 16 * MULT))
        self.down3 = (Down(16 * MULT, 32 * MULT))
        factor = 2 if bilinear else 1   # factor = 1
        self.down4 = (Down(32 * MULT, 64 * MULT // factor))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 11 * 13 * MULT, 512)

        # self.up1 = (Up(1024, 512 // factor, bilinear))
        # self.up2 = (Up(512, 256 // factor, bilinear))
        # self.up3 = (Up(256, 128 // factor, bilinear))
        # self.up4 = (Up(128, 64, bilinear))
        # self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # print('down x shape: ', x.shape)
        x1 = self.inc(x)
        # print('down x1 shape: ', x1.shape)  # torch.Size([4, 8, 182, 218])
        x2 = self.down1(x1)
        # print('down x2 shape: ', x2.shape)  # torch.Size([4, 16, 91, 109])
        x3 = self.down2(x2)
        # print('down x3 shape: ', x3.shape)  # torch.Size([4, 32, 45, 54])
        x4 = self.down3(x3)
        # print('down x4 shape: ', x4.shape)  # torch.Size([4, 64, 22, 27])
        x5 = self.down4(x4)
        # print('down x5 shape: ', x5.shape)  # torch.Size([4, 128, 11, 13])
        x_flatten = self.flatten(x5)
        # print('down x flatten shape: ', x_flatten.shape)     # torch.Size([4, 18304])
        x_encoder_out = self.fc(x_flatten)
        # print('down encoder output shape: ', x_encoder_out.shape)

        return x_encoder_out, x1, x2, x3, x4

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        # self.up1 = torch.utils.checkpoint(self.up1)
        # self.up2 = torch.utils.checkpoint(self.up2)
        # self.up3 = torch.utils.checkpoint(self.up3)
        # self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def features(self, x):
        features, *rest = self.forward(x)
        features_site_removal = features[:, self.n_classes: ]
        # print('features_site_removal shape: ', features_site_removal.shape)
        # print('features_site shape: ', features.shape)
        # return features_site_removal
        # return features
        return features[:, self.n_classes: ]
        # return features[:, : self.n_classes]
