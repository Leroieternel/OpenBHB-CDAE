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

'''  
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
'''

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # print('bilinear: ', bilinear)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print('ConvTranspose2d shape: ', x1.shape)    # torch.Size([4, 64, 22, 26])
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # diffY = shape[0] - x1.size()[2]
        # diffX = shape[1] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        # print('x shape: ', x.shape)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)





class UNet_Decoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_Decoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        MULT = 2

        self.fc = nn.Linear(1024, 64 * MULT * 11 * 13)
        factor = 2 if bilinear else 1
        self.up1 = (Up(64 * MULT, 32 * MULT // factor, bilinear))
        self.up2 = (Up(32 * MULT, 16 * MULT // factor, bilinear))
        self.up3 = (Up(16 * MULT, 8 * MULT // factor, bilinear))
        self.up4 = (Up(8 * MULT, 4 * MULT, bilinear))
        self.outc = (OutConv(4 * MULT, n_classes))



    def forward(self, x, en_x1, en_x2, en_x3, en_x4):


        x1 = self.fc(x)
        x2 = x1.view(x1.shape[0], 64 * 2, 11, 13)     # torch.Size([4, 128, 11, 13])

        x3 = self.up1(x2, en_x4)             
        x4 = self.up2(x3, en_x3)
        x5 = self.up3(x4, en_x2)
        x6 = self.up4(x5, en_x1)

        logits = self.outc(x6)

        return logits

    def use_checkpointing(self):
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)