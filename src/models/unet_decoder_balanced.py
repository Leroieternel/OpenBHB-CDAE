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





class UNet_Decoder_b(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_Decoder_b, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        
        # self.inc = (DoubleConv(n_channels, 4))
        # self.down1 = (Down(4, 8))
        # self.down2 = (Down(8, 16))
        # self.down3 = (Down(16, 32))
        # self.down4 = (Down(32, 64))

        MULT = 2

        self.fc = nn.Linear(512, 64 * MULT * 11 * 13)
        factor = 2 if bilinear else 1
        self.up1 = (Up(64 * MULT, 32 * MULT // factor, bilinear))
        self.up2 = (Up(32 * MULT, 16 * MULT // factor, bilinear))
        self.up3 = (Up(16 * MULT, 8 * MULT // factor, bilinear))
        self.up4 = (Up(8 * MULT, 4 * MULT, bilinear))
        self.outc = (OutConv(4 * MULT, n_classes))



    def forward(self, x, en_x1, en_x2, en_x3, en_x4):
        # downsampling (for padding)
        # x_test = torch.randn(x.shape[0], 1, 182, 218)
        # x_test = x_test.to('cuda')
        # en_x1 = self.inc(x_test)
        # en_x2 = self.down1(en_x1)
        # en_x3 = self.down2(en_x2)
        # en_x4 = self.down3(en_x3)
        # en_x5 = self.down4(en_x4)
        # print('en x shape: ', x_test.shape)   # torch.Size([32, 1, 182, 218])
        # print('en x1 shape: ', en_x1.shape)   # torch.Size([4, 8, 182, 218])
        # print('en x2 shape: ', en_x2.shape)   # torch.Size([4, 16, 91, 109])
        # print('en x3 shape: ', en_x3.shape)   # torch.Size([4, 32, 45, 54])
        # print('en x4 shape: ', en_x4.shape)   # torch.Size([4, 64, 22, 27])
        # print('en x5 shape: ', en_x5.shape)   # torch.Size([4, 128, 11, 13])

        x1 = self.fc(x)
        x2 = x1.view(x1.shape[0], 64 * 2, 11, 13)     # torch.Size([4, 128, 11, 13])
        # print('up x2 shape: ', x2.shape)    # torch.Size([4, 128, 11, 13])
        # x3 = self.up1(x2, [22, 27])
        # x4 = self.up2(x3, [45, 54])
        # x5 = self.up3(x4, [91, 109])
        # x6 = self.up4(x5, [182, 218])

        x3 = self.up1(x2, en_x4)             
        x4 = self.up2(x3, en_x3)
        x5 = self.up3(x4, en_x2)
        x6 = self.up4(x5, en_x1)

        logits = self.outc(x6)
        # print('up x/fc shape: ', x.shape)   # torch.Size([32, 512])
        # print('up x1 shape: ', x1.shape)    # torch.Size([4, 18304])
        # print('up x2 shape: ', x2.shape)    # torch.Size([4, 128, 11, 13])
        # print('up x3 shape: ', x3.shape)    # torch.Size([4, 64, 22, 27])
        # print('up x4 shape: ', x4.shape)    # torch.Size([4, 32, 45, 54])
        # print('up x5 shape: ', x5.shape)    # torch.Size([4, 16, 91, 109])
        # print('up x6 shape: ', x6.shape)    # torch.Size([4, 8, 182, 218])
        # print('up logits shape: ', logits.shape)   # torch.Size([4, 1, 182, 218])

        return logits

    def use_checkpointing(self):
        # self.inc = torch.utils.checkpoint(self.inc)
        # self.down1 = torch.utils.checkpoint(self.down1)
        # self.down2 = torch.utils.checkpoint(self.down2)
        # self.down3 = torch.utils.checkpoint(self.down3)
        # self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)