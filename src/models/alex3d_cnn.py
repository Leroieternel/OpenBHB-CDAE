"""
Model implemented in https://doi.org/10.5281/zenodo.4309677 by Abrol et al., 2021
"""
from torch import nn
import math

class AlexNet3D_Dropout(nn.Module):
    def __init__(self, num_classes=1, mode="classifier"):
        """
        :param num_classes: int, number of classes
        :param mode:  "classifier" or "encoder" (returning 128-d vector)
        """
        super(AlexNet3D_Dropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            # nn.Conv3d(192, 192, kernel_size=3, padding=1),
            # nn.BatchNorm3d(192),
            # nn.ReLU(inplace=True),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d(1),
        )
        self.mode = mode

        self.regressor = nn.Sequential(nn.Dropout(),
                                       nn.Linear(512, 64),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(),
                                       nn.Linear(64, num_classes)
                                       )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        print('encoder out x: ', x)
        if self.mode == "encoder":
            return x
        x = self.regressor(x).squeeze()
        return x