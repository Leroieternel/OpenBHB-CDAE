import torch
import torch.nn as nn
import torch.nn.functional as F

class Wi_Net_3d(nn.Module):
    def __init__(self, input_dim=448, output_dim=64, dropout_rate=0.5):

        super(Wi_Net_3d, self).__init__()


        self.layers = nn.Sequential(
            # first linear + BN + ReLU + Dropout
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            # second linear + BN + ReLU + Dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            # third linear
            nn.Linear(128, output_dim),

        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                layer.bias.data.fill_(0)

    def forward(self, x):
        
        return self.layers(x)

