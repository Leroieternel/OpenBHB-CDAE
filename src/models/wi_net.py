import torch
import torch.nn as nn

class Wi_Net(nn.Module):
    def __init__(self, input_dim=448, output_dim=64, dropout_rate=0.5):

        super(Wi_Net, self).__init__()


        self.layers = nn.Sequential(
            # first linear + BN + ReLU + Dropout
            nn.Linear(input_dim, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # second linear+ BN + ReLU + Dropout
            nn.Linear(300, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # third linear + BN + ReLU + Dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # fourth linear + sigmoid
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        return self.layers(x)

