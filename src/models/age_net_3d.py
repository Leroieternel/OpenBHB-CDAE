import torch
import torch.nn as nn

class Age_Net_3d(nn.Module):
    def __init__(self, input_dim=448, output_dim=1):

        super(Age_Net_3d, self).__init__()

        
        self.layers = nn.Sequential(

            # first linear+ BN + ReLU + Dropout
            # nn.Linear(input_dim, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),

            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),

            # second linear + BN + ReLU + Dropout
            nn.Dropout(),
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # third linear + sigmoid
            nn.Linear(64, output_dim)            
        )

    def forward(self, x):
        # print('encoder output features: ', x)
        x = self.layers(x)
        return x.squeeze()

