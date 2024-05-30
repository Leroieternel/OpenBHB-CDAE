import torch
import torch.nn as nn

class Age_Net_balanced(nn.Module):
    def __init__(self, input_dim=507, output_dim=1, dropout_rate=0.5):

        super(Age_Net_balanced, self).__init__()

        
        self.layers = nn.Sequential(
            # first linear + BN + ReLU + Dropout
            nn.Linear(input_dim, 256),
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
            nn.Identity()
        
        
        )
        
        # self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # print('encoder output features: ', x)
        
        return self.layers(x)
        # return self.layer(x)

