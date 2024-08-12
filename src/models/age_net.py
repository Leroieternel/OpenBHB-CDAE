import torch
import torch.nn as nn

class Age_Net(nn.Module):
    def __init__(self, input_dim=960, output_dim=1, dropout_rate=0.5):

        super(Age_Net, self).__init__()

        
        self.layers = nn.Sequential(
            # # # first linear + BN + ReLU + Dropout

            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # second linear+ BN + ReLU + Dropout
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # third linear + BN + ReLU + Dropout
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # fourth linear + sigmoid
            nn.Linear(512, output_dim),
            nn.Identity()
        

            # nn.Linear(input_dim, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),

            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),

            # nn.Linear(128, output_dim),
            # nn.Identity()
        
        )
        
        # self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # print('encoder output features: ', x)
        
        return self.layers(x)
        # return self.layer(x)

