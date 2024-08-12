import torch.nn as nn
import math

class AlexNet3D_Encoder(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNet3D_Encoder, self).__init__()

        # first convolution block
        # input shape: (bs, 1, 182, 218, 182)
        self.conv0 = nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0)  
        # self.conv1 = nn.Conv3d(8, 16, kernel_size=5, stride=1, padding=2)  # torch.Size([bs, 8, 182, 218, 182])
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=3)   

        # second convolution block
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0)  # (batch_size, 16, 60, 72, 60)
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)      


        # third convolution block
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)   # (batch_size, 32, 30, 36, 30)
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU(inplace=True)
        # self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)    

        # fourth convolution block
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)   # (batch_size, 64, 15, 18, 15)
        self.bn4 = nn.BatchNorm3d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.AdaptiveMaxPool3d(1)
        # self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)       

        # self.fc = nn.Linear(64 * 7 * 9 * 7, 1024)

        # mode selection
        # self.mode = mode
        # fc layer
        self.regressor = nn.Sequential(nn.Dropout(),
                                       nn.Linear(512, 64),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(),
                                       nn.Linear(64, num_classes)
                                       )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 通过每一层
        print('input encoder x shape: ', x.shape)
        x = self.conv0(x)
        # x = self.conv1(x)
        print('conv1 x shape: ', x.shape)  # torch.Size([4, 64, 89, 107, 89])
        x = self.bn1(x)
        x1 = self.relu1(x)
        print('x1 shape: ', x1.shape)    # torch.Size([4, 64, 89, 107, 89])
        x = self.pool1(x1)
        print('pool1 shape: ', x.shape)  # torch.Size([4, 64, 29, 35, 29])
        

        x = self.conv2(x)
        print('conv2 shape: ', x.shape)  # torch.Size([4, 128, 27, 33, 27])
        x = self.bn2(x)
        x2 = self.relu2(x)
        print('x2 shape: ', x2.shape)    # torch.Size([4, 128, 27, 33, 27])
        x = self.pool2(x2)               # torch.Size([4, 256, 9, 11, 9])
        

        x = self.conv3(x)
        print('conv3 shape: ', x.shape)  # torch.Size([4, 256, 9, 11, 9])
        x = self.bn3(x)
        x3 = self.relu3(x)
        print('x3 shape: ', x3.shape)    # torch.Size([4, 256, 9, 11, 9])
        # x = self.pool3(x3)
        

        x = self.conv4(x)
        print('conv4 shape: ', x.shape)  # torch.Size([4, 512, 9, 11, 9])
        x = self.bn4(x)
        x4 = self.relu4(x)
        print('x4 shape: ', x4.shape)    # torch.Size([4, 512, 9, 11, 9])
        x = self.pool4(x4)

        # print('x shape: ', x.shape)      # torch.Size([4, 64, 7, 9, 7])
        

        x = x.view(x.size(0), -1)  # Flatten
        print('encoder out shape: ', x.shape)    # torch.Size([4, 512])
        print('encoder out x: ', x)
        age_pred = self.regressor(x).squeeze()  # 通过全连接层
        # print('age pred shape: ', age_pred.shape)    # torch.Size([4])
        # if self.mode == "encoder":
            # x = self.fc(x)
            # print('encoder out x shape: ', x.shape)   # torch.Size([16, 1024])

        return age_pred, x, x1, x2, x3, x4 
