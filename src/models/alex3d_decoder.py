import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet3D_Decoder(nn.Module):
    def __init__(self):
        super(AlexNet3D_Decoder, self).__init__()
        
        # self.fc = nn.Linear(1024, 64 * 7 * 9 * 7)  # 
        # 上采样和转置卷积层来逆向操作编码器的步骤
        self.up4 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)

        self.conv4 = nn.Conv3d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)  
        self.conv3 = nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU(inplace=True)

        self.up2 = nn.ConvTranspose3d(32 , 16, kernel_size=2, stride=2)  # stride 和 padding 要逆转池化层
        self.conv2 = nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU(inplace=True)

        self.up1 = nn.ConvTranspose3d(16, 8, kernel_size=5, stride=3)
        self.conv1 = nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv3d(8, 1, kernel_size=1)


    def pad_to_match(self, x, target):
        # 计算尺寸差异
        diff_depth = target.size()[2] - x.size()[2]
        diff_height = target.size()[3] - x.size()[3]
        diff_width = target.size()[4] - x.size()[4]

        # 应用填充
        x_padded = F.pad(x, [
            diff_width // 2, diff_width - diff_width // 2,
            diff_height // 2, diff_height - diff_height // 2,
            diff_depth // 2, diff_depth - diff_depth // 2
        ])

        return x_padded
    
    # x1: torch.Size([4, 8, 182, 218, 182])
    # x2: torch.Size([4, 16, 60, 72, 60])
    # x3: torch.Size([4, 32, 30, 36, 30])
    # x4: torch.Size([4, 64, 15, 18, 15])
    def forward(self, x, x1, x2, x3, x4):
        # 全连接层先将encoded features展开
        # x = self.fc(x)
        x = x.view(-1, 64, 4, 5, 4)     # torch.Size([4, 64, 7, 9, 7])
        # print('x fc shape: ', x.shape)    

        # 使用解码器的第一层，并与x3合并
        x_up1 = self.up4(x)
        
        # print('x_up4 shape: ', x_up1.shape)  # torch.Size([4, 32, 14, 18, 14])
        x = self.pad_to_match(x_up1, x4)     # torch.Size([4, 32, 15, 18, 15])
        # print('x matched shape: ', x.shape)
        x = torch.cat((x, x4), dim=1)
        # print('x cat1 shape: ', x.shape)     # torch.Size([16, 64, 15, 18, 15])
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        # print('conv4 shape: ', x.shape)      # torch.Size([16, 64, 15, 18, 15])
        

        # 接着处理x2合并
        x = self.up3(x)
        # print('x_up3 shape: ', x.shape)      # torch.Size([4, 32, 30, 36, 30])
        x = self.pad_to_match(x, x3)
        x = torch.cat((x, x3), dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)    
        # print('conv3 shape: ', x.shape)      # torch.Size([4, 32, 30, 36, 30])
        # 接着处理x1合并
        x = self.up2(x)
        
        # print('x_up2 shape: ', x.shape)      # torch.Size([4, 16, 60, 72, 60])
        x = self.pad_to_match(x, x2)
        x = torch.cat((x, x2), dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # print('conv2 shape: ', x.shape)      # torch.Size([4, 16, 60, 72, 60])

        # 
        x = self.up1(x)
        # print('x_up1 shape: ', x.shape)      # torch.Size([4, 8, 182, 218, 182])
        x = self.pad_to_match(x, x1)
        x = torch.cat((x, x1), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # print('conv1 shape: ', x.shape)      # torch.Size([4, 8, 182, 218, 182])

        # only change channels by 1x1 convolution kernel
        x = self.conv0(x)                    # torch.Size([4, 1, 182, 218, 182])
        

        # print('decoder output shape: ', x.shape)    # torch.Size([4, 1, 182, 218, 182])
        
        return x
