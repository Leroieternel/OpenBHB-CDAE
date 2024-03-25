import glob
import numpy as np
import torch
import os
import models
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = models.UNet(n_channels=1, n_classes=1)
    optimizer = torch.optim.RMSprop(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 

    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/unet_30_1am_0325.pth', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 测试模式
    # net.eval()
    test_img = np.load('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_img.npy')
    print('test shape: ', test_img.shape)
    test_img = np.expand_dims(test_img, axis=0)
    print('test shape: ', test_img.shape)
    test_img = torch.from_numpy(test_img).to(device=device, dtype=torch.float32)
        # 预测
    net.eval()
    pred = net(test_img)
    print(pred.shape)
    pred_numpy = pred.cpu().detach().numpy()
    pred_numpy = pred_numpy.reshape(182, 218)

    print(pred_numpy)


    # pred_numpy[pred_numpy >= 0.15] = 255
    # pred_numpy[pred_numpy < 0.15] = 0
    plt.figure()
    plt.imshow(pred_numpy, cmap="gray")
    plt.savefig('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_mse.jpg')
