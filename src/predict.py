import glob
import numpy as np
import torch
import os
import models
import matplotlib.pyplot as plt
# from PIL import Image

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the network, single channel, 1 class
    net = models.UNet(n_channels=1, n_classes=1)
    optimizer = torch.optim.RMSprop(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 

    # cpy unet to device
    net.to(device=device)
    # load model parameters
    checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/unet_50_0134_mse_bs4_0410.pth', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # test
    # net.eval()
    np.set_printoptions(threshold=np.inf)


    test_img = np.load('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_img.npy')
    plt.figure()
    plt.imshow(test_img[0], cmap="gray")
    # np.save('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_pred.npy', input_numpy)
    plt.savefig('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_img.jpg')
    # image = Image.fromarray(test_img)
    # image.save('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_img.jpg')
    # train_sample = np.load('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/train_img_sample.npy')
    # print('test shape: ', test_img)
    test_img = np.expand_dims(test_img, axis=0)
    print('test shape: ', test_img.shape)
    test_img = torch.from_numpy(test_img).to(device=device, dtype=torch.float32)
        # predict
    net.eval()
    pred = net(test_img)
    print(pred.shape)
    pred_numpy = pred.cpu().detach().numpy()
    pred_numpy = pred_numpy.reshape(182, 218)

    print(pred_numpy.shape)
    # input_numpy = test_img.reshape(182, 218)
    # np.set_printoptions(threshold=np.inf)
    # print(pred_numpy)
    # pred_numpy_crop = pred_numpy[20: 160, 20: 196]
    # pred_numpy_crop = pred_numpy[2: 180, 2: 216]
    # pred_numpy_crop = pred_numpy
    # print(pred_numpy_crop.shape)
    # pred_numpy[pred_numpy >= 66] = 255
    # pred_numpy[pred_numpy < 66] = 0
    plt.figure()
    plt.imshow(pred_numpy, cmap="gray")
    # np.save('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_pred.npy', input_numpy)
    plt.savefig('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_mse_1_1.jpg')

