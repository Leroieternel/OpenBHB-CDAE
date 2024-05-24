import glob
import numpy as np
import torch
import os
import models
import matplotlib.pyplot as plt
from torch import nn
# from PIL import Image

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the network, single channel, 1 class
    model_encoder = models.UNet_Encoder(n_channels=1)
    model_decoder = models.UNet_Decoder(n_channels=1, n_classes=1)
    param_encoder = list(model_encoder.parameters())
    param_decoder = list(model_decoder.parameters())
    params = param_encoder + param_decoder

    optimizer = torch.optim.RMSprop(params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 

    # cpy unet to device
    model_encoder = model_encoder.to(device=device)
    model_decoder = model_decoder.to(device=device)
    # load model parameters
    checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0514_all_3.5_5_15_2_4_2.5_epoch300.pth', map_location=device)
    model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
    model_decoder.load_state_dict(checkpoint['model_decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    

    # test
    # net.eval()
    np.set_printoptions(threshold=np.inf)


    '''
    # test_img = np.load('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_img.npy')
    test_img = np.load('/scratch_net/murgul/jiaxia/saved_models/test_image_0.npy')
    # plt.figure()
    # plt.imshow(test_img[0], cmap="gray")
    # np.save('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_pred.npy', input_numpy)
    # plt.savefig('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_img.jpg')
    # image = Image.fromarray(test_img)
    # image.save('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_img.jpg')
    # train_sample = np.load('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/train_img_sample.npy')
    # print('test shape: ', test_img)
    test_img = np.expand_dims(test_img, axis=0)
    print('test shape: ', test_img.shape)
    test_img = torch.from_numpy(test_img).to(device=device, dtype=torch.float32)
        # predict
    model_encoder.eval()
    model_decoder.eval()
    pred_1, torch_X_en_x1, torch_X_en_x2, torch_X_en_x3, torch_X_en_x4 = model_encoder(test_img)
    pred_2 = model_decoder(pred_1, torch_X_en_x1, torch_X_en_x2, torch_X_en_x3, torch_X_en_x4)
    pred_3, a, b, c, d = model_encoder(pred_2)
    pred_4 = model_decoder(pred_3, a, b, c, d)

    print(pred_4.shape)
    pred_numpy = pred_4.cpu().detach().numpy()
    pred_numpy = pred_numpy.reshape(182, 218)

    print(pred_numpy)
    # print('test_img[0][0]: ', test_img[0][0])
    criterion = nn.MSELoss()
    # mse_loss = criterion(torch.from_numpy(pred_numpy), test_img[0][0].cpu())
    # print('MSE loss: ', mse_loss)
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
    plt.savefig('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_image_predicted.jpg')
    '''

    # for name, param in model_encoder.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    model_encoder.eval().cpu()
    # print('hi')
    # print(model_encoder.inc.double_conv[0].weight)
    # for name, param in model_encoder.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    tensor = torch.rand(4, 1, 182, 218)
    test_img = np.load('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/torch_x_sample.npy')
    # test_img = np.expand_dims(test_img, axis=0)
    test_img = torch.from_numpy(test_img)
    print('test img: ', test_img)
    with torch.no_grad():
        output, *rest = model_encoder(test_img)
        print('original output: ', output[:, : 64])
        output_features = model_encoder.features(test_img)
        print('output features: ', output_features)

