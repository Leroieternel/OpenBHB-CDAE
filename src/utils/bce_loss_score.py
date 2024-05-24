import torch


def bce_logits_loss(w_z, a):
    print('w_z: ', w_z)
    sigmoid_w_z = torch.sigmoid(w_z)
    # print('sigmoid shape: ', sigmoid_w_z.shape)    # torch.Size([4, 64])
    bce_loss_temp = torch.zeros((sigmoid_w_z.shape[0], ))
    # print('bce_loss_temp: ', bce_loss_temp)
    for i in range(sigmoid_w_z.shape[0]):
        # print('a[i]: ', a[i])
        # print('sigmoid_w_z[i] shape: ', sigmoid_w_z[i].shape)
        # print('log(sigmoid_w_z[i]): ', torch.log(sigmoid_w_z[i]))
        # print('torch.log(1 - sigmoid_w_z[i]): ', torch.log(1 - sigmoid_w_z[i]))
        # print('torch.dot(a[i], torch.log(sigmoid_w_z[i])): ', torch.dot(a[i], torch.log(sigmoid_w_z[i])))
        # print('torch.dot((1 - a[i]), torch.log(1 - sigmoid_w_z[i])): ', torch.dot((1 - a[i]), torch.log(1 - sigmoid_w_z[i])))
        bce_loss_temp[i] = -(torch.dot(a[i], torch.log(sigmoid_w_z[i])) + torch.dot((1 - a[i]), torch.log(1 - sigmoid_w_z[i])))
        bce_loss_temp[i] /= 64
    print('bce shape: ', bce_loss_temp.shape)
    print('bce: ', bce_loss_temp)
    # bce_loss_total = 0
    # for i in range(a.shape[0]):
    #     print('bce i: ', i)
    #     bce_loss_total += bce_loss_temp[i]
    # print('bce_loss_total shape: ', bce_loss_total.shape)
    
    bce_loss = torch.mean(bce_loss_temp, dim=0)
    
    return bce_loss