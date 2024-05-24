import torch

def correlation_loss(x_hat, x, eps=1e-8):
    # compute correlation
    x_hat_mean = torch.mean(x_hat, dim=(2, 3), keepdim=True)    # [bs*sps, 1]  
    print('x hat mean shape: ', x_hat_mean.shape)   # [bs*sps, 1]
    # print('x hat mean: ', x_hat_mean)
    x_mean = torch.mean(x, dim=(2, 3), keepdim=True)
    print('x mean shape: ', x_mean.shape)
    # print('x mean: ', x_mean)

    # print('x_hat - x_hat_mean: ', x_hat - x_hat_mean)      # torch.Size([bs*sps, 1, 182, 218])
    # print('x - x_mean: ', x - x_mean)
    cov = torch.mean((x_hat - x_hat_mean) * (x - x_mean), dim=(2, 3))
    print('cov: ', cov)  # torch.Size([4, 1])
    
    # compute standard deviation
    x_hat_std = torch.std(x_hat, dim=(2, 3), keepdim=True) + eps   # torch.Size([bs*sps, 1, 1, 1])
    print('x_hat_std shape: ', x_hat_std.shape)      # [32, 1]
    # print('x hat std: ', x_hat_std)   
    x_std = torch.std(x, dim=(2, 3), keepdim=True) + eps
    print('x std shape: ', x_std.shape)    # torch.Size([4, 1, 1, 1])
    # indices = (x_hat_std > 1e-3) & (x_std > 1e-3)

    x_hat_std = x_hat_std.reshape(4, 1)
    x_std = x_std.reshape(4, 1)
    print('x_hat_std.reshape: ', x_hat_std)
    print('x_std.reshape: ', x_std)

    use_mask = torch.ones_like(x_hat_std.flatten(), dtype=torch.bool)

    for i in range(x_std.shape[0]):
        # print('i: ', i)
        if x_hat_std[i] < 1e-3 or x_std[i] < 1e-3:
            cov[i] = 0
            use_mask[i] = False

    print('cov new: ', cov)
    cov, x_hat_std, x_std = cov[use_mask], x_hat_std[use_mask], x_std[use_mask]
    corr_loss = torch.mean(torch.abs(cov) / (x_hat_std * x_std))
    
    return corr_loss 