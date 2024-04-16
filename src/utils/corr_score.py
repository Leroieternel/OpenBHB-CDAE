import torch

def correlation_loss(x_hat, x, eps=1e-8):
    # compute correlation
    x_hat_mean = torch.mean(x_hat, dim=(2, 3), keepdim=True)    # [bs*sps, 1]  
    # print('x hat mean shape: ', x_hat_mean.shape)   # [32, 1]
    print('x hat mean: ', x_hat_mean)
    x_mean = torch.mean(x, dim=(2, 3), keepdim=True)
    # print('x mean shape: ', x_mean.shape)
    print('x mean: ', x_mean)
    # print('x hat shape: ', x_hat.shape)
    cov = torch.mean((x_hat - x_hat_mean) * (x - x_mean))
    print('cov: ', cov)
    
    # compute standard deviation
    x_hat_std = torch.std(x_hat, dim=(2, 3), keepdim=True) + eps   # [bs*sps, 1]
    # print('x_hat_std shape: ', x_hat_std.shape)      # [32, 1]
    print('x hat std: ', x_hat_std)
    x_std = torch.std(x, dim=(2, 3), keepdim=True) + eps
    print('x std: ', x_std)
    
    # compute correlation loss
    corr_loss = torch.sum(cov / (x_hat_std * x_std))
    
    return corr_loss 