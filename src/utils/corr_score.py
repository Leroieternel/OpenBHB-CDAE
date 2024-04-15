import torch
from torch import Tensor

def correlation_loss(x_hat, x, eps=1e-8):
    # compute correlation
    x_hat_mean = torch.mean(x_hat, dim=0, keepdim=True)
    x_mean = torch.mean(x, dim=0, keepdim=True)
    cov = torch.mean((x_hat - x_hat_mean) * (x - x_mean), dim=0)
    
    # compute standard deviation
    x_hat_std = torch.std(x_hat, dim=0) + eps
    x_std = torch.std(x, dim=0) + eps
    
    # compute correlation loss
    corr_loss = torch.sum(cov / (x_hat_std * x_std))
    
    return -corr_loss  # Minimizing negative correlation is equivalent to maximizing positive correlation
