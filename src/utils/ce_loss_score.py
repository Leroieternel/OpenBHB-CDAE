import torch
import torch.nn.functional as F

def cross_entropy_with_probs(inputs, target_probs):
    """
    Compute cross entropy loss between inputs (logits) and target probabilities.
    
    :param inputs: Tensor of shape (N, C) where C is number of classes.
    :param target_probs: Tensor of same shape as inputs containing probabilities for each class.
    :return: Scalar tensor representing the cross entropy loss.
    """
    # Applying softmax to convert logits into probabilities
    log_input_probs = F.log_softmax(inputs, dim=1)
    
    # Calculating cross entropy loss
    cross_entropy_loss = -torch.sum(target_probs * log_input_probs, dim=1)
    
    # Mean loss across all examples in the batch
    mean_loss = torch.mean(cross_entropy_loss)
    
    return mean_loss
