import torch
import torch.nn.functional as F

def custom_cross_entropy_with_probs(inputs, target_probs):
    """
    Compute cross entropy loss between inputs (logits) and target probabilities.
    
    :param inputs: Tensor of shape (N, C) where C is number of classes.
    :param target_probs: Tensor of same shape as inputs containing probabilities for each class.
    :return: Scalar tensor representing the cross entropy loss.
    """
    # Applying softmax to convert logits into probabilities
    input_probs = F.softmax(inputs, dim=1)
    
    # Calculating the log of probabilities
    log_input_probs = torch.log(input_probs + 1e-6)  # Adding a small constant to prevent log(0)
    
    # Calculating cross entropy loss
    cross_entropy_loss = -torch.sum(target_probs * log_input_probs, dim=1)
    
    # Mean loss across all examples in the batch
    mean_loss = torch.mean(cross_entropy_loss)
    
    return mean_loss

# Example usage
N, C = 4, 5  # N is batch size, C is number of classes
logits = torch.randn(N, C, requires_grad=True)  # Example logits from some model
target_probs = torch.full((N, C), 1.0 / C)  # Uniform probability distribution as target

# Calculating the loss
loss = custom_cross_entropy_with_probs(logits, target_probs)

# Perform backward pass to compute gradients
loss.backward()

print("Calculated Loss:", loss.item())
