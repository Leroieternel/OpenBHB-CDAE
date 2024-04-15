import numpy as np
import matplotlib.pyplot as plt
import torch

# Number of tensors
num_tensors = 32
# Length of each tensor
tensor_length = 64

# Initialize a list to hold the tensors
tensors = []

# Loop to create each tensor
for _ in range(num_tensors):
    # Create a tensor of zeros
    tensor = torch.zeros(tensor_length)
    # Randomly select an index
    rand_index = torch.randint(0, tensor_length, (1,))
    # Set the randomly selected index to 1
    tensor[rand_index] = 1
    # Add the tensor to the list
    tensors.append(tensor)

# Stack the list of tensors into a single tensor
print(len(tensors))
stacked_tensors = torch.stack(tensors)

print(stacked_tensors.shape) # Should print torch.Size([32, 64])
print(stacked_tensors[24]) # Should print torch.Size([32, 64])


tensor1 = torch.randn((32, 64))
tensor2 = torch.randn((32, 448))

# 使用torch.cat函数沿着特定维度拼接它们
concatenated_tensor = torch.cat((tensor1, tensor2), dim=1)

# 检查拼接后的张量形状，应该为(32, 512)
print(concatenated_tensor.shape)



# train_img = np.load('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/train_img_sample.npy')
# print('train image shape: ', train_img.shape)
# plt.figure()
# plt.imshow(train_img, cmap="gray")    
# plt.show()
# plt.savefig('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/train_img_sample.jpg')

