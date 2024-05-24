import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



# 读取.npy文件
# 2  4  0  3  2  2  0  1
# 10 24 3  17 10 10 3  1
data = np.load('/scratch_net/murgul/jiaxia/saved_models/train_image_3_1.npy')
data1 = np.load('/scratch_net/murgul/jiaxia/saved_models/train_image_5_1.npy')
data2 = np.load('/scratch_net/murgul/jiaxia/saved_models/train_image_1_1.npy')
data3 = np.load('/scratch_net/murgul/jiaxia/saved_models/train_image_4_1.npy')
data4 = np.load('/scratch_net/murgul/jiaxia/saved_models/train_image_3_2.npy')
data5 = np.load('/scratch_net/murgul/jiaxia/saved_models/train_image_3_2_1.npy')
data6 = np.load('/scratch_net/murgul/jiaxia/saved_models/train_image_1_2.npy')
data7 = np.load('/scratch_net/murgul/jiaxia/saved_models/train_image_2_2.npy')
data8 = np.load('/scratch_net/murgul/jiaxia/saved_models/train_image_4_2.npy')
print('data shape: ', data4.shape)
# 确保数据在合理的像素范围内，例如0-255
# 如果数据不在这个范围内，你可能需要根据你的数据调整这一步
# data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
# data = data.astype(np.uint8)
# mapping = {3: 0, 1: 1, 10: 2, 17: 3, 24: 4}
# plt.imsave('/scratch_net/murgul/jiaxia/saved_models/train_image_site_10_1.jpg', data[0], cmap='gray')
# plt.imsave('/scratch_net/murgul/jiaxia/saved_models/train_image_site_24.jpg', data1[0], cmap='gray')
# plt.imsave('/scratch_net/murgul/jiaxia/saved_models/train_image_site_3_1.jpg', data2[0], cmap='gray')
# plt.imsave('/scratch_net/murgul/jiaxia/saved_models/train_image_site_17.jpg', data3[0], cmap='gray')
# plt.imsave('/scratch_net/murgul/jiaxia/saved_models/train_image_site_10_2.jpg', data4[0], cmap='gray')
# plt.imsave('/scratch_net/murgul/jiaxia/saved_models/train_image_site_10_2.jpg', data5[0], cmap='gray')
# plt.imsave('/scratch_net/murgul/jiaxia/saved_models/train_image_site_3_2.jpg', data6[0], cmap='gray')
# plt.imsave('/scratch_net/murgul/jiaxia/saved_models/train_image_site_1.jpg', data7[0], cmap='gray')
plt.imsave('/scratch_net/murgul/jiaxia/saved_models/train_image_site_17_2.jpg', data8[0], cmap='gray')


# 或者使用PIL保存图像
# image = Image.fromarray(data[0])
# image.save('/scratch_net/murgul/jiaxia/saved_models/train_image_3.jpg')
