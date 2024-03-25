#%%

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
sys.path.append("/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/starting_kit")
from estimator1 import FeatureExtractor

# Need to specify where to find the brain and GM masks using env variables.
os.environ["VBM_MASK"] = "/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"
os.environ["QUASIRAW_MASK"] = "/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/quasiraw_space-MNI152_desc-brain_T1w.nii.gz"


X_flat = np.zeros((1, 2512678), dtype=np.single)
X = FeatureExtractor(dtype="quasiraw").transform(X_flat)
print("Shape of selected features", X.shape)
# %%

from problem import get_train_data, DatasetHelper, get_test_data

train_dataset = DatasetHelper(data_loader=get_test_data)
X_train, y_train = train_dataset.get_data()   # (3227, 3659572) (3227, 2)
df_labels_train = train_dataset.labels_to_dataframe()
print('y train shape: ',y_train.shape)

#%%

print(df_labels_train.head())
print(df_labels_train.describe(include=(float, )))

#%%

import matplotlib.pyplot as plt

BS = 10 # Dataloader
SPS = 8 # Slices per subject
dtype = "quasiraw"
# all_X = FeatureExtractor(dtype=dtype).transform(X_train)
temp_X = FeatureExtractor(dtype=dtype)   # temp x shape:  FeatureExtractor(dtype='quasiraw')
# print('temp x shape: ', temp_X)
batch_X = FeatureExtractor(dtype=dtype).transform(X_train[:BS])    # (10, 1, 182, 218, 182)
# print('all X train shape: ', all_X.shape)    
print("Shape of selected features:", batch_X.shape)   # (10, 1, 182, 218, 182): 10 subjects
# print(train_dataset.get_channels_info(batch_X))   # 0      T1w
#train_dataset.plot_data(batch_X, sample_id=0, channel_id=0)

# Randomly select BS*SPS indices out of 182 with repetition: e.g. choose 80 samples
indices = np.random.choice(182, size=BS*SPS, replace=True)
print('selected indices: ', indices)

# X: samples trained per batch
X = []
for i in range(BS):   # 10 subjects
      print(indices[i*SPS:(i+1)*SPS])
      X.append(batch_X[i, :, :, :, indices[i*SPS:(i+1)*SPS]]) 
      # print(batch_X[i, :, :, :, indices[i*SPS:(i+1)*SPS]].shape)  # (8, 1, 182, 218)

X = np.array(X)
print(X.shape)
X = X.reshape(BS*SPS, *X.shape[2:])
print('one batch shape: ', X.shape) 


#%%
print('one image shape: ', X[38][0].shape) 

plt.figure()

plt.imshow(X[43][0], cmap="gray")
x_test = np.expand_dims(X[43][0], axis=0)
print(x_test.shape)
np.save('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/test_img.npy', x_test)



# # %%
# mask_pred = np.load('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/masks_shape.npy')
# print(mask_pred.shape)
# plt.figure()
# plt.imshow(mask_pred[2][0], cmap="gray")
# %%

