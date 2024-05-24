import pandas as pd
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import torch
from data import OpenBHB, bin_age, FeatureExtractor_balanced, OpenBHB_balanced
from nilearn.masking import unmask
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from collections import OrderedDict
from torchvision import transforms
from data.transforms import Crop, Pad, Cutout
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
import os


# 读取 .tsv 文件
file_path = '/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data/train.tsv'
data = pd.read_csv(file_path, sep='\t')
# site_count = []
# 筛选 'site' 列中值为 3 的行
# for i in range(0, 64):
#     site_count.append(data[data['site'] == i].shape[0])

x_arr = np.load('/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data/train.npy', mmap_mode="r")
y_arr = data[["age", "site"]].values
site_21 = [21]
filtered_data_21 = data[data['site'].isin(site_21)]
print('filtered_data_21: ', filtered_data_21)

site_21_img = x_arr[1931, 1145]
print('x arr shape: ', x_arr.shape)
print('site_21_img shape: ', site_21_img.shape)

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """ Select only the requested data associatedd features from the the
    input buffered data.
    """
    MODALITIES = OrderedDict([
        ("vbm", {
            "shape": (1, 121, 145, 121),
            "size": 519945}),
        ("quasiraw", {
            "shape": (1, 182, 218, 182),
            "size": 1827095}),
        ("xhemi", {
            "shape": (8, 163842),
            "size": 1310736}),
        ("vbm_roi", {
            "shape": (1, 284),
            "size": 284}),
        ("desikan_roi", {
            "shape": (7, 68),
            "size": 476}),
        ("destrieux_roi", {
            "shape": (7, 148),
            "size": 1036})
    ])
    MASKS = {
        "vbm": {
            "path": None,
            "thr": 0.05},
        "quasiraw": {
            "path": None,
            "thr": 0}
    }

    def __init__(self, dtype, mock=False):
        """ Init class.
        Parameters
        ----------
        dtype: str
            the requested data: 'vbm', 'quasiraw', 'vbm_roi', 'desikan_roi',
            'destrieux_roi' or 'xhemi'.
        """

        if dtype not in self.MODALITIES:
            raise ValueError("Invalid input data type.")
        self.dtype = dtype

        data_types = list(self.MODALITIES.keys())
        index = data_types.index(dtype)
        
        cumsum = np.cumsum([item["size"] for item in self.MODALITIES.values()])
        
        if index > 0:
            self.start = cumsum[index - 1]
        else:
            self.start = 0
        self.stop = cumsum[index]
        
        self.masks = dict((key, val["path"]) for key, val in self.MASKS.items())
        self.masks["vbm"] = "./data/masks/cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"
        self.masks["quasiraw"] = "./data/masks/quasiraw_space-MNI152_desc-brain_T1w.nii.gz"

        self.mock = mock
        if mock:
            return

        for key in self.masks:
            if self.masks[key] is None or not os.path.isfile(self.masks[key]):
                raise ValueError("Impossible to find mask:", key, self.masks[key])
            arr = nibabel.load(self.masks[key]).get_fdata()
            thr = self.MASKS[key]["thr"]
            arr[arr <= thr] = 0
            arr[arr > thr] = 1
            self.masks[key] = nibabel.Nifti1Image(arr.astype(np.int32), np.eye(4))

    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.mock:
            print("transforming", X.shape)
            data = X.reshape(self.MODALITIES[self.dtype]["shape"])
            #print("mock data:", data.shape)
            return data
        
        # print(X.shape)
        select_X = X[self.start:self.stop]
        if self.dtype in ("vbm", "quasiraw"):
            im = unmask(select_X, self.masks[self.dtype])
            select_X = im.get_fdata()
            # select_X = select_X.transpose(2, 0, 1)
        select_X = select_X.reshape(self.MODALITIES[self.dtype]["shape"])
        print('transformed.shape', select_X.shape)
        return select_X

def get_transforms(site_21_img):
    selector = FeatureExtractor("quasiraw")  # selector: FeatureExtractor(dtype='quasiraw')

    # if opts.tf == 'none':
    #     aug = transforms.Lambda(lambda x: x)

    # elif opts.tf == 'crop':
    #     aug = transforms.Compose([
    #         Crop((1, 121, 128, 121), type="random"),
    #         Pad((1, 128, 128, 128))
    #     ])  

    # elif opts.tf == 'cutout':
    #     aug = Cutout(patch_size=[1, 32, 32, 32], probability=0.5)

    # elif opts.tf == 'all':
    # aug = transforms.Compose([
    #     Cutout(patch_size=[1, 32, 32, 32], probability=0.5),
    #     Crop((1, 121, 128, 121), type="random"),
    #     Pad((1, 128, 128, 128))
    # ])
    
    # T_pre = transforms.Lambda(lambda x: selector.transform(x))    # T_pre: lambda object
    T_pre = selector.transform(site_21_img)
    
    # T_train = transforms.Compose([
    #     T_pre,
    #     # aug,
    #     transforms.Lambda(lambda x: torch.from_numpy(x).float()),
    #     # transforms.Normalize(mean=0.0, std=1.0)
    #     transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-7))
    # ])

    # T_test = transforms.Compose([
    #     T_pre,
    #     transforms.Lambda(lambda x: torch.from_numpy(x).float()),
    #     # transforms.Normalize(mean=0.0, std=1.0)
    #     transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-7))
    # ])

    return T_pre

T_train = get_transforms(site_21_img)
print('T train shape: ', T_train.shape)
# T_train = NViewTransform(T_train, 1)
# # train_dataset = OpenBHB('/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data', train=True, internal=True, transform=T_train)
# train_loader = torch.utils.data.DataLoader(T_train, batch_size=1, shuffle=True, num_workers=1,
#                                                persistent_workers=True, drop_last=True)
# for idx, (images, ages, sites) in enumerate(train_loader):
#     print('single train image shape: ', images[0].shape)  

print('T train shape: ', T_train)     # Compose()

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=3,
#                                                persistent_workers=True, drop_last=True)

# T_train = NViewTransform(T_train, 1)

# plt.plot(site_count)

# # 添加标题和标签
# # plt.title('Line Graph of Data')
# plt.xlabel('site number')
# plt.ylabel('counts')

# # 显示图表
# plt.show()
# plt.savefig('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/site_distribution.jpg')