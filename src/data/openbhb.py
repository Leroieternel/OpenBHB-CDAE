import numpy as np
import os
import nibabel
import torch
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from collections import OrderedDict
from nilearn.masking import unmask

def bin_age(age_real: torch.Tensor):
    bins = [i for i in range(4, 92, 2)]
    age_binned = age_real.clone()
    # bins[::-1]: [90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 
    # 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]
    for value in bins[::-1]:    
        age_binned[age_real <= value] = value
    return age_binned.long()

def read_data(path, dataset, fast):    # read train: (395, 3659572)  internal val: (362, 2)  external val: (395, 2)
    print(f"Read {dataset.upper()}")
    df = pd.read_csv(os.path.join(path, dataset + ".tsv"), sep="\t")
    # df = pd.read_csv(os.path.join(path, "test" + ".tsv"), sep="\t")
    df.loc[df["split"] == "external_test", "site"] = np.nan

    y_arr = df[["age", "site"]].values

    # x_arr = np.zeros((10, 3659572))
    if not fast:
        x_arr = np.load(os.path.join(path, dataset + ".npy"), mmap_mode="r")
    
    print("- y size [original]:", y_arr.shape)
    print("- x size [original]:", x_arr.shape)
    return x_arr, y_arr

class OpenBHB(torch.utils.data.Dataset):
    # def __init__(self, root, train=True, internal=True, transform=None, fast=False, load_feats=None):
    def __init__(self, root, train=True, internal=True, transform=None, fast=False):
        # print('###########################OpenBHB#########################')
        self.root = root

        if train and not internal:
            raise ValueError("Invalid configuration train=True and internal=False")
        
        self.train = train
        self.internal = internal
        
        dataset = "train"
        if not train:
            if internal:
                dataset = "internal_val"
            else:
                dataset = "external_val"
        
        self.X, self.y = read_data(root, dataset, fast)
        self.T = transform
        # self.label = label
        self.fast = fast

        self.bias_feats = None
        # if load_feats:
        #     print("Loading biased features", load_feats)
        #     self.bias_feats = torch.load(load_feats, map_location="cpu")
        
        print(f"Read {len(self.X)} records")      # total length of current reading samples

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):   
        # if not self.fast:
        #     x = self.X[index]
        # else:
        #     x = self.X[0]
        x = self.X[index]

        y = self.y[index]

        if self.T is not None:
            x = self.T(x)
        
        # sample, age, site
        age, site = y[0], y[1]
        # if self.label == "bin":
        #     age = bin_age(torch.tensor(age))
        
        if self.bias_feats is not None:
            return x, age, self.bias_feats[index]
        else:
            return x, age, site

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
        self.masks["vbm"] = "/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"
        self.masks["quasiraw"] = "/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/quasiraw_space-MNI152_desc-brain_T1w.nii.gz"

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
        # print('transformed.shape', select_X.shape)   # （1, 182, 218, 182）
        return select_X

'''
if __name__ == '__main__':
    import sys
    from torchvision import transforms
    # from .transforms import Crop, Pad

    selector = FeatureExtractor("quasiraw")

    T_pre = transforms.Lambda(lambda x: selector.transform(x))
    T_train = transforms.Compose([
        T_pre,
        # Crop((1, 121, 128, 121), type="random"),
        # Pad((1, 128, 128, 128)),
        transforms.Lambda(lambda x: torch.from_numpy(x)),
        transforms.Normalize(mean=0.0, std=1.0)
    ])

    train_loader = torch.utils.data.DataLoader(OpenBHB(sys.argv[1], train=True, internal=True, transform=T_train),
                                               batch_size=4, shuffle=True, num_workers=8,
                                               persistent_workers=True)
    
    x, y1, y2 = next(iter(train_loader))
    print(x.shape, y1, y2)
'''