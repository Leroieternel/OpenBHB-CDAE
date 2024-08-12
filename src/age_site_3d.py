import numpy as np
import torch
import os
import models
import argparse
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchvision import datasets
from data import FeatureExtractor, OpenBHB, bin_age
from data.transforms import Crop, Pad, Cutout
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
from util_3d import compute_age_mae, compute_site_ba
from models.wi_net import Wi_Net
from models.age_net import Age_Net

# parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Weakly contrastive learning for brain age predictin",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tf', type=str, help='data augmentation', choices=['none', 'crop', 'cutout', 'all'], default='none')
    parser.add_argument('--data_dir', type=str, help='path of data dir', default='/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data')
    parser.add_argument('--bs', type=int, help='num. of subjects per batch', default=2)
    parser.add_argument('--sps', type=int, help='Slices per subject', default=1)
    parser.add_argument('--n_views', type=int, help='num. of multiviews', default=1)
    parser.add_argument('--device', type=str, help='torch device', default='cuda')

    opts = parser.parse_args()
    return opts

# data transformation
def get_transforms(opts):
    selector = FeatureExtractor("quasiraw")   # selector: FeatureExtractor(dtype='quasiraw')

    if opts.tf == 'none':
        aug = transforms.Lambda(lambda x: x)

    elif opts.tf == 'crop':
        aug = transforms.Compose([
            Crop((1, 121, 128, 121), type="random"),
            Pad((1, 128, 128, 128))
        ])  

    elif opts.tf == 'cutout':
        aug = Cutout(patch_size=[1, 32, 32, 32], probability=0.5)

    elif opts.tf == 'all':
        aug = transforms.Compose([
            Cutout(patch_size=[1, 32, 32, 32], probability=0.5),
            Crop((1, 121, 128, 121), type="random"),
            Pad((1, 128, 128, 128))
        ])
    
    T_pre = transforms.Lambda(lambda x: selector.transform(x))    # T_pre: lambda object
    
    T_train = transforms.Compose([
        T_pre,
        aug,
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        # transforms.Normalize(mean=0.0, std=1.0)
        transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-7))
    ])

    T_val = transforms.Compose([
        T_pre,
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        # transforms.Normalize(mean=0.0, std=1.0)
        transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-7))
    ])

    return T_train, T_val

# load data
def load_data(opts):
    T_train, T_test = get_transforms(opts)
    T_train = NViewTransform(T_train, opts.n_views)
    T_test = NViewTransform(T_test, opts.n_views)

    train_dataset = OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.bs, shuffle=True, num_workers=3,
                                               persistent_workers=True, drop_last=True)
    train_loader_score = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train),
                                                     batch_size=opts.bs, shuffle=True, num_workers=3,
                                                     persistent_workers=True)
    test_internal = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=True, transform=T_test), 
                                                batch_size=opts.bs, shuffle=False, num_workers=3,
                                                persistent_workers=True, drop_last=True)
    test_external = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=False, transform=T_test), 
                                                batch_size=opts.bs, shuffle=False, num_workers=3,
                                                persistent_workers=True, drop_last=True)
    return train_loader, train_loader_score, test_internal, test_external


if __name__ == "__main__":
    print('hi')
    opts = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, train_loader_score, test_loader_int, test_loader_ext = load_data(opts)
    
    # load the network, single channel, 1 class
    model_encoder = models.AlexNet3D_Encoder_1(num_classes=1).to(device)
    # load model parameters
    checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/dae_0616_2_epoch50.pth', map_location='cuda')

    model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])  #
    model_encoder = model_encoder.to(device)
    model_encoder.eval()
    

    mae_train, mae_int, mae_ext = compute_age_mae(model_encoder, train_loader_score, test_loader_int, test_loader_ext, opts)
    print("Age MAE:", mae_train, mae_int, mae_ext)
    # compute_site_ba(model_encoder, train_loader_score, test_loader_int, test_loader_ext, opts)
    ba_train, ba_int = compute_site_ba(model_encoder, train_loader_score, test_loader_int, test_loader_ext, opts)
    print("Site BA:", ba_train, ba_int)  
    print("Age MAE:", mae_train, mae_int, mae_ext) 
    challenge_metric = ba_int**0.3 * mae_ext + (1 / 64) ** 0.3 * mae_int
    print("Challenge score", challenge_metric)

