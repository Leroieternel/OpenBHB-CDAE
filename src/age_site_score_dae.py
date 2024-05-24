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
from util import compute_age_mae, compute_site_ba
from models.wi_net import Wi_Net

# parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Weakly contrastive learning for brain age predictin",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tf', type=str, help='data augmentation', choices=['none', 'crop', 'cutout', 'all'], default='none')
    parser.add_argument('--data_dir', type=str, help='path of data dir', default='/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data')
    parser.add_argument('--bs', type=int, help='num. of subjects per batch', default=4)
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
    # print('T train shape: ', T_train)     # Compose()
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
    model_encoder = models.UNet_Encoder(n_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, train_loader_score, test_loader_int, test_loader_ext = load_data(opts)
    
    
    # load the network, single channel, 1 class
    model_encoder = models.UNet_Encoder(n_channels=1)
    model_decoder = models.UNet_Decoder(n_channels=1, n_classes=1)
    wi_net = Wi_Net(input_dim=960, output_dim=64, dropout_rate=0.5)
    param_encoder = list(model_encoder.parameters())
    param_decoder = list(model_decoder.parameters())
    params = param_encoder + param_decoder

    optimizer = torch.optim.RMSprop(params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 

    # cpy unet to device
    model_encoder = model_encoder.to(device='cuda')
    wi_net = wi_net.to(device='cuda')

    # load model parameters
    
    # checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/cdae_500_mse_bs4_sps1_0510_balanced_4_8_5_3_4_2.pth', map_location=device)
    # checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0514_3.5_5_15_2_4_2.5_inh10.pth', map_location=device)
    # checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/dae_300_mse_bs4_sps1_0517_all_7_epoch150.pth', map_location=device)
    # checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0518_all_15_epoch100_inh.pth', map_location=device)
    
    checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/dae_300_mse_bs4_sps1_0517_all_7_epoch100_inh.pth', map_location=device)
    model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
    # model_decoder.load_state_dict(checkpoint['model_decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    wi_net.load_state_dict(checkpoint['wi_net_state_dict'])
    model_encoder.eval()
    wi_net.eval()
    
    # print('hi')
    # print(model_encoder.inc.double_conv[0].weight)
    # print('******************************')
    # for name, param in model_encoder.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    
    # mae_train, mae_int, mae_ext = compute_age_mae(model_encoder, train_loader_score, test_loader_int, test_loader_ext, opts)
    # print("Age MAE:", mae_train, mae_int, mae_ext)
    # compute_site_ba(model_encoder, train_loader_score, test_loader_int, test_loader_ext, opts)
    ba_train, ba_int = compute_site_ba(model_encoder, train_loader_score, test_loader_int, test_loader_ext, opts)
    print("Site BA:", ba_train, ba_int)  
    # print("Age MAE:", mae_train, mae_int, mae_ext) 
    # challenge_metric = ba_int**0.3 * mae_ext + (1 / 64) ** 0.3 * mae_int
    # print("Challenge score", challenge_metric)

