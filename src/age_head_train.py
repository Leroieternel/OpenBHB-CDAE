import numpy as np
import torch
import os
import time
import models
import argparse
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchvision import datasets
import wandb
from data import FeatureExtractor, OpenBHB, bin_age
from data.transforms import Crop, Pad, Cutout
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
from util import compute_age_mae, compute_site_ba
from models.wi_net import Wi_Net
from models.age_net import Age_Net

# parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Weakly contrastive learning for brain age predictin",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tf', type=str, help='data augmentation', choices=['none', 'crop', 'cutout', 'all'], default='none')
    parser.add_argument('--data_dir', type=str, help='path of data dir', default='/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data')
    parser.add_argument('--bs', type=int, help='num. of subjects per batch', default=4)
    parser.add_argument('--sps', type=int, help='Slices per subject', default=1)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=300)
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-4)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step'], default='step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='decay rate for learning rate (for step)')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="700,800,900")
    parser.add_argument('--lr_decay_step', type=int, help='decay rate step (overwrites lr_decay_epochs', default=10)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=5e-5)
    parser.add_argument('--gradient_clipping', type=int, help='max gradient norm', default=1.0)
    parser.add_argument('--amp', type=arg2bool, help='use amp', default=False)
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

class Age_regressor(nn.Module):
    def __init__(self):
        super(Age_regressor, self).__init__()
        self.encoder = models.UNet_Encoder(n_channels=1)  # initialize encoder
        self.mlp = Age_Net()  # initialize MLP

    def forward(self, x):
        x = self.encoder.features(x)  # forward encoder
        x = self.mlp(x)      # forward MLP
        return x
    
def load_checkpoint(model, optimizer, scheduler, checkpoint):
    # checkpoint = torch.load(filename, map_location=torch.device('cuda'))  
    model.encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
    # model.mlp.load_state_dict(checkpoint['model_decoder_state_dict']) 
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

def train(train_loader, model, optimizer, opts, epoch, train_loss, params):    
    '''
    load bs*sps slices per call
    '''
    # scaler = torch.cuda.amp.GradScaler() if opts.amp else None    # None
    model.train()
    # features = []
    # age_labels = []

    '''
    TRAIN for each batch
    '''
    idx = 0
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    
    for idx, (images, ages, _) in enumerate(train_loader): 
        images = torch.cat(images, dim=0)
        images_numpy = images.numpy() 
        indices = np.random.choice(182, size=opts.bs * opts.sps, replace=True)
        X = []
        if images.shape[0] == opts.bs:
            for i in range(opts.bs):   # opts.bs subjects
                # print(indices[i * opts.sps: (i+1) * opts.sps])
                # print('site label gt: ', sites[i])
                
                X.append(images_numpy[i, :, :, :, indices[i * opts.sps: (i+1) * opts.sps]]) 
                # sample_a = np.zeros(64, dtype=np.float32)   # sample_a: one hot site label


            # print('y site label: ', y_site)


            X = np.array(X)     # (6, 8, 1, 182, 218)
            X = X.reshape(opts.bs * opts.sps, *X.shape[2:])
            torch_X = torch.from_numpy(X).to(opts.device) 
            ages = torch.repeat_interleave(ages, opts.sps).to(opts.device)

            if idx == 1:
                print('input images shape: ', torch_X.shape)
                print('input images: ', torch_X)
            print('labels: ', ages)
            
            # forward propagation
            with torch.cuda.amp.autocast():
                feature = model(torch_X)
                print('predicted value: ', feature)
                print('feature shape: ', feature.shape)
                

                criterion = nn.MSELoss()
                loss = criterion(feature, ages.float())
                if epoch % 5 == 0:
                    print('loss: ', loss)

                optimizer.zero_grad()
                
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, opts.gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                print('backprop ok')

            if idx % 10 == 0:
                wandb.log({'epoch': epoch, 'loss': loss.item()})

if __name__ == "__main__":
    print('hi')
    opts = parse_arguments()
    wandb.init(project='openbhb_cdae_0520_age_training', dir='/scratch_net/murgul/jiaxia/saved_models')
    model_encoder = models.UNet_Encoder(n_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, train_loader_score, test_loader_int, test_loader_ext = load_data(opts)
    
    
    # load the network, single channel, 1 class
    model_encoder = models.UNet_Encoder(n_channels=1)
    model_decoder = models.UNet_Decoder(n_channels=1, n_classes=1)
    model_age_mlp = Age_Net()
    model = Age_regressor()
    param_encoder = list(model_encoder.parameters())
    param_decoder = list(model_decoder.parameters())
    param_mlp = list(model_age_mlp.parameters())
    params = param_encoder + param_decoder + param_mlp

    optimizer = torch.optim.RMSprop(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 

    # cpy unet to device
    # model_age_mlp = Age_regressor()
    # model_encoder = model_encoder.to(device='cuda')
    model = model.to(device='cuda')
    model.train()
    # load model parameters
    
    # checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/cdae_500_mse_bs4_sps1_0510_balanced_4_8_5_3_4_2.pth', map_location=device)
    checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0518_all_15_epoch100_inh.pth', map_location=device)
    load_checkpoint(model, optimizer, scheduler, checkpoint)

    train_loss = []
    for epoch in range(0, opts.epochs):
        # adjust_learning_rate(opts, optimizer, epoch)
        train(train_loader, model, optimizer, opts, epoch, train_loss, param_mlp)
        if epoch == 49:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model.encoder.state_dict(),
                'model_mlp_state_dict': model.mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_50_mse_bs4_sps1_0520_3_age_mlp_epoch50.pth')
        if epoch == 99:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model.encoder.state_dict(),
                'model_mlp_state_dict': model.mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0520_3_age_mlp_epoch100.pth')
        if epoch == 199:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model.encoder.state_dict(),
                'model_mlp_state_dict': model.mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_50_mse_bs4_sps1_0520_3_age_mlp_epoch200.pth')

    checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model.encoder.state_dict(),
                'model_mlp_state_dict': model.mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
    torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_50_mse_bs4_sps1_0520_3_age_mlp.pth')


