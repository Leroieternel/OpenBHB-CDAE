import datetime
import math
import os
from random import gauss
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import argparse
import models
import losses
import time
import wandb
import torch.utils.tensorboard

from torch import nn
from torchvision import transforms
from torchvision import datasets
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
from util import compute_age_mae, compute_site_ba
from data import FeatureExtractor, OpenBHB, bin_age, OpenBHB_balanced_64
from data.transforms import Crop, Pad, Cutout
from tqdm import tqdm
from models.age_net import Age_Net
from utils.dice_score import dice_loss




def parse_arguments():
    parser = argparse.ArgumentParser(description="Weakly contrastive learning for brain age predictin",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Misc
    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--save_freq', type=int, help='save frequency', default=50)
    parser.add_argument('--data_dir', type=str, help='path of data dir', default='/data')
    parser.add_argument('--amp', type=arg2bool, help='use amp', default=False)
    parser.add_argument('--clip_grad', type=arg2bool, help='clip gradient to prevent nan', default=False)

    # Model
    parser.add_argument('--model', type=str, help='model architecture', default='unet')

    # Optimizer
    parser.add_argument('--epochs', type=int, help='number of epochs', default=300)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step'], default='step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='decay rate for learning rate (for step)')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="700,800,900")
    parser.add_argument('--lr_decay_step', type=int, help='decay rate step (overwrites lr_decay_epochs', default=10)
    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=5e-5)
    parser.add_argument('--gradient_clipping', type=int, help='max gradient norm', default=1.0)

    # Data
    # parser.add_argument('--train_all', type=arg2bool, help='train on all dataset including validation (int+ext)', default=True)
    parser.add_argument('--tf', type=str, help='data augmentation', choices=['none', 'crop', 'cutout', 'all'], default='all')
    
    # Loss 
    parser.add_argument('--delta_reduction', type=str, help='use mean or sum to reduce 3d delta mask (only for method=threshold)', default='sum')
    parser.add_argument('--temp', type=float, help='loss temperature', default=0.1)
    parser.add_argument('--alpha', type=float, help='infonce weight', default=1.)
    parser.add_argument('--n_views', type=int, help='num. of multiviews', default=1)
    parser.add_argument('--bs', type=int, help='num. of subjects per batch', default=4)
    parser.add_argument('--sps', type=int, help='Slices per subject', default=1)

    opts = parser.parse_args()

    if opts.lr_decay_step is not None:
        opts.lr_decay_epochs = list(range(opts.lr_decay_step, opts.epochs, opts.lr_decay_step))
        print(f"Computed decay epochs based on step ({opts.lr_decay_step}):", opts.lr_decay_epochs)
    else:
        iterations = opts.lr_decay_epochs.split(',')
        opts.lr_decay_epochs = list([])
        for it in iterations:
            opts.lr_decay_epochs.append(int(it))
        
    return opts


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

    T_test = transforms.Compose([
        T_pre,
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        # transforms.Normalize(mean=0.0, std=1.0)
        transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-7))
    ])

    return T_train, T_test


def load_data(opts):
    T_train, T_test = get_transforms(opts)
    # print('T train shape: ', T_train)     # Compose()
    T_train = NViewTransform(T_train, opts.n_views)

    # train_dataset = OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train, label=opts.label,
    #                         load_feats=None)
    train_dataset = OpenBHB_balanced_64(opts.data_dir, train=True, internal=True, transform=T_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.bs, shuffle=True, num_workers=3,
                                               persistent_workers=True, drop_last=True)
    # train_loader_score = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train),
    #                                                  batch_size=opts.bs, shuffle=True, num_workers=3,
    #                                                  persistent_workers=True, drop_last=True)
    test_internal = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=True, transform=T_test), 
                                                batch_size=opts.bs, shuffle=False, num_workers=3,
                                                persistent_workers=True, drop_last=True)
    test_external = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=False, transform=T_test), 
                                                batch_size=opts.bs, shuffle=False, num_workers=3,
                                                persistent_workers=True, drop_last=True)
    # return train_loader, train_loader_score, test_internal, test_external
    return train_loader, test_internal, test_external

def load_model(opts):
    if 'resnet' in opts.model:
        model = models.SupConResNet(opts.model, feat_dim=128)

    elif 'unet' in opts.model:
        model = models.UNet(n_channels=1, n_classes=1)
    
    else:
        raise ValueError("Unknown model", opts.model)

    if opts.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    model = model.to(opts.device)
   
    return model


def train(train_loader, model_encoder, model_decoder, model_age, optimizer, opts, epoch, train_loss, params):    
    '''
    load bs*sps slices per call
    '''
    
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    

    t1 = time.time()
    for idx, (images, ages, _) in enumerate(train_loader):   # labels: torch.Size([6])

        

        ''' transform dataloader images: [10, 1, 182, 218, 182] to [80, 1, 182, 218]
            pick 10 slices (last dimension) per subject
            one batch: 80 images, size: 182 x 218 each image
        '''
        ''' 
            to regress age:
            transform dataloader labels: (6, 2) to (6 * 8, 2)
            duplicate same labels x8 since different slices from same sample shares same label
        '''
        '''
            to generate images: labels are the same as imput images
        '''
        

        images = torch.cat(images, dim=0)  # images: torch.Size([10, 1, 182, 218, 182])
        images_numpy = images.numpy()   # (6, 1, 182, 218, 182)
        ages = ages.float()

        # randomly choose opts.sps slices per subject: (one batch: 10 subjects, each pick 8 slices)
        indices = np.random.choice(182, size=opts.bs * opts.sps, replace=True)    # randomly pick 80 slices from images
        X = []
        for i in range(opts.bs):   # 10 subjects
            print(indices[i * opts.sps: (i+1) * opts.sps])
            X.append(images_numpy[i, :, :, :, indices[i * opts.sps: (i+1) * opts.sps]]) 

        X = np.array(X)     # (6, 8, 1, 182, 218)
        X = X.reshape(opts.bs * opts.sps, *X.shape[2:])    # X: (80, 1, 182, 218)

        # print('transformed mean: ', X[24][0].mean())
        # np.save('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/train_img_sample.npy', X[24][0])

        torch_X = torch.from_numpy(X).to(opts.device)     # torch.Size([48, 1, 182, 218])
        true_masks = torch.from_numpy(X).to(opts.device) 
        torch_age = ages.repeat_interleave(opts.sps).to(opts.device) 


        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        print('torch_X: ', torch_X)
        print('torch age: ', torch_age)

        

        # start training
        with torch.cuda.amp.autocast():   # automatic mixed precision
            recon_criterion = nn.MSELoss()
        # with open('ffff', 'w') as f:
            z_i = model_encoder(torch_X)
            mask_pred = model_decoder(z_i)
            recon_loss = recon_criterion(mask_pred, true_masks.float())

            age_pred = model_age(z_i)
            age_loss = recon_criterion(age_pred, torch_age)
            
            print('recon loss: ', recon_loss)
            print('age loss: ', age_loss)

            if epoch <= 29:
                loss = 7 * recon_loss
            else:
                loss = 8 * recon_loss + 0.1 * age_loss.float()
      
            train_loss.append(loss.item())    

        
            optimizer.zero_grad()
            # grad_scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, opts.gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            print('backprop ok')

        if idx % 50 == 0:

            wandb.log({'epoch': epoch, 'loss': loss.item(), 'recon_loss:': recon_loss.item(), 'age_loss': age_loss.item()})
            wandb.log({"original_images": wandb.Image(torch_X[0]), "xi_a_hat_images": wandb.Image(mask_pred[0])})


    # return masks_pred

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    opts = parse_arguments()

    # train_loader, train_loader_score, test_loader_int, test_loader_ext = load_data(opts)
    train_loader, test_loader_int, test_loader_ext = load_data(opts)
    # model = load_model(opts)
    model_encoder = models.UNet_Encoder_1(n_channels=1)
    model_encoder = model_encoder.to(opts.device)
    model_decoder = models.UNet_Decoder_1(n_channels=1, n_classes=1)
    model_decoder = model_decoder.to(opts.device)
    model_age = Age_Net(input_dim=1024).float().to(opts.device)

    model_encoder.train()
    model_decoder.train()
    model_age.train()

    param_encoder = list(model_encoder.parameters())
    param_decoder = list(model_decoder.parameters())
    param_age_mlp = list(model_age.parameters())
    params = param_encoder + param_decoder + param_age_mlp

    optimizer = torch.optim.RMSprop(params, lr=opts.lr, weight_decay=opts.weight_decay, momentum=opts.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 

    print('Config:', opts)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)

    # if opts.amp:
    #     print("Using AMP")
    
    # start_time = time.time()
    # best_acc = 0.

    # training 
    train_loss = []
    wandb.init(project='openbhb_cdae_0602_ae_mlp_1', dir='/scratch_net/murgul/jiaxia/saved_models')
    config = wandb.config
    config.learning_rate = opts.lr
    wandb.watch(model_decoder, log="all")
    wandb.watch(model_encoder, log="all")
    wandb.watch(model_age, log="all")

    print('start training ')

    for epoch in range(0, opts.epochs):
        adjust_learning_rate(opts, optimizer, epoch)
        train(train_loader, model_encoder, model_decoder, model_age, optimizer, opts, epoch, train_loss, params)
        if epoch == 29:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/ae_0602_1_wo_epoch30.pth')
        if epoch == 49:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/ae_0602_1_wo_epoch50.pth')
        
        if epoch == 99:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/ae_0602_1_wo_epoch100.pth')
        if epoch == 149:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/ae_0602_1_wo_epoch150.pth')
        if epoch == 199:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/ae_0602_1_wo_epoch200.pth')
        if epoch == 249:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/ae_0602_1_wo_epoch250.pth')
        if epoch == 299:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/ae_0602_1_wo_epoch300.pth')

    print('train loss: ', train_loss)
    train_loss_numpy = np.array(train_loss)
    np.save('/scratch_net/murgul/jiaxia/saved_models/ae_0602_1_wo_loss.npy', train_loss_numpy)
    
    checkpoint = {
            'epoch': epoch,
            'model_encoder_state_dict': model_encoder.state_dict(),
            'model_decoder_state_dict': model_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }
    torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/ae_0602_1_wo.pth')


    

