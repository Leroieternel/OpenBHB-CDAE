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
import torch.utils.tensorboard

from torch import nn
from torchvision import transforms
from torchvision import datasets
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
from util import compute_age_mae, compute_site_ba
from data import FeatureExtractor, OpenBHB, bin_age
from data.transforms import Crop, Pad, Cutout
from tqdm import tqdm
from utils.dice_score import dice_loss




def parse_arguments():
    parser = argparse.ArgumentParser(description="Weakly contrastive learning for brain age predictin",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Misc
    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
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
    parser.add_argument('--train_all', type=arg2bool, help='train on all dataset including validation (int+ext)', default=True)
    parser.add_argument('--tf', type=str, help='data augmentation', choices=['none', 'crop', 'cutout', 'all'], default='all')
    
    # Loss 
    parser.add_argument('--delta_reduction', type=str, help='use mean or sum to reduce 3d delta mask (only for method=threshold)', default='sum')
    parser.add_argument('--temp', type=float, help='loss temperature', default=0.1)
    parser.add_argument('--alpha', type=float, help='infonce weight', default=1.)
    parser.add_argument('--n_views', type=int, help='num. of multiviews', default=1)
    parser.add_argument('--bs', type=int, help='num. of subjects per batch', default=2)
    parser.add_argument('--sps', type=int, help='Slices per subject', default=8)

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
    selector = FeatureExtractor("quasiraw")
    
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
    
    T_pre = transforms.Lambda(lambda x: selector.transform(x))
    T_train = transforms.Compose([
        T_pre,
        aug,
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Normalize(mean=0.0, std=1.0)
    ])

    T_test = transforms.Compose([
        T_pre,
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Normalize(mean=0.0, std=1.0)
    ])

    return T_train, T_test


def load_data(opts):
    T_train, T_test = get_transforms(opts)
    # print('T train shape: ', T_train)     # Compose()
    T_train = NViewTransform(T_train, opts.n_views)

    # train_dataset = OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train, label=opts.label,
    #                         load_feats=None)
    train_dataset = OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train, load_feats=None)
    print("Total dataset length:", len(train_dataset))    # 3227


    valint_feats, valext_feats = None, None
    valint = OpenBHB(opts.data_dir, train=False, internal=True, transform=T_train, load_feats=valint_feats)
    valext = OpenBHB(opts.data_dir, train=False, internal=False, transform=T_train, load_feats=valext_feats)

    # valint = OpenBHB(opts.data_dir, train=False, internal=True, transform=T_train,
    #                     label=opts.label, load_feats=valint_feats)
    # valext = OpenBHB(opts.data_dir, train=False, internal=False, transform=T_train,
    #                     label=opts.label, load_feats=valext_feats)
        # print('valint length: ', len(valint))    # 362
        # print('valext length: ', len(valext))    # 395

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.bs, shuffle=True, num_workers=3,
                                               persistent_workers=True, drop_last=True)
    train_loader_score = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train),
                                                     batch_size=opts.bs, shuffle=True, num_workers=3,
                                                     persistent_workers=True, drop_last=True)
    test_internal = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=True, transform=T_test), 
                                                batch_size=opts.bs, shuffle=False, num_workers=3,
                                                persistent_workers=True, drop_last=True)
    test_external = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=False, transform=T_test), 
                                                batch_size=opts.bs, shuffle=False, num_workers=3,
                                                persistent_workers=True, drop_last=True)
    return train_loader, train_loader_score, test_internal, test_external

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


def train(train_loader, model, optimizer, opts, epoch):    
    '''
    load bs*sps slices per call
    '''

    scaler = torch.cuda.amp.GradScaler() if opts.amp else None    # None
    model.train()

    t1 = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):   # labels: torch.Size([6])
        # if idx == 0: 
            # print('idx: ', idx)
        print('image type: ', type(images))     # list
        print('label shape from original dataloader: ', labels.shape)    # tensor([14.4000, 24.0000,  6.4700,  8.9973, 36.0000, 19.0000, 29.0000, 24.0000,
                               # 26.2724, 23.0000], dtype=torch.float64)
            # print('len(images): ', len(images))   # 1
            # print('single train image shape: ', images[0].shape)   # torch.Size([10, 1, 182, 218, 182])
            # print('single train labels shape: ', labels[0])     # tensor(14.4000, dtype=torch.float64)
        

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
        # labels_numpy = labels.numpy()   # (6,)
        # print('numpy labels: ', labels_numpy)  # e.g. [21. 18.]
        # print('numpy labels shape: ', labels_numpy.shape)   # (2,)


        # randomly choose opts.sps slices per subject: (one batch: 10 subjects, each pick 8 slices)
        indices = np.random.choice(182, size=opts.bs * opts.sps, replace=True)    # randomly pick 80 slices from images
        print('selected indices: ', indices)
        X = []
        # y = []
        for i in range(opts.bs):   # 10 subjects
            # print('labels[i].item(): ', labels[i].item())
            # y.append(labels_numpy[i])
            print(indices[i * opts.sps: (i+1) * opts.sps])
            X.append(images_numpy[i, :, :, :, indices[i * opts.sps: (i+1) * opts.sps]]) 
            # print(batch_X[i, :, :, :, indices[i*SPS:(i+1)*SPS]].shape)  # (8, 1, 182, 218)

        X = np.array(X)     # (6, 8, 1, 182, 218)
        X = X.reshape(opts.bs * opts.sps, *X.shape[2:])    # X: (80, 1, 182, 218)
        # print('label shape: ', len(y))   # bs
        # y = [val for val in y for i in range(opts.sps)]
        # y = np.array(y)  # (48,) 
        # print('label shape after repeating: ', y.shape)
        # print('y: ', y)

        # ytorch = torch.from_numpy(y)
        # print('torch y shape: ', ytorch.shape)            # torch.Size([16])  bs*sps
        torch_X = torch.from_numpy(X).to(opts.device)     # torch.Size([48, 1, 182, 218])
        true_masks = torch.from_numpy(X).to(opts.device) 
        # torch_y = torch.from_numpy(y).to(opts.device)     # torch.Size([48])
        # torch_X.to(opts.device)       # torch_X: torch.Size([80, 1, 182, 218])
        # print('input torch shape: ', torch_X.shape)   

        # bsz = labels.shape[0]   # 16
        # print(bsz)


        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.MSELoss()

        # start training
        with tqdm(total=3227*182, desc=f'Epoch {epoch}/{opts.epochs}') as pbar:
            with torch.cuda.amp.autocast(scaler is not None):   # automatic mixed precision
                masks_pred = model(torch_X)   # output dimension: torch.Size([48, 1, 182, 218])
                print('masks shape: ', masks_pred.shape)    # torch.Size([bs*sps, 1, 182, 218])


                # loss = criterion(masks_pred.squeeze(1), true_masks.float())
                loss = criterion(masks_pred.squeeze(1), true_masks.float().squeeze(1))
                # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float().squeeze(1), multiclass=False)

            
            optimizer.zero_grad()
            grad_scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.update(opts.bs * opts.sps)

    return masks_pred

if __name__ == '__main__':
    torch.cuda.empty_cache()
    opts = parse_arguments()
    
    set_seed(opts.trial)

    train_loader, train_loader_score, test_loader_int, test_loader_ext = load_data(opts)
    model = load_model(opts)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, momentum=opts.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score

    # optimizer = load_optimizer(model, opts)



    model_name = opts.model
    if opts.warm:
        model_name = f"{model_name}_warm"
    if opts.amp:
        model_name = f"{model_name}_amp"
    
    
    run_name = (f"{model_name}_"
                # f"{optimizer_name}_"
                f"tf{opts.tf}_"
                f"lr{opts.lr}_{opts.lr_decay}_step{opts.lr_decay_step}_rate{opts.lr_decay_rate}_"
                f"temp{opts.temp}_"
                f"wd{opts.weight_decay}_"
                f"bsz{opts.bs}_views{opts.n_views}_"
                f"trainall_{opts.train_all}_"
                # f"kernel_{kernel_name}_"
                # f"f{opts.alpha}_lambd{opts.lambd}_"
                f"trial{opts.trial}")
    print('run name: ', run_name)
    save_dir = os.path.join(opts.save_dir, f"openbhb_models", run_name)
    ensure_dir(save_dir)


    print('Config:', opts)
    print('Model:', model.__class__.__name__)
    # print('Criterion:', infonce)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)

    if opts.amp:
        print("Using AMP")
    
    start_time = time.time()
    best_acc = 0.

    

    for epoch in range(1, opts.epochs + 1):
        adjust_learning_rate(opts, optimizer, epoch)
        # with tqdm(total=3227*182, desc=f'Epoch {epoch}/{opts.epochs}') as pbar:

            # train one batch
        train(train_loader, model, optimizer, opts, epoch)

    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }
    torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/unet_30_1am_0325.pth')


        # t2 = time.time()
    #     print(f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f} loss {loss_train:.4f}")

    #     if epoch % opts.save_freq == 0:

    #         mae_train, mae_int, mae_ext = compute_age_mae(model, train_loader_score, test_loader_int, test_loader_ext, opts)
    #         print("Age MAE:", mae_train, mae_int, mae_ext)

    #         ba_train, ba_int, ba_ext = compute_site_ba(model, train_loader_score, test_loader_int, test_loader_ext, opts)
    #         print("Site BA:", ba_train, ba_int, ba_ext)

    #         challenge_metric = ba_int**0.3 * mae_ext
    #         print("Challenge score", challenge_metric)
    
    #     save_file = os.path.join(save_dir, f"weights.pth")
    #     save_model(model, optimizer, opts, epoch, save_file)
    
    # mae_train, mae_int, mae_ext = compute_age_mae(model, train_loader_score, test_loader_int, test_loader_ext, opts)
    # print("Age MAE:", mae_train, mae_int, mae_ext)

    # ba_train, ba_int, ba_ext = compute_site_ba(model, train_loader_score, test_loader_int, test_loader_ext, opts)
    # print("Site BA:", ba_train, ba_int, ba_ext)
    
    # challenge_metric = ba_int**0.3 * mae_ext
    # print("Challenge score", challenge_metric)

    

