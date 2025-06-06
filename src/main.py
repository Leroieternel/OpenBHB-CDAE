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
# from main_mse import get_transforms
from problem import get_train_data, DatasetHelper


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
    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')

    # Optimizer
    parser.add_argument('--epochs', type=int, help='number of epochs', default=300)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step'], default='step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='decay rate for learning rate (for step)')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="700,800,900")
    parser.add_argument('--lr_decay_step', type=int, help='decay rate step (overwrites lr_decay_epochs', default=10)
    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--optimizer', type=str, help="optimizer (adam or sgd)", choices=["adam", "sgd"], default="adam")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=5e-5)

    # Data
    parser.add_argument('--train_all', type=arg2bool, help='train on all dataset including validation (int+ext)', default=True)
    parser.add_argument('--tf', type=str, help='data augmentation', choices=['none', 'crop', 'cutout', 'all'], default='all')
    
    # Loss 
    parser.add_argument('--method', type=str, help='loss function', choices=['supcon', 'yaware', 'threshold', 'expw'], default='supcon')
    parser.add_argument('--kernel', type=str, help='Kernel function (not for supcon)', choices=['cauchy', 'gaussian', 'rbf'], default=None)
    parser.add_argument('--delta_reduction', type=str, help='use mean or sum to reduce 3d delta mask (only for method=threshold)', default='sum')
    parser.add_argument('--temp', type=float, help='loss temperature', default=0.1)
    parser.add_argument('--alpha', type=float, help='infonce weight', default=1.)
    parser.add_argument('--sigma', type=float, help='gaussian-rbf kernel sigma / cauchy gamma', default=1)
    parser.add_argument('--n_views', type=int, help='num. of multiviews', default=2)
    parser.add_argument('--bs', type=int, help='num. of subjects per batch', default=10)
    parser.add_argument('--sps', type=int, help='Slices per subject', default=8)
    

    opts = parser.parse_args()

    if opts.batch_size > 256:
        print("Forcing warm")
        opts.warm = True

    if opts.lr_decay_step is not None:
        opts.lr_decay_epochs = list(range(opts.lr_decay_step, opts.epochs, opts.lr_decay_step))
        print(f"Computed decay epochs based on step ({opts.lr_decay_step}):", opts.lr_decay_epochs)
    else:
        iterations = opts.lr_decay_epochs.split(',')
        opts.lr_decay_epochs = list([])
        for it in iterations:
            opts.lr_decay_epochs.append(int(it))

    if opts.warm:   # batch size > 256
        opts.warmup_from = 0.01
        opts.warm_epochs = 10
        if opts.lr_decay == 'cosine':
            eta_min = opts.lr * (opts.lr_decay_rate ** 3)
            opts.warmup_to = eta_min + (opts.lr - eta_min) * (
                    1 + math.cos(math.pi * opts.warm_epochs / opts.epochs)) / 2
        else:
            opts.milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
            opts.warmup_to = opts.lr

    if opts.method == 'supcon':
        print('method == supcon, binning age')
        opts.label = 'bin'
    else:
        print('method != supcon, using real age value')
        opts.label = 'cont'

    if opts.method == 'supcon' and opts.kernel is not None:
        print('Invalid kernel for supcon')
        exit(0)
    
    if opts.method != 'supcon' and opts.kernel is None:
        print('Kernel cannot be None for method != supcon')
        exit(1)
    
    if opts.model == 'densenet121':
        opts.n_views = 1
        
    return opts


def get_transforms(opts):
    selector = FeatureExtractor("vbm")
    
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

    # opts.bs, opts.sps
    
    train_dataset = DatasetHelper(data_loader=get_train_data)
    T_train, T_test = get_transforms(opts)
    X_train, y_train = train_dataset.get_data()  # (3227, 3659572) (3227, 2)
    # print('X train shape: ', X_train.shape)
    # df_labels_train = train_dataset.labels_to_dataframe()
    dtype = "quasiraw"
    batch_X = FeatureExtractor(dtype=dtype).transform(X_train[: opts.bs])  # (10, 1, 182, 218, 182)
    indices = np.random.choice(182, size=opts.bs * opts.sps, replace=True)
    print('selected indices: ', indices)

    # X: samples trained per batch
    X = []
    for i in range(opts.bs):   # 10 subjects
        print(indices[i * opts.sps: (i+1) * opts.spsS])
        X.append(batch_X[i, :, :, :, indices[i * opts.sps:(i+1) * opts.sps]]) 
        # print(batch_X[i, :, :, :, indices[i*SPS:(i+1)*SPS]].shape)  # (8, 1, 182, 218)

    X = np.array(X)
    print(X.shape)
    X = X.reshape(opts.bs * opts.sps, *X.shape[2:])

    # train_dataset = OpenBHB(opts.data_dir, train=True, internal=True, transform=batch_X, label=opts.label,
    #                         load_feats=None)
    # if opts.train_all:
        # valint_feats, valext_feats = None, None
        # if opts.biased_features is not None:
        #     valint_feats = opts.biased_features.replace('.pth', '_valint.pth')
        #     valext_feats = opts.biased_features.replace('.pth', '_valext.pth')

        # valint = OpenBHB(opts.data_dir, train=False, internal=True, transform=T_train,
        #                  label=opts.label, load_feats=valint_feats)
        # valext = OpenBHB(opts.data_dir, train=False, internal=False, transform=T_train,
        #                  label=opts.label, load_feats=valext_feats)
        # train_dataset = torch.utils.data.ConcatDataset([train_dataset, valint, valext])
        # print('valint length: ', len(valint))    # 362
        # print('valext length: ', len(valext))    # 395
    print("Total dataset length:", len(train_dataset))    # 


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=3,
                                               persistent_workers=True)
    train_loader_score = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train, label=opts.label),
                                                     batch_size=opts.batch_size, shuffle=True, num_workers=3,
                                                     persistent_workers=True)
    test_internal = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=True, transform=T_test), 
                                                batch_size=opts.batch_size, shuffle=False, num_workers=3,
                                                persistent_workers=True)
    test_external = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=False, transform=T_test), 
                                                batch_size=opts.batch_size, shuffle=False, num_workers=3,
                                                persistent_workers=True)
    return train_loader, train_loader_score, test_internal, test_external

def load_model(opts):
    if 'resnet' in opts.model:
        model = models.SupConResNet(opts.model, feat_dim=128)
    elif 'alexnet' in opts.model:
        model = models.SupConAlexNet(feat_dim=128)
    elif 'densenet121' in opts.model:
        model = models.SupConDenseNet(feat_dim=128)
    elif 'unet' in opts.model:
        model = models.unet3d(feat_dim=128)
    
    else:
        raise ValueError("Unknown model", opts.model)

    if opts.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    model = model.to(opts.device)


    def gaussian_kernel(x):
        x = x - x.T
        torch.pi = math.pi
        return torch.exp(-(x**2) / (2*(opts.sigma**2))) / (math.sqrt(2*torch.pi)*opts.sigma)
    
    def rbf(x):
        x = x - x.T
        return torch.exp(-(x**2)/(2*(opts.sigma**2)))
    
    def cauchy(x):
        x = x - x.T
        return  1. / (opts.sigma*(x**2) + 1)

    kernels = {
        'none': None,
        'cauchy': cauchy,
        'gaussian': gaussian_kernel,
        'rbf': rbf
    }

    infonce = losses.KernelizedSupCon(method=opts.method, temperature=opts.temp, 
                                      kernel=kernels[opts.kernel], delta_reduction=opts.delta_reduction)
    infonce = infonce.to(opts.device)

    
    return model, infonce

def load_optimizer(model, opts):
    if opts.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, 
                                    momentum=opts.momentum,
                                    weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

    return optimizer

def train(train_loader, model, infonce, optimizer, opts, epoch):
    loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    scaler = torch.cuda.amp.GradScaler() if opts.amp else None    # None
    model.train()

    t1 = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):
        if idx == 0: 
            print('idx: ', idx)
            print('single train image shape: ', images[0].shape)   # torch.Size([16, 1, 121, 145, 121])  should be [8, 10, 1, 182, 218, 182]
            print('single train labels shape: ', labels[0])     # tensor(10.5014, dtype=torch.float64)
        data_time.update(time.time() - t1)

        images = torch.cat(images, dim=0).to(opts.device)
        bsz = labels.shape[0]

        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast(scaler is not None):   # automatic mixed precision
            projected = model(images)
            print('projected = model(images) shape: ', projected.shape)
            projected = torch.split(projected, [bsz]*opts.n_views, dim=0)  # [bsz, bsz]: new size after splitting
            projected = torch.cat([f.unsqueeze(1) for f in projected], dim=1)
            print('projected shape: ', projected.shape)    # torch.Size([16, 2, 128])
            running_loss = infonce(projected, labels.to(opts.device))
        
        optimizer.zero_grad()
        if scaler is None:   # execute this
            running_loss.backward()
            if opts.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        else:
            scaler.scale(running_loss).backward()
            if opts.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()
        
        loss.update(running_loss.item(), bsz)
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(train_loader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}\t")

    return loss.avg, batch_time.avg, data_time.avg

if __name__ == '__main__':
    torch.cuda.empty_cache()
    opts = parse_arguments()
    
    set_seed(opts.trial)

    train_loader, train_loader_score, test_loader_int, test_loader_ext = load_data(opts)
    model, infonce = load_model(opts)
    optimizer = load_optimizer(model, opts)

    model_name = opts.model
    if opts.warm:
        model_name = f"{model_name}_warm"
    if opts.amp:
        model_name = f"{model_name}_amp"
    
    method_name = opts.method
    if opts.method == 'threshold':
        method_name = f"{method_name}_reduction_{opts.delta_reduction}"

    optimizer_name = opts.optimizer
    if opts.clip_grad:
        optimizer_name = f"{optimizer_name}_clipgrad"

    kernel_name = opts.kernel
    if opts.kernel == "gaussian" or opts.kernel == 'rbf':
        kernel_name = f"{kernel_name}_sigma{opts.sigma}"
    elif opts.kernel == 'cauchy':
        kernel_name = f"{kernel_name}_gamma{opts.sigma}"
    
    run_name = (f"{model_name}_{method_name}_"
                f"{optimizer_name}_"
                f"tf{opts.tf}_"
                f"lr{opts.lr}_{opts.lr_decay}_step{opts.lr_decay_step}_rate{opts.lr_decay_rate}_"
                f"temp{opts.temp}_"
                f"wd{opts.weight_decay}_"
                f"bsz{opts.batch_size}_views{opts.n_views}_"
                f"trainall_{opts.train_all}_"
                f"kernel_{kernel_name}_"
                # f"f{opts.alpha}_lambd{opts.lambd}_"
                f"trial{opts.trial}")
    print('run name: ', run_name)
    save_dir = os.path.join(opts.save_dir, f"openbhb_models", run_name)
    ensure_dir(save_dir)

    opts.model_class = model.__class__.__name__
    opts.criterion = infonce.__class__.__name__
    opts.optimizer_class = optimizer.__class__.__name__

    print('Config:', opts)
    print('Model:', model.__class__.__name__)
    print('Criterion:', infonce)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)

    # writer = torch.utils.tensorboard.writer.SummaryWriter(tb_dir)
    if opts.amp:
        print("Using AMP")
    
    start_time = time.time()
    best_acc = 0.

   
    for epoch in range(1, opts.epochs + 1):
        adjust_learning_rate(opts, optimizer, epoch)

        t1 = time.time()
        loss_train, batch_time, data_time = train(train_loader, model, infonce, optimizer, opts, epoch)
        t2 = time.time()
        print(f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f} loss {loss_train:.4f}")

        if epoch % opts.save_freq == 0:

            mae_train, mae_int, mae_ext = compute_age_mae(model, train_loader_score, test_loader_int, test_loader_ext, opts)
            print("Age MAE:", mae_train, mae_int, mae_ext)

            ba_train, ba_int, ba_ext = compute_site_ba(model, train_loader_score, test_loader_int, test_loader_ext, opts)
            print("Site BA:", ba_train, ba_int, ba_ext)

            challenge_metric = ba_int**0.3 * mae_ext
            print("Challenge score", challenge_metric)
    
        save_file = os.path.join(save_dir, f"weights.pth")
        save_model(model, optimizer, opts, epoch, save_file)
    
    mae_train, mae_int, mae_ext = compute_age_mae(model, train_loader_score, test_loader_int, test_loader_ext, opts)
    print("Age MAE:", mae_train, mae_int, mae_ext)

    ba_train, ba_int, ba_ext = compute_site_ba(model, train_loader_score, test_loader_int, test_loader_ext, opts)
    print("Site BA:", ba_train, ba_int, ba_ext)
    
    challenge_metric = ba_int**0.3 * mae_ext
    print("Challenge score", challenge_metric)


