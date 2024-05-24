import math
import torch
import random
import numpy as np
import os
import wandb
import torch.nn.functional as F
import models
from pathlib import Path


class NViewTransform:
    """Create N augmented views of the same image"""
    def __init__(self, transform, N):
        self.transform = transform
        self.N = N

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.N)]

def arg2bool(val):
    if isinstance(val, bool):
        return val
    
    elif isinstance(val, str):
        if val == "true":
            return True
        
        if val == "false":
            return False
    
    val = int(val)
    assert val == 0 or val == 1
    return val == 1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MAE():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.outputs = []
        self.targets = []
        self.avg = np.inf
    
    def update(self, outputs, targets):
        self.outputs.append(outputs.detach())
        self.targets.append(targets.detach())
        self.avg = F.l1_loss(torch.cat(self.outputs, 0), torch.cat(self.targets, 0))

class Accuracy():
    def __init__(self, topk=(1,)):
        self.reset()
        self.topk = topk
    
    def reset(self):
        self.outputs = []
        self.targets = []
        self.avg = np.inf
    
    def update(self, outputs, targets):
        self.outputs.append(outputs.detach())
        self.targets.append(targets.detach())
        self.avg = accuracy(torch.cat(self.outputs, 0), torch.cat(self.targets, 0), self.topk)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state_dict = model.state_dict()
    if torch.cuda.device_count() > 1:
        state_dict = model.module.state_dict()

    state = {
        'opts': opt,
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'run_id': wandb.run.id
    }
    torch.save(state, save_file)
    del state

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.lr_decay == 'cosine':
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        print("lr_decay_epochs:", args.lr_decay_epochs)
        print("Types of elements:", [type(x) for x in args.lr_decay_epochs])
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

@torch.no_grad()
def gather_age_feats(model, dataloader, opts):
    features = []
    age_labels = []

    model.eval()
    for idx, (images, ages, _) in enumerate(dataloader):
        # if idx == 1:
        #     break
        # images = images.to(opts.device)
        # print('type(images): ', type(images))
        images = torch.cat(images, dim=0)  # images: torch.Size([bs, 1, 182, 218, 182])
        # print('images shape: ', images.shape)
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
            feature = model.features(torch_X)
            # feature = wi_net(encoder_feature)
            features.append(feature)
            # print('model features shape: ', model.features(torch_X).shape)
            # features.append(model.features(torch_X))
            for i in range(opts.sps):
                age_labels.append(ages)
                
            if idx == 1:
                print('torch X shape: ', torch_X.shape)
                print('torch X: ', torch_X)
                print('feature: ', feature)
                print('age feature shape: ', feature.shape)
                # print('feature: ', model.features(torch_X))
                # print('age feature shape: ', model.features(torch_X).shape)

            # print('features: ', features)
            # print('feature length: ', len(features))
            # print('age length: ', len(age_labels))
    
    # print('torch.cat(features, 0) shape: ', torch.cat(features, 0).shape)
    return torch.cat(features, 0).cpu().numpy(), torch.cat(age_labels, 0).cpu().numpy()

@torch.no_grad()
def compute_age_mae(model, train_loader, test_int, test_ext, opts):
    age_estimator = models.AgeEstimator()

    print("Training age estimator")
    train_X, train_y = gather_age_feats(model, train_loader, opts)
    print('Fitting: ')
    mae_train = age_estimator.fit(train_X, train_y)

    print("Computing BA")
    print('reading internal validation dataset: ')
    int_X, int_y = gather_age_feats(model, test_int, opts)
    print('train_x age mae score: ', age_estimator.score(train_X, train_y))
    print('int x age shape: ', int_X.shape)
    print('int y age shape: ', int_y.shape)
    print('reading external validation dataset: ')
    ext_X, ext_y = gather_age_feats(model, test_ext, opts)
    print('ext x age shape: ', ext_X.shape)
    print('ext y age shape: ', ext_y.shape)
    mae_int = age_estimator.score(int_X, int_y)
    mae_ext = age_estimator.score(ext_X, ext_y)

    return mae_train, mae_int, mae_ext

@torch.no_grad()
def gather_site_feats(model, dataloader, opts):
    features = []
    site_labels = []

    model.eval()
    for idx, (images, _, sites) in enumerate(dataloader):
        # if idx == 1:
        #     break
        # images = images.to(opts.device)
        # features.append(model.features(images))
        # site_labels.append(sites)
        images = torch.cat(images, dim=0)  # images: torch.Size([bs, 1, 182, 218, 182])
        images_numpy = images.numpy() 
        indices = np.random.choice(182, size=opts.bs * opts.sps, replace=True)
        X = []
        if images_numpy.shape[0] == opts.bs:
            for i in range(opts.bs):   # opts.bs subjects
                # print(indices[i * opts.sps: (i+1) * opts.sps])
                # print('site label gt: ', sites[i])
                X.append(images_numpy[i, :, :, :, indices[i * opts.sps: (i+1) * opts.sps]])
                # sample_a = np.zeros(64, dtype=np.float32)   # sample_a: one hot site label


            # print('y site label: ', y_site)


            X = np.array(X)     # (4, 1, 1, 182, 218)
            X = X.reshape(opts.bs * opts.sps, *X.shape[2:]) 
            torch_X = torch.from_numpy(X).to(opts.device) 
            np.save('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/torch_x_sample.npy', torch_X.cpu().numpy())
            # encoder_feature = model.features(torch_X)
            # feature = wi_net(encoder_feature)
            feature = model.features(torch_X)
            # print('model.features(torch_X): ', model.features(torch_X))
            # features.append(model.features(torch_X))
            features.append(feature)
            for i in range(opts.sps):
                site_labels.append(sites)
            
            if idx == 1:
                print('torch X shape: ', torch_X.shape)
                print('torch X: ', torch_X)
                print('feature: ', feature)
                print('site feature shape: ', feature.shape)
                # print('feature: ', model.features(torch_X))
                # print('site feature shape: ', model.features(torch_X).shape)
                # break
                # print('features: ', features)
                # print('torch.cat(features, 0) shape: ', torch.cat(features, 0))
                # print('site label batch 1: ', sites)
        # print('site features shape: ', )
    # print('torch.cat(features, 0) shape: ', torch.cat(features, 0))
    return torch.cat(features, 0).cpu().numpy(), torch.cat(site_labels, 0).cpu().numpy()
    

@torch.no_grad()
def compute_site_ba(model, train_loader, test_int, test_ext, opts):
    site_estimator = models.SiteEstimator()

    print("Training site estimator")
    train_X, train_y = gather_site_feats(model, train_loader, opts)
    # print('train site features: ', train_X)
    ba_train = site_estimator.fit(train_X, train_y)
    # site_estimator.plot_confusion_matrix(train_X, train_y)
    
    print("Computing BA")
    int_X, int_y = gather_site_feats(model, test_int, opts)
    ext_X, ext_y = gather_site_feats(model, test_ext, opts)
    print('int x site shape: ', int_X.shape)
    print('int y site shape: ', int_y.shape)
    print('train_x site ba score: ', site_estimator.score(train_X, train_y))
    ba_int = site_estimator.score(int_X, int_y)
    # ba_ext = site_estimator.score(ext_X, ext_y)

    return ba_train, ba_int
    # return ba_train, ba_int, ba_ext   
    