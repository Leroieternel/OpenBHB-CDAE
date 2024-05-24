import datetime
import math
import os
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
from data import FeatureExtractor, OpenBHB, bin_age, FeatureExtractor_balanced, OpenBHB_balanced
from data.transforms import Crop, Pad, Cutout
from tqdm import tqdm
from utils.corr_score import correlation_loss
from utils.bce_loss_score import bce_logits_loss
import wandb
from models.wi_net import Wi_Net
from gpustat import GPUStatCollection



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

    train_dataset = OpenBHB_balanced(opts.data_dir, train=True, internal=True, transform=T_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.bs, shuffle=True, num_workers=3,
                                               persistent_workers=True, drop_last=True)
    test_internal = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=True, transform=T_test), 
                                                batch_size=opts.bs, shuffle=False, num_workers=3,
                                                persistent_workers=True, drop_last=True)
    test_external = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=False, transform=T_test), 
                                                batch_size=opts.bs, shuffle=False, num_workers=3,
                                                persistent_workers=True, drop_last=True)
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


def train(train_loader, model, optimizer, opts, epoch, train_loss, params):    
    '''
    load bs*sps slices per call
    '''
    # scaler = torch.cuda.amp.GradScaler() if opts.amp else None    # None
    model.train()

    t1 = time.time()

    '''
    TRAIN for each batch
    '''
    idx = 0
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    
    for idx, (images, ages, sites) in enumerate(train_loader):   # labels: torch.Size([6])
        # if idx == 4:
        #     break
        print('label shape from original dataloader: ', sites.shape)    # tensor([14.4000, 24.0000,  6.4700,  8.9973, 36.0000, 19.0000, 29.0000, 24.0000,
                               # 26.2724, 23.0000], dtype=torch.float64)
        # print('len(images): ', len(images))   # 1
        # print('single train site labels shape: ', sites[0])
        # print('single train image shape: ', images[0].shape)   # torch.Size([10, 1, 182, 218, 182])
            # print('single train labels shape: ', labels[0])     # tensor(14.4000, dtype=torch.float64)
        

        ''' 
            transform dataloader images: [10, 1, 182, 218, 182] to [80, 1, 182, 218]
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
        

        images = torch.cat(images, dim=0)  # images: torch.Size([bs, 1, 182, 218, 182])
        images_numpy = images.numpy()   # (bs, 1, 182, 218, 182)
        sites_numpy = sites.numpy()   # (bs,)
        # print('numpy labels: ', sites_numpy)  # e.g. [ 4. 24. 49.  7.]
        # print('numpy labels shape: ', sites_numpy.shape)   # (bs,)


        # randomly choose opts.sps slices per subject: (one batch: opts.bs subjects, each pick 8 slices)
        indices = np.random.choice(182, size=opts.bs * opts.sps, replace=True)    # randomly pick 80 slices from images
        X = []
        # pack site ground truth labels a (one hot) for one batch
        y_site = []
        for i in range(opts.bs):   # opts.bs subjects
            # print(indices[i * opts.sps: (i+1) * opts.sps])
            # print('site label gt: ', sites[i])
            X.append(images_numpy[i, :, :, :, indices[i * opts.sps: (i+1) * opts.sps]]) 
            sample_a = np.zeros(5, dtype=np.float32)   # sample_a: one hot site label
            sample_a[int(sites[i])] = 1.0    # sample_a = 1 index: site - 1
            y_site.append(sample_a)

        # print('y site label: ', y_site)


        X = np.array(X)     # (6, 8, 1, 182, 218)
        X = X.reshape(opts.bs * opts.sps, *X.shape[2:])    # X: (bs*sps, 1, 182, 218)
        y_site = [val for val in y_site for i in range(opts.sps)]
        # print('y site length: ', len(y_site))
        # print('y site duplicated label: ', y_site)
        y_site = np.array(y_site)
        torch_a = torch.from_numpy(y_site)       # a: true label (tensor) for one batch
        a = torch_a.to(opts.device)     # torch.Size([bs*sps])
        print('a: ', a)
        # if idx == 0:
        #     print('saving images: ')
        #     np.save('/scratch_net/murgul/jiaxia/saved_models/train_image_3_1.npy', X[0])
        #     np.save('/scratch_net/murgul/jiaxia/saved_models/train_image_5_1.npy', X[1])
        #     np.save('/scratch_net/murgul/jiaxia/saved_models/train_image_1_1.npy', X[2])
        #     np.save('/scratch_net/murgul/jiaxia/saved_models/train_image_4_1.npy', X[3])
        # if idx == 1:
        #     print('saving images: ')
        #     np.save('/scratch_net/murgul/jiaxia/saved_models/train_image_3_2.npy', X[0])
        #     np.save('/scratch_net/murgul/jiaxia/saved_models/train_image_3_2_1.npy', X[1])
        #     np.save('/scratch_net/murgul/jiaxia/saved_models/train_image_1_2.npy', X[2])
        #     np.save('/scratch_net/murgul/jiaxia/saved_models/train_image_2_2.npy', X[3])
        # if idx == 3:
        #     np.save('/scratch_net/murgul/jiaxia/saved_models/train_image_4_2.npy', X[0])

        torch_X = torch.from_numpy(X).to(opts.device)     # torch.Size([bs*sps, 1, 182, 218])
        std = torch_X.std(dim=[1, 2, 3], unbiased=False) 
        # print('std: ', std)
        true_masks = torch.from_numpy(X).to(opts.device) 

        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        

        # Initialize 'fake' labels b for one batch
        b_batch = []

        # Loop to create each tensor
        for i in range(opts.bs * opts.sps):
            # for single b: create a tensor of zeros, length=64
            b_single = torch.zeros(5)

            # Randomly select an index which is different from ground truth label index
            rand_index = torch.randint(0, 5, (1,))
            while rand_index == torch.argmax(torch_a[i]):
                rand_index = torch.randint(0, 5, (1,))
            # Set the randomly selected index to 1
            b_single[rand_index] = 1
            # Add the tensor to the list
            b_batch.append(b_single)
        print('b_batch: ', b_batch)

        # Stack the list of tensors into a single tensor
        b_batch = torch.stack(b_batch)
        b_batch = b_batch.to(opts.device) 
        print('b_batch shape: ', b_batch.shape)


        # start training
        # with tqdm(total=3227*182, desc=f'Epoch {epoch}/{opts.epochs}') as pbar:
        with torch.cuda.amp.autocast():   # automatic mixed precision

            '''
            masks_encoder_out: a^ + zi
            zi_b_batch: zi + b
            xi_b_hat: zi + b --> decoder --> xi_b^
            zi_hat_b_hat: xi_b_hat --> encoder --> zi^ + b^
            xi_a_hat: zi_hat_b_hat --> decoder --> reconstructed xi_a^

            mask_recon: vanilla autoencoder output (masks_encoder_out directly passes through decoder)

            '''

            # print('torch X shape: ', torch_X.shape)      # torch.Size([4, 1, 182, 218])
            masks_encoder_out, torch_X_en_x1, torch_X_en_x2, torch_X_en_x3, torch_X_en_x4 = model_encoder(torch_X)   # input one batch image to encoder
            # print('encoder output shape: ', masks_encoder_out.shape)    # supposed: torch.Size([bs*sps, 512])

            zi = masks_encoder_out[:, 5: ]   # zi for output of the encoder
            # print('zi: ', zi)
            # print('zi shape: ', zi.shape)
            
            a_hat = masks_encoder_out[:, : 5]    # a^ for output of the encoder (zt)
            # print('a hat: ', a_hat)   # torch.Size([bs*sps, 64])
            zi_b_batch = torch.cat((b_batch, zi), dim=1)
            # print('zi_b_batch shape: ', zi_b_batch.shape)     # torch.Size([4, 512])
            a_zi = torch.cat((a, zi), dim=1)

            xi_b_hat= model_decoder(zi_b_batch, torch_X_en_x1, torch_X_en_x2, torch_X_en_x3, torch_X_en_x4)
            # print('xi_b_hat shape: ', xi_b_hat.shape)         # torch.Size([4, 1, 182, 218])
            # mask_recon = model_decoder(masks_encoder_out)
            mask_recon = model_decoder(a_zi, torch_X_en_x1, torch_X_en_x2, torch_X_en_x3, torch_X_en_x4)
            # print('mask_recon shape: ', mask_recon.shape)     # torch.Size([4, 1, 182, 218])


            zi_hat_b_hat, xi_b_hat_en_x1, xi_b_hat_en_x2, xi_b_hat_en_x3, xi_b_hat_en_x4 = model_encoder(xi_b_hat)
            zi_hat = zi_hat_b_hat[:, 5: ]
            b_hat = zi_hat_b_hat[:, : 5]
            zi_hat_a = torch.cat((a, zi_hat), dim=1)
            # print('zi_hat_a shape: ', zi_hat_a.shape)   # torch.Size([4, 512])
            xi_a_hat = model_decoder(zi_hat_a, xi_b_hat_en_x1, xi_b_hat_en_x2, xi_b_hat_en_x3, xi_b_hat_en_x4)

            print('forward ok')

            # define w_t and w_i
            wt = nn.Identity()
            wt_zt = wt(a_hat)
            # print('wt(zt) shape: ', wt_zt.shape)    # torch.Size([4, 64])
            print('wt(zt): ', wt_zt)

            wi_zi = model_wi(zi)
            # print('wi(zi) shape: ', wi_zi.shape)      # torch.Size([4, 64])
            print('wi(zi): ', wi_zi)

            '''
            compute 6 losses
            1. reconstruction loss on xi_b^ and xi_a^
            2. adversarial excitation loss
            3. adversarial inhibition loss
            4. cycle loss: cycle-consistency loss on xi_a^ and xi_a (MSE)
            5. latent loss: cycle-consistency loss on latent space zi^ and zi (MSE)
            6. correlation loss on xi_b^ and xi_a
            final loss: weighted sum of 1-6

            '''

            
            recon_criterion = nn.MSELoss()
            bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')

            
            recon_loss = recon_criterion(mask_recon, true_masks.float())
            print('recon loss: ', recon_loss)

            exc_loss = bce_criterion(wt_zt, a)
            inh_loss = -bce_criterion(wi_zi, a)

            # exc_loss = bce_logits_loss(wt_zt, a)
            # inh_loss = -bce_logits_loss(wi_zi, a)

            print('exc loss: ', exc_loss)
            print('inh loss: ', inh_loss)

            cycle_loss = recon_criterion(xi_a_hat, true_masks.float())
            print('cycle loss: ', cycle_loss)
            latent_loss = recon_criterion(zi_hat, zi)
            print('latent loss: ', latent_loss)
            cc_loss = correlation_loss(xi_b_hat, torch_X)    # torch.Size([bs*sps, 1, 182, 218])
            print('cc loss: ', cc_loss)

            # recon_loss_hat = recon_criterion(xi_a_hat, true_masks.float())

            if epoch <= 2:
                loss = 1.0 * recon_loss + 8.0 * exc_loss + 5 * inh_loss + 1 * cycle_loss + 4.0 * latent_loss 
            else:
                loss = 4 * recon_loss + 8.0 * exc_loss + 5 * inh_loss + 3 * cycle_loss + 4.0 * latent_loss + 2 * cc_loss
            # loss = 1.0 * recon_loss + 1.0 * exc_loss + 1 * inh_loss + 0.5 * cycle_loss + 4.0 * latent_loss + 20.0 * cc_loss
            print('loss: ', loss)
            print('loss shape: ', loss.shape)
            train_loss.append(loss.item())  
            
            # print('cuda usage: ', torch.cuda.memory_allocated(device=opts.device))
            # print('current cuda: ', torch.cuda.current_device())
            # print('cuda version: ', torch.version.cuda)
            # print('cuda usage: ', torch.cuda.memory_summary())
            # print('cuda name: ', torch.cuda.get_device_name(0))

            t = torch.cuda.get_device_properties(0).total_memory 
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a
            # print('cuda total_memory: ', t)
            # print('cuda reserved memory: ', r)
            # print('cuda allocated memory: ', a)
            # print('free memory left: ', f)
            gpu_stats = GPUStatCollection.new_query()
                        
            for gpu in gpu_stats:
                print(f"GPU {gpu.index}: {gpu.name}, utilization: {gpu.utilization}%")

        
        
        # back propagation

            optimizer.zero_grad()
            
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, opts.gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            print('backprop ok')

        if idx % 50 == 0:

            wandb.log({'epoch': epoch, 'loss': loss.item(), 'recon_loss: ': recon_loss.item(), 
                        'cycle_loss': cycle_loss.item(), 'latent_loss': latent_loss.item(), 
                        'inh_loss': inh_loss.item(), 'exc_loss': exc_loss.item(), 'cc_loss': cc_loss.item()})
            wandb.log({"original_images": wandb.Image(torch_X[0]), "xi_b_hat_images": wandb.Image(xi_b_hat[0]), "xi_a_hat_images": wandb.Image(xi_a_hat[0])})


            # pbar.update(opts.bs * opts.sps)

        idx += 1

    # return masks_pred

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
        
    opts = parse_arguments()
    wandb.init(project='openbhb_cdae_0507_balanced_dataset', dir='/scratch_net/murgul/jiaxia/saved_models')
    config = wandb.config
    config.learning_rate = opts.lr
    
    set_seed(opts.trial)

    # train_loader, train_loader_score, test_loader_int, test_loader_ext = load_data(opts)
    train_loader, test_loader_int, test_loader_ext = load_data(opts)
    model = load_model(opts)
    model_encoder = models.UNet_Encoder(n_channels=1)
    model_encoder = model_encoder.to(opts.device)
    model_decoder = models.UNet_Decoder(n_channels=1, n_classes=1)
    model_decoder = model_decoder.to(opts.device)
    model_wi = Wi_Net()
    model_wi = model_wi.to(opts.device)
    # model = models.UNet(n_channels=1, n_classes=1)

    param_encoder = list(model_encoder.parameters())
    param_decoder = list(model_decoder.parameters())
    params = param_encoder + param_decoder

    optimizer = torch.optim.RMSprop(params, lr=opts.lr, weight_decay=opts.weight_decay, momentum=opts.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 

    

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
                f"trial{opts.trial}")
    print('run name: ', run_name)
    save_dir = os.path.join(opts.save_dir, f"openbhb_models", run_name)
    ensure_dir(save_dir)


    print('Config:', opts)
    print('Model:', model.__class__.__name__)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)

    if opts.amp:
        print("Using AMP")
    
    start_time = time.time()
    best_acc = 0.

    # training 
    train_loss = []
    # torch.backends.cudnn.enabled = False
    wandb.watch(model_decoder, log="all")
    wandb.watch(model_encoder, log="all")
    wandb.watch(model_wi, log="all")


    for epoch in range(0, opts.epochs):
        adjust_learning_rate(opts, optimizer, epoch)
        train(train_loader, model, optimizer, opts, epoch, train_loss, params)
        if epoch == 49:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_500_mse_bs4_sps1_0510_balanced_4_8_5_3_4_2_epoch50.pth')


    torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_500_mse_bs4_sps1_0510_balanced_4_8_5_3_4_2.pth')

    print('train loss: ', train_loss)
    train_loss_numpy = np.array(train_loss)
    np.save('/scratch_net/murgul/jiaxia/saved_models/cdae_500_mse_bs4_sps1_0510_balanced_4_8_5_3_4_2loss.npy', train_loss_numpy)
    
    checkpoint = {
            'epoch': epoch,
            'model_encoder_state_dict': model_encoder.state_dict(),
            'model_decoder_state_dict': model_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }
    torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_500_mse_bs4_sps1_0510_balanced_4_8_5_3_4_2.pth')


    

