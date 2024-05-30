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
from data import FeatureExtractor, OpenBHB, bin_age, OpenBHB_balanced_64, OpenBHB_balanced
from data.transforms import Crop, Pad, Cutout
from tqdm import tqdm
from utils.corr_score import correlation_loss
from utils.bce_loss_score import bce_logits_loss
from utils.ce_loss_score import cross_entropy_with_probs
import wandb
from models.wi_net import Wi_Net
from models.age_net import Age_Net
from gpustat import GPUStatCollection



def parse_arguments():
    parser = argparse.ArgumentParser(description="Weakly contrastive learning for brain age predictin",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Misc
    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    # parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
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

    train_dataset = OpenBHB_balanced_64(opts.data_dir, train=True, internal=True, transform=T_train)
    # train_loader_score = torch.utils.data.DataLoader(OpenBHB_balanced_64(opts.data_dir, train=True, internal=True, transform=T_train),
    #                                                  batch_size=opts.bs, shuffle=True, num_workers=3,
    #                                                  persistent_workers=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.bs, shuffle=True, num_workers=3,
                                               persistent_workers=True, drop_last=True)
    test_internal = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=True, transform=T_test), 
                                                batch_size=opts.bs, shuffle=False, num_workers=3,
                                                persistent_workers=True, drop_last=True)
    test_external = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=False, transform=T_test), 
                                                batch_size=opts.bs, shuffle=False, num_workers=3,
                                                persistent_workers=True, drop_last=True)
    return train_loader, test_internal, test_external



def train(train_loader, model_encoder, model_decoder, model_wi, model_age, optimizer, opts, epoch, train_loss, params, fake_label):    
    '''
    load bs*sps slices per call
    '''
    # scaler = torch.cuda.amp.GradScaler() if opts.amp else None    # None
    print(f'***********************************************  Epoch{epoch}  ***************************************************')
    # model_encoder.train()
    # model_decoder.train()
    # model_wi.train()
    # model_age.train()

    t1 = time.time()

    '''
    TRAIN for each batch
    '''
    idx = 0
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    for idx, (images, ages, sites) in enumerate(train_loader):   # labels: torch.Size([6])
        # if idx == 1:
        #     break
        ages = ages.float()
        
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

        # randomly choose opts.sps slices per subject: (one batch: opts.bs subjects, each pick 8 slices)
        indices = np.random.choice(182, size=opts.bs * opts.sps, replace=True)    # randomly pick 80 slices from images
        X = []
        # pack site ground truth labels a (one hot) for one batch
        y_site = []
        
        for i in range(opts.bs):   # opts.bs subjects
            # print(indices[i * opts.sps: (i+1) * opts.sps])
            # print('site label gt: ', sites[i])
            X.append(images_numpy[i, :, :, :, indices[i * opts.sps: (i+1) * opts.sps]]) 
            sample_a = np.zeros(64, dtype=np.float32)   # sample_a: one hot site label
            sample_a[int(sites[i])] = 1.0    # sample_a = 1 index: site - 1
            y_site.append(sample_a)


        X = np.array(X)     # (6, 8, 1, 182, 218)
        X = X.reshape(opts.bs * opts.sps, *X.shape[2:])    # X: (bs*sps, 1, 182, 218)
        y_site = [val for val in y_site for i in range(opts.sps)]
        
        
        # print('y site length: ', len(y_site))
        # print('y site duplicated label: ', y_site)
        y_site = np.array(y_site)
        torch_a = torch.from_numpy(y_site)       # a: true label (tensor) for one batch
        a = torch_a.to(opts.device)     # torch.Size([bs*sps])
        # np.save('/scratch_net/murgul/jiaxia/saved_models/test_image_0.npy', X[0])
        torch_X = torch.from_numpy(X).to(opts.device)     # torch.Size([bs*sps, 1, 182, 218])
        torch_age = ages.repeat_interleave(opts.sps).to(opts.device) 

        std = torch_X.std(dim=[1, 2, 3], unbiased=False) 
        # print('std: ', std)
        true_masks = torch.from_numpy(X).to(opts.device) 

        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        # Initialize 'fake' labels b for one batch
        b_batch = []

        # Loop to create each tensor
        for i in range(opts.bs * opts.sps):
            # for single b: create a tensor of zeros, length=64
            b_single = torch.zeros(64)

            # Randomly select an index which is different from ground truth label index
            rand_index = torch.randint(0, 64, (1,))
            while rand_index == torch.argmax(torch_a[i]):
                rand_index = torch.randint(0, 64, (1,))
            # Set the randomly selected index to 1
            b_single[rand_index] = 1
            # Add the tensor to the list
            b_batch.append(b_single)
        # print('b_batch: ', b_batch)

        # Stack the list of tensors into a single tensor
        b_batch = torch.stack(b_batch)
        b_batch = b_batch.to(opts.device) 

        # start training
        with torch.cuda.amp.autocast():   # automatic mixed precision
        # with open('ffff', 'w') as f:
            '''
            masks_encoder_out: a^ + zi
            zi_b_batch: zi + b
            xi_b_hat: zi + b --> decoder --> xi_b^
            zi_hat_b_hat: xi_b_hat --> encoder --> zi^ + b^
            xi_a_hat: zi_hat_b_hat --> decoder --> reconstructed xi_a^

            mask_recon: vanilla autoencoder output (zi + ground truth site label a --> decoder --> mask_recon)

            '''

            masks_encoder_out, torch_X_en_x1, torch_X_en_x2, torch_X_en_x3, torch_X_en_x4 = model_encoder(torch_X)   # input one batch image to encoder

            zi = masks_encoder_out[:, 64: ]   # zi for output of the encoder
            
            # print('zi: ', zi)
            print('zi shape: ', zi.shape)
            
            a_hat = masks_encoder_out[:, : 64]    # a^ for output of the encoder (zt)
            zi_b_batch = torch.cat((b_batch, zi), dim=1)
            a_zi = torch.cat((a, zi), dim=1)

            xi_b_hat= model_decoder(zi_b_batch, torch_X_en_x1, torch_X_en_x2, torch_X_en_x3, torch_X_en_x4)
            mask_recon = model_decoder(a_zi, torch_X_en_x1, torch_X_en_x2, torch_X_en_x3, torch_X_en_x4)

            zi_hat_b_hat, xi_b_hat_en_x1, xi_b_hat_en_x2, xi_b_hat_en_x3, xi_b_hat_en_x4 = model_encoder(xi_b_hat)
            zi_hat = zi_hat_b_hat[:, 64: ]
            b_hat = zi_hat_b_hat[:, : 64]
            zi_hat_a = torch.cat((a, zi_hat), dim=1)
            xi_a_hat = model_decoder(zi_hat_a, xi_b_hat_en_x1, xi_b_hat_en_x2, xi_b_hat_en_x3, xi_b_hat_en_x4)
            age_pred = model_age(zi)

            # print('forward ok')

            # define w_t and w_i
            wt = nn.Identity()
            wt_zt = wt(a_hat)
            wi_zi = model_wi(zi)

            sigmoid = nn.Sigmoid()

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

            # fake_label = torch.full((4, 64), 0.5).to(opts.device)

            exc_loss = bce_criterion(wt_zt, a)
            ce_criterion = cross_entropy_with_probs
            inh_loss = ce_criterion(wi_zi, fake_label)
            # inh_loss = -bce_criterion(wi_zi, a)
            # exc_loss = F.binary_cross_entropy(wt_zt, a, reduction='sum')
            # inh_loss = F.binary_cross_entropy(wi_zi, fake_label, reduction='sum')
            age_loss = recon_criterion(age_pred, torch_age)
            # inh_loss = 1 / (1 + recon_criterion(wi_zi, a))

            cycle_loss = recon_criterion(xi_a_hat, true_masks.float())
            latent_loss = recon_criterion(zi_hat, zi)
            cc_loss = correlation_loss(xi_b_hat, torch_X)    # torch.Size([bs*sps, 1, 182, 218])

            if epoch < 20:
                loss = 3 * recon_loss + 6.0 * exc_loss + 0.1 * inh_loss + 1 * cycle_loss + 3 * latent_loss + 3.5 * cc_loss
            else:
                loss = 3 * recon_loss + 6.0 * exc_loss + 0.1 * inh_loss + 0.5 * cycle_loss + 3 * latent_loss + 3.5 * cc_loss + 0.5 * age_loss.float()
            # else:
            #     loss = 3 * recon_loss + 4.0 * exc_loss + 0.5 * inh_loss + 3 * cycle_loss + 3 * latent_loss + 3.5 * cc_loss + 0.5 * age_loss.float()
            # if epoch <= 1:
            # loss = age_loss.float()
            # loss = 5 * recon_loss + 3.0 * exc_loss + 0.5 * inh_loss + 0.5 * cycle_loss + 4.0 * latent_loss + 3.5 * cc_loss + 0.1 * age_loss.float()

            train_loss.append(loss.item())  

            if idx == 1:
                print('a: ', a)
                print('b_batch: ', b_batch)
                print('age labels: ', ages)
                print('age_pred: ', age_pred)
                print('sigmoid(zt): ', sigmoid(a_hat))
                print('zt: ', a_hat)
                print('sigmoid(wi(zi)): ', sigmoid(wi_zi))
                print('zi: ', zi)
                print('recon loss: ', recon_loss)
                print('exc loss: ', exc_loss)
                print('inh loss: ', inh_loss)
                print('age loss: ', age_loss)
                print('cycle loss: ', cycle_loss)
                print('latent loss: ', latent_loss)
                print('cc loss: ', cc_loss)
                print('loss: ', loss)

            
            print('zi: ', zi)
            print('sigmoid(wi(zi)): ', sigmoid(wi_zi))
        
        # back propagation

            optimizer.zero_grad()
            grad_scaler.scale(loss.float()).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, opts.gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            # print('backprop ok')

        if idx % 50 == 0:

            wandb.log({'epoch': epoch, 'loss': loss.item(), 'recon_loss: ': recon_loss.item(), 
                        'cycle_loss': cycle_loss.item(), 'latent_loss': latent_loss.item(), 
                        'inh_loss': inh_loss.item(), 'exc_loss': exc_loss.item(), 'cc_loss': cc_loss.item(), 'age_loss': age_loss.item()})
            wandb.log({"original_images": wandb.Image(torch_X[0]), "xi_b_hat_images": wandb.Image(xi_b_hat[0]), "xi_a_hat_images": wandb.Image(xi_a_hat[0])})

    # return masks_pred

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
        
    opts = parse_arguments()
    run = wandb.init(project='openbhb_cdae_0529_balanced_64_2', dir='/scratch_net/murgul/jiaxia/saved_models')
    print("WandB Project Name:", run.project)
    config = wandb.config
    config.learning_rate = opts.lr
    
    # set_seed(opts.trial)

    # train_loader, train_loader_score, test_loader_int, test_loader_ext = load_data(opts)
    train_loader, test_loader_int, test_loader_ext = load_data(opts)
    model_encoder = models.UNet_Encoder(n_channels=1)
    model_encoder = model_encoder.to(opts.device)
    model_decoder = models.UNet_Decoder(n_channels=1, n_classes=1)
    model_decoder = model_decoder.to(opts.device)
    model_wi = Wi_Net(input_dim=960, output_dim=64)
    model_wi = model_wi.to(opts.device)
    model_age = Age_Net(input_dim=960).float()
    model_age = model_age.to(opts.device)
    model_encoder.train()
    model_decoder.train()
    model_wi.train()
    model_age.train()

    param_encoder = list(model_encoder.parameters())
    param_decoder = list(model_decoder.parameters())
    param_wi = list(model_wi.parameters())
    param_age_mlp = list(model_age.parameters())
    params = param_encoder + param_decoder + param_wi + param_age_mlp
    # params = param_encoder + param_decoder + param_wi

    optimizer = torch.optim.RMSprop(params, lr=opts.lr, weight_decay=opts.weight_decay, momentum=opts.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 

    print('Config:', opts)
    # print('Model:', model.__class__.__name__)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)

    if opts.amp:
        print("Using AMP")
    
    # start_time = time.time()
    # best_acc = 0.

    # training 
    train_loss = []
    fake_label = torch.full((4, 64), 1/64).to(opts.device)
    # random_tensor = torch.rand(opts.bs * opts.sps, 64).to(opts.device)
    # fake_label_1 = random_tensor / random_tensor.sum(dim=1, keepdim=True)

    # print('fake label: ', fake_label_1)
    # torch.backends.cudnn.enabled = False
    wandb.watch(model_decoder, log="all")
    wandb.watch(model_encoder, log="all")
    wandb.watch(model_wi, log="all")

    for epoch in range(0, opts.epochs):
        adjust_learning_rate(opts, optimizer, epoch)
        train(train_loader, model_encoder, model_decoder, model_wi, model_age, optimizer, opts, epoch, train_loss, params, fake_label)
        if epoch == 29:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'wi_net_state_dict': model_wi.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0529_7_sc_epoch30.pth')

        if epoch == 49:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'wi_net_state_dict': model_wi.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0529_7_sc_epoch50.pth')

        if epoch == 99:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'wi_net_state_dict': model_wi.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0529_7_sc_epoch100.pth')
        if epoch == 149:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'wi_net_state_dict': model_wi.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0529_7_sc_epoch150.pth')

        if epoch == 199:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'wi_net_state_dict': model_wi.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0529_7_sc_epoch200.pth')
        if epoch == 249:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'wi_net_state_dict': model_wi.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0529_7_sc_epoch250.pth')
        if epoch == 299:
            checkpoint = {
                'epoch': epoch,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'wi_net_state_dict': model_wi.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0529_7_sc_epoch300.pth')
        
        # if epoch % 50 == 0:
            # save_file = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")
            # save_model(model, optimizer, opts, epoch, save_file)

            # mae_train, mae_int, mae_ext = compute_age_mae(model_encoder, train_loader_score, test_loader_int, test_loader_ext, opts)
            # writer.add_scalar("train/mae", mae_train, epoch)
            # writer.add_scalar("test/mae_int", mae_int, epoch)
            # writer.add_scalar("test/mae_ext", mae_ext, epoch)
            # print("Age MAE:", mae_train, mae_int, mae_ext)

            # ba_train, ba_int, ba_ext = compute_site_ba(model_encoder, train_loader_score, test_loader_int, test_loader_ext, opts)
            # writer.add_scalar("train/site_ba", ba_train, epoch)
            # writer.add_scalar("test/ba_int", ba_int, epoch)
            # writer.add_scalar("test/ba_ext", ba_ext, epoch)
            # print("Site BA:", ba_train, ba_int, ba_ext)

            # challenge_metric = ba_int**0.3 * mae_ext + (1 / 64) ** 0.3 * mae_int
            # writer.add_scalar("test/score", challenge_metric, epoch)
            # print("Challenge score", challenge_metric)

    print('train loss: ', train_loss)
    train_loss_numpy = np.array(train_loss)
    np.save('/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0529_7_sc_loss.npy', train_loss_numpy)
    
    checkpoint = {
            'epoch': epoch,
            'model_encoder_state_dict': model_encoder.state_dict(),
            'model_decoder_state_dict': model_decoder.state_dict(),
            'wi_net_state_dict': model_wi.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }
    torch.save(checkpoint, '/scratch_net/murgul/jiaxia/saved_models/cdae_300_mse_bs4_sps1_0529_7_sc.pth')

    


    

