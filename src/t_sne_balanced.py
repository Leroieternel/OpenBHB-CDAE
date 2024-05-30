
import numpy as np
import torch
import os
import models
import argparse
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchvision import datasets
from data import FeatureExtractor, OpenBHB, bin_age, OpenBHB_balanced
from data.transforms import Crop, Pad, Cutout
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
from util import compute_age_mae, compute_site_ba
from models.wi_net import Wi_Net
from models.age_net import Age_Net
from models.wi_net_balanced import Wi_Net_balanced
from models.age_net_balanced import Age_Net_balanced
from sklearn.manifold import TSNE

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

    train_dataset = OpenBHB_balanced(opts.data_dir, train=True, internal=True, transform=T_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.bs, shuffle=True, num_workers=3,
                                               persistent_workers=True, drop_last=True)
    # train_loader_score = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train),
    #                                                  batch_size=opts.bs, shuffle=True, num_workers=3,
    #                                                  persistent_workers=True)
    # test_internal = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=True, transform=T_test), 
    #                                             batch_size=opts.bs, shuffle=False, num_workers=3,
    #                                             persistent_workers=True, drop_last=True)
    # test_external = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=False, transform=T_test), 
    #                                             batch_size=opts.bs, shuffle=False, num_workers=3,
    #                                             persistent_workers=True, drop_last=True)
    return train_loader


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
            # np.save('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/torch_x_sample.npy', torch_X.cpu().numpy())
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
        # print('torch.cat(features, 0).cpu().numpy() shape: ', torch.cat(features, 0).cpu().numpy().shape)
        # print('torch.cat(site_labels, 0).cpu().numpy() shape: ', torch.cat(site_labels, 0).cpu().numpy().shape)
    return torch.cat(features, 0).cpu().numpy(), torch.cat(site_labels, 0).cpu().numpy()
    


if __name__ == "__main__":
    print('hi')
    opts = parse_arguments()
    # model_encoder = models.UNet_Encoder_1(n_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = load_data(opts)
    
    
    # load the network, single channel, 1 class
    model_encoder = models.UNet_Encoder_b_wosc(n_channels=1, n_classes=5)
    model_decoder = models.UNet_Decoder_b_wosc(n_channels=1, n_classes=1)
    wi_net = Wi_Net(input_dim=507, output_dim=5, dropout_rate=0.5)
    age_net = Age_Net(input_dim=507)
    param_encoder = list(model_encoder.parameters())
    param_decoder = list(model_decoder.parameters())
    param_wi = list(wi_net.parameters())
    param_age_mlp = list(age_net.parameters())
    # params = param_encoder + param_decoder + param_wi
    params = param_encoder + param_decoder + param_wi + param_age_mlp
    optimizer = torch.optim.RMSprop(params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 

    checkpoint = torch.load('/scratch_net/murgul/jiaxia/saved_models/cdae_500_mse_bs4_sps1_0529_5.pth', map_location='cuda')  #

    model_encoder.load_state_dict(checkpoint['model_encoder_state_dict']) 
    model_encoder = model_encoder.to(device)
    # wi_net = wi_net.to(device)
    model_encoder.eval()
    # model_decoder.eval()
    wi_net.eval()
    age_net.eval()

    train_features, labels = gather_site_feats(model_encoder, train_loader, opts)
    # tsne = TSNE(n_components=2, random_state=42)


    tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=800)
    transformed_features = tsne.fit_transform(train_features)
    print('trainsformed features shape: ', transformed_features.shape)
    print(transformed_features[:, 0])
    print('##########################################################')
    print(transformed_features[:, 1])

    # t-SNE 结果的可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Data from 5 Sites')
    plt.legend(*scatter.legend_elements(), title="Sites")
    print(labels[:100])
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
    plt.savefig('hi_balanced.jpg')


