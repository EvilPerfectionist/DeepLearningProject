import os
import argparse
import numpy as np
import torch
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from torch.utils.data import DataLoader
from dataset import customed_dataset
from memory_network import Memory_Network
from networks import unet_generator, Discriminator2
from helpers import print_args, print_losses
from helpers import save_sample, adjust_learning_rate
from util import zero_grad

def init_training(args):
    """Initialize the data loader, the networks, the optimizers and the loss functions."""
    datasets = dict()
    datasets['train'] = customed_dataset(img_path = args.train_data_path, img_size = args.img_size, km_file_path = args.km_file_path)
    datasets['val'] = customed_dataset(img_path = args.val_data_path, img_size = args.img_size,km_file_path = args.km_file_path)
    for phase in ['train', 'val']:
        print('{} dataset len: {}'.format(phase, len(datasets[phase])))

    # define loaders
    data_loaders = {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        'val': DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }

    # check CUDA availability and set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    # set up models
    mem = Memory_Network(mem_size = args.mem_size, color_feat_dim = args.color_feat_dim, spatial_feat_dim = args.spatial_feat_dim, top_k = args.top_k, alpha = args.alpha).to(device)
    generator = unet_generator(args.color_feat_dim, args.img_size).to(device)
    discriminator = Discriminator2(args.color_feat_dim, args.img_size).to(device)

    # set networks as training mode
    generator = generator.train()
    discriminator = discriminator.train()

    # adam optimizer
    optimizers = {
        'gen': torch.optim.Adam(generator.parameters(), lr=args.base_lr),
        'disc': torch.optim.Adam(discriminator.parameters(), lr=args.base_lr),
        'mem': torch.optim.Adam(mem.parameters(), lr = args.base_lr)
    }

    # losses
    losses = {
        'l1': torch.nn.L1Loss(reduction='mean'),
        'disc': torch.nn.BCELoss(reduction='mean'),
        'smoothl1': torch.nn.SmoothL1Loss(reduction='mean')
    }

    # make save dir, if it does not exists
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # load weights if the training is not starting from the beginning
    global_step = args.start_epoch * len(data_loaders['train']) if args.start_epoch > 0 else 0
    if args.start_epoch > 0:

        generator.load_state_dict(torch.load(
            os.path.join(args.save_path, 'checkpoint_ep{}_gen.pt'.format(args.start_epoch - 1)),
            map_location=device
        ))
        discriminator.load_state_dict(torch.load(
            os.path.join(args.save_path, 'checkpoint_ep{}_disc.pt'.format(args.start_epoch - 1)),
            map_location=device
        ))

    return global_step, device, data_loaders, mem, generator, discriminator, optimizers, losses


def train_and_validation(args):
    """Initialize generator, discriminator, memory_network and run the train and validation process."""
    global_step, device, data_loaders, mem, generator, discriminator, optimizers, losses = init_training(args)
    #  run training process
    for epoch in range(args.start_epoch, args.end_epoch):
        print('\n========== EPOCH {} =========='.format(epoch))

        for phase in ['train', 'val']:

            # running losses for generator
            epoch_gen_adv_loss = 0.0
            epoch_gen_l1_loss = 0.0

            # running losses for discriminator
            epoch_disc_real_loss = 0.0
            epoch_disc_fake_loss = 0.0
            epoch_disc_real_acc = 0.0
            epoch_disc_fake_acc = 0.0

            if phase == 'train':
                print('TRAINING:')
            else:
                print('VALIDATION:')

            for idx, batch in enumerate(data_loaders[phase]):

                res_input = batch['res_input'].to(device)
                color_feat = batch['color_feat'].to(device)
                index = batch['index'].to(device)
                img_l = (batch['l_channel'] / 100.0).to(device)
                img_ab = (batch['ab_channel'] / 110.0).to(device)
                real_img_lab = torch.cat([img_l, img_ab], dim=1).to(device)
                print("res_input" + str(res_input.shape))

                # generate targets
                print(img_l.size(0))
                target_ones = torch.ones(img_l.size(0), 1).to(device)
                target_zeros = torch.zeros(img_l.size(0), 1).to(device)

                ### 1) Train spatial feature extractor
                if phase == 'train':
                    optimizers['mem'].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    res_feature = mem(res_input)
                    print("res_feature" + str(res_feature.shape))

                    if phase == 'train':
                        mem_loss = mem.unsupervised_loss(res_feature, color_feat, args.color_thres)
                        mem_loss.backward()
                        optimizers['mem'].step()

                ### 2) Update Memory module
                if phase == 'train':
                    with torch.no_grad():
                        res_feature = mem(res_input)
                        mem.memory_update(res_feature, color_feat, args.color_thres, index)

                if phase == 'val':
                    top1_feature, _ = mem.topk_feature(res_feature, 1)
                    color_feat = top1_feature[:, 0, :]

                ### 3) Train Generator
                if phase == 'train':
                    optimizers['gen'].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    dis_color_feat = torch.cat([torch.unsqueeze(color_feat, 2) for _ in range(args.img_size)], dim = 2)
                    dis_color_feat = torch.cat([torch.unsqueeze(dis_color_feat, 3) for _ in range(args.img_size)], dim = 3)
                    fake_img_ab = generator(img_l, color_feat)
                    fake = discriminator(fake_img_ab, img_l, dis_color_feat)
                    fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1).to(device)

                    g_loss_GAN = losses['disc'](fake, target_ones)
                    g_loss_smoothL1 = losses['smoothl1'](fake_img_ab, img_ab)
                    g_loss = g_loss_GAN + g_loss_smoothL1

                    if phase == 'train':
                        g_loss.backward()
                        optimizers['gen'].step()

                epoch_gen_adv_loss += g_loss_GAN.item()
                epoch_gen_l1_loss += g_loss_smoothL1.item()

                ### 4) Train Discriminator
                if phase == 'train':
                    optimizers['disc'].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    prediction_real = discriminator(img_ab, img_l, dis_color_feat)
                    prediction_fake = discriminator(fake_img_ab.detach(), img_l, dis_color_feat)

                    d_loss_real = losses['disc'](prediction_real, target_ones * args.smoothing)
                    d_loss_fake = losses['disc'](prediction_fake, target_zeros)
                    d_loss = d_loss_real + d_loss_fake

                    if phase == 'train':
                        d_loss.backward()
                        optimizers['disc'].step()

                epoch_disc_real_loss += d_loss_real.item()
                epoch_disc_fake_loss += d_loss_fake.item()
                epoch_disc_real_acc += np.mean(prediction_real.detach().cpu().numpy() > 0.5)
                epoch_disc_fake_acc += np.mean(prediction_fake.detach().cpu().numpy() <= 0.5)

                # save the first sample for later
                if phase == 'val' and idx == 0:
                    sample_real_img_lab = real_img_lab
                    sample_fake_img_lab = fake_img_lab

                # display losses
                print_losses(epoch_gen_adv_loss, epoch_gen_l1_loss,
                             epoch_disc_real_loss, epoch_disc_fake_loss,
                             epoch_disc_real_acc, epoch_disc_fake_acc,
                             len(data_loaders[phase]), 1.0)

                if phase == 'val':
                    if epoch % args.save_freq == 0 or epoch == args.end_epoch - 1:
                        gen_path = os.path.join(args.save_path, 'checkpoint_ep{}_gen.pt'.format(epoch))
                        disc_path = os.path.join(args.save_path, 'checkpoint_ep{}_disc.pt'.format(epoch))
                        mem_path = os.path.join(args.save_path, 'checkpoint_ep{}_mem.pt'.format(epoch))
                        torch.save(generator.state_dict(), gen_path)
                        torch.save(discriminator.state_dict(), disc_path)
                        torch.save({'mem_model' : mem.state_dict(),
                                     'mem_key' : mem.spatial_key.cpu(),
                                     'mem_value' : mem.color_value.cpu(),
                                     'mem_age' : mem.age.cpu(),
                                     'mem_index' : mem.top_index.cpu()}, mem_path)
                        print('Checkpoint.')

                    # display sample images
                    save_sample(
                        sample_real_img_lab,
                        sample_fake_img_lab,
                        args.img_size,
                        os.path.join(args.save_path, 'sample_ep{}.png'.format(epoch))
                    )

def define_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description='Animation Colorization with Memory-Augmented Networks and a Few Shots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Arguments for initializing dataset
    parser.add_argument("--train_data_path", type = str, default = '/home/leon/DeepLearning/Project/DogTrouble/', help = 'Path to load the training data')
    parser.add_argument("--val_data_path", type = str, default = '/home/leon/DeepLearning/Project/DogTrouble/', help = 'Path to load the validation data')
    parser.add_argument("--img_size", type = int, default = 128, help = 'Height and weight of the images the networks will process')
    parser.add_argument("--km_file_path", type = str, default = './pts_in_hull.npy', help = 'Extra file for mapping color pairs in ab channels into Q(313) categories')
    # Arguments for initializing dataLoader
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--num_workers', type = int, default = 4)
    # Arguments for initializing networks
    parser.add_argument("--mem_size", type = int, default = 360, help = 'The number of color and spatial features that will be stored in the memory_network respectively')
    parser.add_argument("--color_feat_dim", type = int, default = 313, help = 'Dimension of color feaures extracted from an image')
    parser.add_argument("--spatial_feat_dim", type = int, default = 512, help = 'Dimension of spatial feaures extracted from an image')
    parser.add_argument("--top_k", type = int, default = 256, help = 'Select the top k spatial feaures in memory_network which relate to input query')
    parser.add_argument("--alpha", type = float, default = 0.1, help = 'Bias term in the unsupervised loss')
    # Arguments for setting the optimizers
    parser.add_argument("--base_lr", type = float, default = 1e-4, help = 'Base learning rate for the networks.')
    #Arguments for saving network parameters and real and fake images
    parser.add_argument('--save_path', type = str, default='../checkpoints', help = 'Save and load path for the network weights.')
    parser.add_argument('--save_freq', type = int, default = 50, help = 'Save frequency during training.')
    #Arguments for setting start epoch and end epoch
    parser.add_argument('--start_epoch', type = int, default = 0, help = 'If start_epoch>0, load previously saved weigth from the save_path.')
    parser.add_argument('--end_epoch', type = int, default = 200)
    # Rest arguments
    parser.add_argument("--color_thres", type = float, default = 0.7, help = 'Threshold for deciding the case 1 and case 2 in updating the memory_network')
    parser.add_argument('--smoothing', type = float, default = 0.9, help = 'Argument for implementing one-sided label smoothing')
    return parser.parse_args()

if __name__ == '__main__':
    args = define_arguments()
    # display arguments
    print_args(args)
    # train and validation
    train_and_validation(args)
