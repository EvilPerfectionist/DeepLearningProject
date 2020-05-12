import os
import argparse
import numpy as np
import torch
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from torch.utils.data import DataLoader
from dataset import mydata
from memory_network import Memory_Network
from networks import Generator, unet_generator, Discriminator, Discriminator2, weights_init_normal
from helpers import print_args, print_losses
from helpers import save_sample, adjust_learning_rate
from util import zero_grad

def init_training(args):
    """Initialize the data loader, the networks, the optimizers and the loss functions."""
    datasets = dict()
    datasets['train'] = mydata(img_path = args.train_data_path, img_size = args.img_size, km_file_path = args.km_file_path, color_info = args.color_info)
    datasets['test'] = mydata(img_path = args.test_data_path, img_size = args.img_size,km_file_path = args.km_file_path, color_info = args.color_info)
    for phase in ['train', 'test']:
        print('{} dataset len: {}'.format(phase, len(datasets[phase])))

    # define loaders
    data_loaders = {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        'test': DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }

    # check CUDA availability and set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    # set up models
    mem = Memory_Network(mem_size = args.mem_size, color_info = args.color_info, color_feat_dim = args.color_feat_dim, spatial_feat_dim = args.spatial_feat_dim, top_k = args.top_k, alpha = args.alpha).to(device)
    generator = Generator(args.gen_norm).to(device)
    discriminator = Discriminator(args.disc_norm).to(device)
    generator2 = unet_generator(1, 2, args.n_feats, args.color_feat_dim).to(device)
    discriminator2 = Discriminator2(3, args.color_feat_dim, args.img_size).to(device)

    # initialize weights
    if args.apply_weight_init:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # adam optimizer with reduced momentum
    optimizers = {
        'gen': torch.optim.Adam(generator.parameters(), lr=args.base_lr_gen, betas=(0.5, 0.999)),
        'disc': torch.optim.Adam(discriminator.parameters(), lr=args.base_lr_disc, betas=(0.5, 0.999)),
        'gen2': torch.optim.Adam(generator2.parameters(), lr=args.base_lr_mem),
        'disc2': torch.optim.Adam(discriminator2.parameters(), lr=args.base_lr_mem),
        'mem': torch.optim.Adam(mem.parameters(), lr = args.base_lr_mem)
    }

    # losses
    losses = {
        'l1': torch.nn.L1Loss(reduction='mean'),
        'disc': torch.nn.BCELoss(reduction='mean'),
        'smoothl1': torch.nn.SmoothL1Loss()
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

    return global_step, device, data_loaders, mem, generator, discriminator, generator2, discriminator2, optimizers, losses


def run_training(args):
    """Initialize and run the training process."""
    global_step, device, data_loaders, mem, generator, discriminator, generator2, discriminator2, optimizers, losses = init_training(args)
    #  run training process
    for epoch in range(args.start_epoch, args.max_epoch):
        print('\n========== EPOCH {} =========='.format(epoch))

        for phase in ['train', 'test']:

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

                # get data
                #img_l, real_img_lab = sample[:, 0:1, :, :].float().to(device), sample.float().to(device)

                # generate targets
                print(img_l.size(0))
                target_ones = torch.ones(img_l.size(0), 1).to(device)
                target_zeros = torch.zeros(img_l.size(0), 1).to(device)

                ### 1) Train spatial feature extractor
                if phase == 'train':
                    res_feature = mem(res_input)
                    print("res_feature" + str(res_feature.shape))
                    mem_loss = mem.unsupervised_loss(res_feature, color_feat, args.color_thres)
                    optimizers['mem'].zero_grad()
                    mem_loss.backward()
                    optimizers['mem'].step()

                    ### 2) Update Memory module
                    with torch.no_grad():
                        res_feature = mem(res_input)
                        mem.memory_update(res_feature, color_feat, args.color_thres, index)

                    dis_color_feat = torch.cat([torch.unsqueeze(color_feat, 2) for _ in range(args.img_size)], dim = 2)
                    dis_color_feat = torch.cat([torch.unsqueeze(dis_color_feat, 3) for _ in range(args.img_size)], dim = 3)
                    fake_img_ab = generator2(img_l, color_feat)
                    real = discriminator2(img_ab, img_l, dis_color_feat)
                    d_loss_real = losses['disc'](real, target_ones)

                    fake = discriminator2(fake_img_ab, img_l, dis_color_feat)
                    d_loss_fake = losses['disc'](fake, target_zeros)
                    d_loss = d_loss_real + d_loss_fake

                    optimizers['disc2'].zero_grad()
                    d_loss.backward()
                    optimizers['disc2'].step()

                    ### 4) Train Generator
                    fake_img_ab = generator2(img_l, color_feat)
                    fake = discriminator2(fake_img_ab, img_l, dis_color_feat)
                    g_loss_GAN = losses['disc'](fake, target_ones)

                    g_loss_smoothL1 = losses['smoothl1'](fake_img_ab, img_ab)
                    g_loss = g_loss_GAN + g_loss_smoothL1

                    optimizers['gen2'].zero_grad()
                    g_loss.backward()
                    optimizers['gen2'].step()

            #     if phase == 'train':
            #         # adjust LR
            #         global_step += 1
            #         adjust_learning_rate(optimizers['gen'], global_step, base_lr=args.base_lr_gen,
            #                              lr_decay_rate=args.lr_decay_rate, lr_decay_steps=args.lr_decay_steps)
            #         adjust_learning_rate(optimizers['disc'], global_step, base_lr=args.base_lr_disc,
            #                              lr_decay_rate=args.lr_decay_rate, lr_decay_steps=args.lr_decay_steps)
            #
            #         # reset generator gradients
            #         optimizers['gen'].zero_grad()
            #
            #     # train / inference the generator
            #     with torch.set_grad_enabled(phase == 'train'):
            #         fake_img_ab = generator(img_l)
            #         fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1).to(device)
            #
            #         # adv loss
            #         adv_loss = losses['disc'](discriminator(fake_img_lab), target_ones)
            #         # l1 loss
            #         l1_loss = losses['l1'](real_img_lab[:, 1:, :, :], fake_img_ab)
            #         # full gen loss
            #         full_gen_loss = (1.0 - args.l1_weight) * adv_loss + (args.l1_weight * l1_loss)
            #
            #         if phase == 'train':
            #             full_gen_loss.backward()
            #             optimizers['gen'].step()
            #
            #     epoch_gen_adv_loss += adv_loss.item()
            #     epoch_gen_l1_loss += l1_loss.item()
            #
            #     if phase == 'train':
            #         # reset discriminator gradients
            #         optimizers['disc'].zero_grad()
            #
            #     # train / inference the discriminator
            #     with torch.set_grad_enabled(phase == 'train'):
            #         prediction_real = discriminator(real_img_lab)
            #         prediction_fake = discriminator(fake_img_lab.detach())
            #
            #         loss_real = losses['disc'](prediction_real, target_ones * args.smoothing)
            #         loss_fake = losses['disc'](prediction_fake, target_zeros)
            #         full_disc_loss = loss_real + loss_fake
            #
            #         if phase == 'train':
            #             full_disc_loss.backward()
            #             optimizers['disc'].step()
            #
            #     epoch_disc_real_loss += loss_real.item()
            #     epoch_disc_fake_loss += loss_fake.item()
            #     epoch_disc_real_acc += np.mean(prediction_real.detach().cpu().numpy() > 0.5)
            #     epoch_disc_fake_acc += np.mean(prediction_fake.detach().cpu().numpy() <= 0.5)
            #
            #     # save the first sample for later
            #     if phase == 'test' and idx == 0:
            #         sample_real_img_lab = real_img_lab
            #         sample_fake_img_lab = fake_img_lab
            #
            # # display losses
            # print_losses(epoch_gen_adv_loss, epoch_gen_l1_loss,
            #              epoch_disc_real_loss, epoch_disc_fake_loss,
            #              epoch_disc_real_acc, epoch_disc_fake_acc,
            #              len(data_loaders[phase]), args.l1_weight)
            #
            # # save after every nth epoch
            # if phase == 'test':
            #     if epoch % args.save_freq == 0 or epoch == args.max_epoch - 1:
            #         gen_path = os.path.join(args.save_path, 'checkpoint_ep{}_gen.pt'.format(epoch))
            #         disc_path = os.path.join(args.save_path, 'checkpoint_ep{}_disc.pt'.format(epoch))
            #         torch.save(generator.state_dict(), gen_path)
            #         torch.save(discriminator.state_dict(), disc_path)
            #         print('Checkpoint.')
            #
            #     # display sample images
            #     save_sample(
            #         sample_real_img_lab,
            #         sample_fake_img_lab,
            #         os.path.join(args.save_path, 'sample_ep{}.png'.format(epoch))
            #     )


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description='Image colorization with GANs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, default='../data',
                        help='Download and extraction path for the dataset.')
    parser.add_argument("--train_data_path", type = str, default = '/home/leon/DeepLearning/Project/Dataset/PussnToots/')
    parser.add_argument("--test_data_path", type = str, default = '/home/leon/DeepLearning/Project/Dataset/PussnToots/')
    parser.add_argument("--img_size", type = int, default = 64)
    parser.add_argument("--km_file_path", type = str, default = './pts_in_hull.npy')
    parser.add_argument("--n_feats", type = int, default = 64)
    parser.add_argument("--color_feat_dim", type = int, default = 313)
    parser.add_argument("--spatial_feat_dim", type = int, default = 512)
    parser.add_argument("--mem_size", type = int, default = 120)
    parser.add_argument("--alpha", type = float, default = 0.1)
    parser.add_argument("--top_k", type = int, default = 30)
    parser.add_argument("--color_thres", type = float, default = 0.7)
    parser.add_argument("--color_info", type = str, default = 'dist', help = 'option should be dist or RGB')
    parser.add_argument('--save_path', type=str, default='../checkpoints',
                        help='Save and load path for the network weights.')
    parser.add_argument('--save_freq', type=int, default=20, help='Save frequency during training.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='If start_epoch>0, load previously saved weigth from the save_path.')
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--smoothing', type=float, default=0.9)
    parser.add_argument('--l1_weight', type=float, default=0.99)
    parser.add_argument('--base_lr_gen', type=float, default=3e-4, help='Base learning rate for the generator.')
    parser.add_argument('--base_lr_disc', type=float, default=6e-5, help='Base learning rate for the discriminator.')
    parser.add_argument("--base_lr_mem", type = float, default = 1e-4, help='Base learning rate for the Memory_Network.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate for both networks.')
    parser.add_argument('--lr_decay_steps', type=float, default=6e4, help='Learning rate decay steps for both networks.')
    parser.add_argument('--gen_norm', type=str, default='batch', choices=['batch', 'instance'],
                        help='Defines the type of normalization used in the generator.')
    parser.add_argument('--disc_norm', type=str, default='batch', choices=['batch', 'instance', 'spectral'],
                        help='Defines the type of normalization used in the discriminator.')
    parser.add_argument('--apply_weight_init', type=int, default=0, choices=[0, 1],
                        help='If set to 1, applies the "weights_init_normal" function from networks.py.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    # display arguments
    print_args(args)

    run_training(args)
