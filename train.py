import os
import argparse
import numpy as np
import torch
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from torch.utils.data import DataLoader
from dataset import customed_dataset
from networks import Generator, Discriminator, Memory_Network, Feature_Integrator, weights_init_normal
from helpers import print_args, print_losses
from helpers import save_sample, adjust_learning_rate

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
    if args.use_memory == True:
        mem = Memory_Network(mem_size = args.mem_size, color_feat_dim = args.color_feat_dim, spatial_feat_dim = args.spatial_feat_dim, top_k = args.top_k, alpha = args.alpha).to(device)
        feature_integrator = Feature_Integrator(3, 1, 200).to(device)
    generator = Generator(args.color_feat_dim, args.img_size, args.gen_norm).to(device)
    discriminator = Discriminator(args.color_feat_dim, args.img_size, args.dis_norm).to(device)

    # initialize weights
    if args.apply_weight_init == True:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # set networks as training mode
    generator = generator.train()
    discriminator = discriminator.train()
    if args.use_memory == True:
        mem = mem.train()
        feature_integrator = feature_integrator.train()

    # adam optimizer
    if args.use_memory == True:
        optimizers = {
            'gen': torch.optim.Adam(generator.parameters(), lr=args.base_lr_gen, betas=(0.5, 0.999)),
            'disc': torch.optim.Adam(discriminator.parameters(), lr=args.base_lr_disc, betas=(0.5, 0.999)),
            'mem': torch.optim.Adam(mem.parameters(), lr = args.base_lr_mem),
            'feat': torch.optim.Adam(feature_integrator.parameters(), lr = args.base_lr_feat)
        }
    else:
        optimizers = {
            'gen': torch.optim.Adam(generator.parameters(), lr=args.base_lr_gen),
            'disc': torch.optim.Adam(discriminator.parameters(), lr=args.base_lr_disc),
        }

    # losses
    losses = {
        'l1': torch.nn.L1Loss(reduction='mean'),
        'disc': torch.nn.BCEWithLogitsLoss(reduction='mean'),
        'smoothl1': torch.nn.SmoothL1Loss(reduction='mean'),
        'KLD': torch.nn.KLDivLoss(reduction='batchmean')
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
        mem_checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_ep{}_mem.pt'.format(args.start_epoch - 1)), map_location=device)
        mem.load_state_dict(mem_checkpoint['mem_model'])
        mem.sptial_key = mem_checkpoint['mem_key']
        mem.color_value = mem_checkpoint['mem_value']
        mem.age = mem_checkpoint['mem_age']
        mem.img_id = mem_checkpoint['img_id']

        feature_integrator.load_state_dict(torch.load(
            os.path.join(args.save_path, 'checkpoint_ep{}_feat.pt'.format(args.start_epoch - 1)),
            map_location=device
        ))

    if args.use_memory == True:
        return global_step, device, data_loaders, mem, feature_integrator, generator, discriminator, optimizers, losses
    else:
        return global_step, device, data_loaders, generator, discriminator, optimizers, losses

def train_and_validation(args):
    """Initialize generator, discriminator, memory_network and run the train and validation process."""
    if args.use_memory == True:
        global_step, device, data_loaders, mem, feature_integrator, generator, discriminator, optimizers, losses = init_training(args)
    else:
        global_step, device, data_loaders, generator, discriminator, optimizers, losses = init_training(args)
    #  run training process
    for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
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
                print(color_feat.shape)
                print(torch.sum(color_feat[0]))
                img_l = (batch['img_l'] / 100.0).to(device)
                img_ab = (batch['img_ab'] / 110.0).to(device)
                img_id = batch['img_id'].to(device)
                real_img_lab = torch.cat([img_l, img_ab], dim=1).to(device)

                # generate targets
                target_ones = torch.ones(img_l.size(0), 1).to(device)
                target_zeros = torch.zeros(img_l.size(0), 1).to(device)

                if phase == 'train':
                    # adjust LR
                    global_step += 1
                    adjust_learning_rate(optimizers['gen'], global_step, base_lr=args.base_lr_gen,
                                         lr_decay_rate=args.lr_decay_rate, lr_decay_steps=args.lr_decay_steps)
                    adjust_learning_rate(optimizers['disc'], global_step, base_lr=args.base_lr_disc,
                                         lr_decay_rate=args.lr_decay_rate, lr_decay_steps=args.lr_decay_steps)
                    if args.use_memory == True:
                        adjust_learning_rate(optimizers['mem'], global_step, base_lr=args.base_lr_mem,
                                             lr_decay_rate=args.lr_decay_rate, lr_decay_steps=args.lr_decay_steps)
                        adjust_learning_rate(optimizers['feat'], global_step, base_lr=args.base_lr_feat,
                                             lr_decay_rate=args.lr_decay_rate, lr_decay_steps=args.lr_decay_steps)

                if args.use_memory == True:
                    ### 1) Train spatial feature extractor
                    if phase == 'train':
                        optimizers['mem'].zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        res_feature = mem(res_input)

                        if phase == 'train':
                            mem_loss = mem.unsupervised_loss(res_feature, color_feat, args.color_thres)
                            mem_loss.backward()
                            optimizers['mem'].step()

                    ### 2) Update Memory module
                    if phase == 'train':
                        with torch.no_grad():
                            res_feature = mem(res_input)
                            mem.memory_update(res_feature, color_feat, args.color_thres, img_id)

                    ### 3) Train Feature_Integrator
                    if args.use_feat_integrator == True:
                        if phase == 'train':
                            optimizers['feat'].zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            top_features, ref_img_ids = mem.topk_feature(res_feature, 3)
                            print("xixi" + str(torch.sum(top_features[0, 0, :])))
                            top_features = torch.transpose(top_features, dim0 = 1, dim1 = 2)
                            print(top_features.shape)
                            combined_features = feature_integrator(top_features)
                            for param in feature_integrator.parameters():
                                print('feature_integrator' + str(param.data))
                            print("haha" + str(torch.sum(combined_features[0])))
                            feat_loss = losses['KLD'](combined_features, color_feat)

                            if phase == 'train':
                                feat_loss.backward()
                                optimizers['feat'].step()

                    if phase == 'val':
                        top1_feature, ref_img_ids = mem.topk_feature(res_feature, 3)
                        color_feat = top1_feature[:, 0, :]
                        if args.use_feat_integrator == True:
                            color_feat = combined_features

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
                    g_loss_L1 = losses['l1'](fake_img_ab, img_ab)
                    g_loss = (1.0 - args.l1_weight) * g_loss_GAN + (args.l1_weight * g_loss_L1)

                    if phase == 'train':
                        g_loss.backward()
                        optimizers['gen'].step()

                epoch_gen_adv_loss += g_loss_GAN.item()
                epoch_gen_l1_loss += g_loss_L1.item()

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
                    sample_ref_img_ids = ref_img_ids
                    print(ref_img_ids)

            # display losses
            print_losses(epoch_gen_adv_loss, epoch_gen_l1_loss,
                         epoch_disc_real_loss, epoch_disc_fake_loss,
                         epoch_disc_real_acc, epoch_disc_fake_acc,
                         len(data_loaders[phase]), 1.0)

            if phase == 'val':
                if epoch % args.save_freq == 0 or epoch == args.end_epoch - 1:
                    gen_path = os.path.join(args.save_path, 'checkpoint_ep{}_gen.pt'.format(epoch))
                    disc_path = os.path.join(args.save_path, 'checkpoint_ep{}_disc.pt'.format(epoch))
                    if args.use_memory == True:
                        mem_path = os.path.join(args.save_path, 'checkpoint_ep{}_mem.pt'.format(epoch))
                        if args.use_feat_integrator == True:
                            feat_path = os.path.join(args.save_path, 'checkpoint_ep{}_feat.pt'.format(epoch))
                    torch.save(generator.state_dict(), gen_path)
                    torch.save(discriminator.state_dict(), disc_path)
                    if args.use_memory == True:
                        torch.save({'mem_model' : mem.state_dict(),
                                     'mem_key' : mem.spatial_key.cpu(),
                                     'mem_value' : mem.color_value.cpu(),
                                     'mem_age' : mem.age.cpu(),
                                     'img_id' : mem.img_id.cpu()}, mem_path)
                        if args.use_feat_integrator == True:
                            torch.save(feature_integrator.state_dict(), feat_path)
                    print('Checkpoint.')

                # display sample images
                save_sample(
                    sample_real_img_lab,
                    sample_fake_img_lab,
                    sample_ref_img_ids,
                    args.img_size,
                    os.path.join(args.save_path, 'sample_ep{}.png'.format(epoch)),
                    os.path.join(args.save_path, 'ref_ep{}.png'.format(epoch))
                )

def define_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description='Animation Colorization with Memory-Augmented Networks and a Few Shots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Arguments for initializing dataset
    parser.add_argument("--train_data_path", type = str, default = '/home/leon/DeepLearning/Project/Dataset/train', help = 'Path to load the training data')
    parser.add_argument("--val_data_path", type = str, default = '/home/leon/DeepLearning/Project/Dataset/val', help = 'Path to load the validation data')
    parser.add_argument("--img_size", type = int, default = 256, help = 'Height and weight of the images the networks will process')
    parser.add_argument("--km_file_path", type = str, default = './pts_in_hull.npy', help = 'Extra file for mapping color pairs in ab channels into Q(313) categories')
    # Arguments for initializing dataLoader
    parser.add_argument('--batch_size', type = int, default = 2)
    parser.add_argument('--num_workers', type = int, default = 4)
    # Arguments for initializing networks
    parser.add_argument('--use_memory', type = bool, default = True, help = 'Use memory or not')
    parser.add_argument('--use_feat_integrator', type = bool, default = False, help = 'Use feature_integrator or not')
    parser.add_argument("--mem_size", type = int, default = 120, help = 'The number of color and spatial features that will be stored in the memory_network respectively')
    parser.add_argument("--color_feat_dim", type = int, default = 313, help = 'Dimension of color feaures extracted from an image')
    parser.add_argument("--spatial_feat_dim", type = int, default = 512, help = 'Dimension of spatial feaures extracted from an image')
    parser.add_argument("--top_k", type = int, default = 64, help = 'Select the top k spatial feaures in memory_network which relate to input query')
    parser.add_argument("--alpha", type = float, default = 0.1, help = 'Bias term in the unsupervised loss')
    parser.add_argument('--gen_norm', type = str, default = 'adain', choices = ['batch', 'adain'], help = 'Defines the type of normalization used in the generator.')
    parser.add_argument('--dis_norm', type = str, default = 'adain', choices=['None', 'adain'], help = 'Defines the type of normalization used in the discriminator.')
    parser.add_argument('--apply_weight_init', type = bool, default = True, choices = [True, False])
    # Arguments for setting the optimizers
    parser.add_argument('--base_lr_gen', type = float, default = 3e-4, help = 'Base learning rate for the generator.')
    parser.add_argument('--base_lr_disc', type = float, default = 6e-5, help = 'Base learning rate for the discriminator.')
    parser.add_argument('--base_lr_mem', type = float, default = 1e-4, help = 'Base learning rate for the memory network.')
    parser.add_argument('--base_lr_feat', type = float, default = 6e-6, help = 'Base learning rate for the feature integrator.')
    parser.add_argument('--lr_decay_rate', type = float, default = 0.1, help = 'Learning rate decay rate for both networks.')
    parser.add_argument('--lr_decay_steps', type = float, default = 6e4, help = 'Learning rate decay steps for both networks.')
    #Arguments for saving network parameters and real and fake images
    parser.add_argument('--save_path', type = str, default='../checkpoints', help = 'Save and load path for the network weights.')
    parser.add_argument('--save_freq', type = int, default = 50, help = 'Save frequency during training.')
    #Arguments for setting start epoch and end epoch
    parser.add_argument('--start_epoch', type = int, default = 0, help = 'If start_epoch>0, load previously saved weigth from the save_path.')
    parser.add_argument('--end_epoch', type = int, default = 200)
    # Rest arguments
    parser.add_argument("--color_thres", type = float, default = 0.7, help = 'Threshold for deciding the case 1 and case 2 in updating the memory_network')
    parser.add_argument('--l1_weight', type = float, default = 0.99)
    parser.add_argument('--smoothing', type = float, default = 0.9, help = 'Argument for implementing one-sided label smoothing')
    return parser.parse_args()

if __name__ == '__main__':
    args = define_arguments()
    # display arguments
    print_args(args)
    # train and validation
    train_and_validation(args)
