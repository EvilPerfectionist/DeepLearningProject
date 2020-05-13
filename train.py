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

    generator2 = generator2.train()
    discriminator2 = discriminator2.train()

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

                if phase == 'test':
                    top1_feature, _ = mem.topk_feature(res_feature, 1)
                    color_feat = top1_feature[:, 0, :]

                ### 3) Train Generator
                if phase == 'train':
                    optimizers['gen2'].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    dis_color_feat = torch.cat([torch.unsqueeze(color_feat, 2) for _ in range(args.img_size)], dim = 2)
                    dis_color_feat = torch.cat([torch.unsqueeze(dis_color_feat, 3) for _ in range(args.img_size)], dim = 3)
                    fake_img_ab = generator2(img_l, color_feat)
                    fake = discriminator2(fake_img_ab, img_l, dis_color_feat)
                    fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1).to(device)

                    g_loss_GAN = losses['disc'](fake, target_ones)
                    g_loss_smoothL1 = losses['smoothl1'](fake_img_ab, img_ab)
                    g_loss = g_loss_GAN + g_loss_smoothL1

                    if phase == 'train':
                        g_loss.backward()
                        optimizers['gen2'].step()

                epoch_gen_adv_loss += g_loss_GAN.item()
                epoch_gen_l1_loss += g_loss_smoothL1.item()

                ### 4) Train Discriminator
                if phase == 'train':
                    optimizers['disc2'].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    prediction_real = discriminator2(img_ab, img_l, dis_color_feat)
                    prediction_fake = discriminator2(fake_img_ab.detach(), img_l, dis_color_feat)

                    d_loss_real = losses['disc'](prediction_real, target_ones * args.smoothing)
                    d_loss_fake = losses['disc'](prediction_fake, target_zeros)
                    d_loss = d_loss_real + d_loss_fake

                    if phase == 'train':
                        d_loss.backward()
                        optimizers['disc2'].step()

                epoch_disc_real_loss += d_loss_real.item()
                epoch_disc_fake_loss += d_loss_fake.item()
                epoch_disc_real_acc += np.mean(prediction_real.detach().cpu().numpy() > 0.5)
                epoch_disc_fake_acc += np.mean(prediction_fake.detach().cpu().numpy() <= 0.5)

                # save the first sample for later
                if phase == 'test' and idx == 0:
                    sample_real_img_lab = real_img_lab
                    sample_fake_img_lab = fake_img_lab

                # display losses
                print_losses(epoch_gen_adv_loss, epoch_gen_l1_loss,
                             epoch_disc_real_loss, epoch_disc_fake_loss,
                             epoch_disc_real_acc, epoch_disc_fake_acc,
                             len(data_loaders[phase]), args.l1_weight)

                if phase == 'test':
                    if epoch % args.save_freq == 0 or epoch == args.max_epoch - 1:
                        gen_path = os.path.join(args.save_path, 'checkpoint_ep{}_gen.pt'.format(epoch))
                        disc_path = os.path.join(args.save_path, 'checkpoint_ep{}_disc.pt'.format(epoch))
                        mem_path = os.path.join(args.save_path, 'checkpoint_ep{}mem.pt'.format(epoch))
                        torch.save(generator2.state_dict(), gen_path)
                        torch.save(discriminator2.state_dict(), disc_path)
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
    parser.add_argument("--train_data_path", type = str, default = '/home/leon/DeepLearning/Project/Dataset/DogTrouble/')
    parser.add_argument("--test_data_path", type = str, default = '/home/leon/DeepLearning/Project/Dataset/DogTrouble/')
    parser.add_argument("--img_size", type = int, default = 128)
    parser.add_argument("--km_file_path", type = str, default = './pts_in_hull.npy')
    parser.add_argument("--n_feats", type = int, default = 64)
    parser.add_argument("--color_feat_dim", type = int, default = 313)
    parser.add_argument("--spatial_feat_dim", type = int, default = 512)
    parser.add_argument("--mem_size", type = int, default = 360)
    parser.add_argument("--alpha", type = float, default = 0.1)
    parser.add_argument("--top_k", type = int, default = 256)
    parser.add_argument("--color_thres", type = float, default = 0.7)
    parser.add_argument("--test_freq", type = int, default = 2)
    parser.add_argument("--color_info", type = str, default = 'dist', help = 'option should be dist or RGB')
    parser.add_argument('--save_path', type=str, default='../checkpoints',
                        help='Save and load path for the network weights.')
    parser.add_argument('--save_freq', type=int, default=50, help='Save frequency during training.')
    parser.add_argument('--batch_size', type=int, default=8)
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

def test_operation(args, generator, mem, te_dataloader, device, e = -1):

    count = 0
    result_path = os.path.join(args.result_path, args.data_name)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    with torch.no_grad():
        for i, batch in enumerate(te_dataloader):
            res_input = batch['res_input'].to(device)
            color_feat = batch['color_feat'].to(device)
            l_channel = (batch['l_channel'] / 100.0).to(device)
            ab_channel = (batch['ab_channel'] / 110.0).to(device)

            bs = res_input.size()[0]

            query = mem(res_input)
            top1_feature, _ = mem.topk_feature(query, 1)
            top1_feature = top1_feature[:, 0, :]
            result_ab_channel = generator(l_channel, top1_feature)

            real_image = torch.cat([l_channel * 100, ab_channel * 110], dim = 1).cpu().numpy()
            fake_image = torch.cat([l_channel * 100, result_ab_channel * 110], dim = 1).cpu().numpy()
            gray_image = torch.cat([l_channel * 100, torch.zeros((bs, 2, args.img_size, args.img_size)).to(device)], dim = 1).cpu().numpy()

            all_img = np.concatenate([real_image, fake_image, gray_image], axis = 2)
            all_img = np.transpose(all_img, (0, 2, 3, 1))
            rgb_imgs = [lab2rgb(ele) for ele in all_img]
            rgb_imgs = np.array((rgb_imgs))
            rgb_imgs = (rgb_imgs * 255.0).astype(np.uint8)

            for t in range(len(rgb_imgs)):

                if e > -1 :
                    img = Image.fromarray(rgb_imgs[t])
                    name = '%03d_%04d_result.png'%(e, count)
                    img.save(os.path.join(result_path, name))

                else:
                    name = '%04d_%s.png'
                    img = rgb_imgs[t]
                    h, w, c = img.shape
                    stride = h // 3
                    original = img[:stride, :, :]
                    original = Image.fromarray(original)
                    original.save(os.path.join(result_path, name%(count, 'GT')))

                    result = img[stride : 2*stride, :, :]
                    result = Image.fromarray(result)
                    result.save(os.path.join(result_path, name%(count, 'result')))

                    if not args.test_only:
                        gray_img = img[2*stride :, :, :]
                        gray_img = Image.fromarray(gray_img)
                        gray_img.save(os.path.join(result_path, name%(count, 'gray')))

                count = count + 1

if __name__ == '__main__':
    args = get_arguments()

    # display arguments
    print_args(args)

    run_training(args)
