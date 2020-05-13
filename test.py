import os
import argparse
import torch
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from torch.utils.data import DataLoader
from memory_network import Memory_Network
from networks import Generator, unet_generator, Discriminator, Discriminator2, weights_init_normal
from helpers import save_test_sample, print_args
from dataset import mydata

def init_test(args):
    """Create the data loader and the generators for testing purposes."""
    # create loader
    test_dataset = mydata(img_path = args.data_path, img_size = args.img_size, km_file_path = args.km_file_path, color_info = args.color_info)
    print('Test dataset len: {}'.format(len(test_dataset)))
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last = False)

    # check CUDA availability and set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    mem = Memory_Network(mem_size = args.mem_size, color_info = args.color_info, color_feat_dim = args.color_feat_dim, spatial_feat_dim = 512, alpha = args.alpha)
    generator = unet_generator(1, 2, args.n_feats, args.color_feat_dim)

    ### Load the pre-trained model
    mem_checkpoint = torch.load(args.mem_model)
    mem.load_state_dict(mem_checkpoint['mem_model'])
    mem.sptial_key = mem_checkpoint['mem_key']
    mem.color_value = mem_checkpoint['mem_value']
    mem.age = mem_checkpoint['mem_age']
    mem.top_index = mem_checkpoint['mem_index']

    generator.load_state_dict(torch.load(args.generator_model))

    mem.to(device)
    mem.spatial_key = mem.sptial_key.to(device)
    mem.color_value = mem.color_value.to(device)
    mem.age = mem.age.to(device)
    generator.to(device)

    generator = generator.eval()

    return generator, mem, test_dataloader, device

def run_test(args):
    """Run the networks on the test set, and save/show the samples."""
    generator, mem, test_dataloader, device = init_test(args)

    # run through the dataset and display the first few images of every batch
    for idx, batch in enumerate(test_dataloader):

        res_input = batch['res_input'].to(device)
        color_feat = batch['color_feat'].to(device)
        index = batch['index'].to(device)
        img_l = (batch['l_channel'] / 100.0).to(device)
        img_ab = (batch['ab_channel'] / 110.0).to(device)

        query = mem(res_input)
        top1_feature, _ = mem.topk_feature(query, 1)
        top1_feature = top1_feature[:, 0, :]
        fake_img_ab = generator(img_l, top1_feature).detach()

        real_image = torch.cat([img_l, img_ab], dim = 1)
        fake_image = torch.cat([img_l, fake_img_ab], dim = 1)

        print('sample {}/{}'.format(idx + 1, len(test_dataloader) + 1))
        save_test_sample(real_image, fake_image, args.img_size,
                         os.path.join(args.save_path, 'test_sample_{}.png'.format(idx)), show=False)


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description='Image colorization with GANs.')
    parser.add_argument('--data_path', type=str, default='/home/leon/DeepLearning/Project/Dataset/DogTrouble/test')
    parser.add_argument("--result_path", type = str, default = '/home/leon/DeepLearning/Project/Dataset/DogTrouble/result')
    parser.add_argument('--save_path', type=str, default='/home/leon/DeepLearning/Project/Dataset/DogTrouble/result', help='Save path for the test imgs.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
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
    parser.add_argument("--mem_model", type = str, default = '/home/leon/DeepLearning/Project/checkpoints/checkpoint_ep50mem.pt')
    parser.add_argument("--generator_model", type = str, default = '/home/leon/DeepLearning/Project/checkpoints/checkpoint_ep50_gen.pt')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()

    # display args
    print_args(args)

    run_test(args)
