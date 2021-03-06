import os
import argparse
import torch
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from torch.utils.data import DataLoader
from networks import Generator, Memory_Network, Feature_Integrator
from helpers import save_test_sample, print_args
from dataset import customed_dataset

def init_test(args):
    """Create the data loader and the generators for testing purposes."""
    # create loader
    test_dataset = customed_dataset(img_path = args.data_path, img_size = args.img_size, km_file_path = args.km_file_path)
    print('Test dataset len: {}'.format(len(test_dataset)))
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last = False)

    # check CUDA availability and set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    if args.use_memory == True:
        mem = Memory_Network(mem_size = args.mem_size, color_feat_dim = args.color_feat_dim, spatial_feat_dim = args.spatial_feat_dim, top_k = args.top_k, alpha = args.alpha)
        if args.use_feat_integrator == True:
            feature_integrator = Feature_Integrator(3, 1, 200)
    generator = Generator(args.color_feat_dim, args.img_size, args.gen_norm)

    if args.use_memory == True:
        ### Load the pre-trained model
        mem_checkpoint = torch.load(args.mem_model_path)
        mem.load_state_dict(mem_checkpoint['mem_model'])
        mem.sptial_key = mem_checkpoint['mem_key']
        mem.color_value = mem_checkpoint['mem_value']
        mem.age = mem_checkpoint['mem_age']
        mem.img_id = mem_checkpoint['img_id']

        if args.use_feat_integrator == True:
            feature_integrator.load_state_dict(torch.load(args.feat_model_path))

    generator.load_state_dict(torch.load(args.mem_generator_model_path))

    if args.use_memory == True:
        mem.to(device)
        mem.spatial_key = mem.sptial_key.to(device)
        mem.color_value = mem.color_value.to(device)
        mem.age = mem.age.to(device)

        if args.use_feat_integrator == True:
            feature_integrator.to(device)
    generator.to(device)


    generator = generator.eval()
    if args.use_memory == True:
        mem = mem.eval()

        if args.use_feat_integrator == True:
                feature_integrator = feature_integrator.eval()

    if args.use_memory == True:
        if args.use_feat_integrator == True:
            return generator, mem, feature_integrator, test_dataloader, device
        else:
            return generator, mem, test_dataloader, device
    else:
        return generator, test_dataloader, device

def run_test(args):
    """Run the networks on the test set, and save/show the samples."""
    if args.use_memory == True:
        if args.use_feat_integrator == True:
            generator, mem, feature_integrator, test_dataloader, device = init_test(args)
        else:
            generator, mem, test_dataloader, device = init_test(args)
    else:
        generator, test_dataloader, device = init_test(args)

    # run through the dataset and display the first few images of every batch
    for idx, batch in enumerate(test_dataloader):

        res_input = batch['res_input'].to(device)
        color_feat = batch['color_feat'].to(device)
        img_l = (batch['img_l'] / 100.0).to(device)
        img_ab = (batch['img_ab'] / 110.0).to(device)
        img_id = batch['img_id'].to(device)

        if args.use_memory == True:
            query = mem(res_input)
            top_features, ref_img_ids = mem.topk_feature(query, 3)
            color_feat = top_features[:, 0, :]
            if args.use_feat_integrator == True:
                top_features = torch.transpose(top_features, dim0 = 1, dim1 = 2)
                color_feat = feature_integrator(top_features)
        fake_img_ab = generator(img_l, color_feat).detach()

        real_image = torch.cat([img_l, img_ab], dim = 1)
        fake_image = torch.cat([img_l, fake_img_ab], dim = 1)

        print('sample {}/{}'.format(idx + 1, len(test_dataloader) + 1))
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_test_sample(real_image, fake_image, ref_img_ids, args.img_size,
                         os.path.join(args.save_path, 'test_sample_{}.png'.format(idx)),
                         os.path.join(args.save_path, 'ref_sample_{}.png'.format(idx)), show=False)


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description='Animation Colorization with Memory-Augmented Networks and a Few Shots.')
    # Arguments for initializing dataset
    parser.add_argument('--data_path', type=str, default='/home/leon/DeepLearning/Project/Dataset/test')
    parser.add_argument("--img_size", type = int, default = 128, help = 'Height and weight of the images the networks will process')
    parser.add_argument("--km_file_path", type = str, default = './pts_in_hull.npy', help = 'Extra file for mapping color pairs in ab channels into Q(313) categories')
    # Arguments for initializing dataLoader
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--num_workers', type = int, default = 4)
    # Arguments for initializing networks
    parser.add_argument('--use_memory', type = bool, default = True, help = 'Use memory or not')
    parser.add_argument('--use_feat_integrator', type = bool, default = False, help = 'Use feature_integrator or not')
    parser.add_argument("--mem_size", type = int, default = 1200, help = 'The number of color and spatial features that will be stored in the memory_network respectively')
    parser.add_argument("--color_feat_dim", type = int, default = 313, help = 'Dimension of color feaures extracted from an image')
    parser.add_argument("--spatial_feat_dim", type = int, default = 512, help = 'Dimension of spatial feaures extracted from an image')
    parser.add_argument("--top_k", type = int, default = 256, help = 'Select the top k spatial feaures in memory_network which relate to input query')
    parser.add_argument("--alpha", type = float, default = 0.1, help = 'Bias term in the unsupervised loss')
    parser.add_argument('--gen_norm', type = str, default = 'adain', choices = ['batch', 'adain'], help = 'Defines the type of normalization used in the generator.')
    parser.add_argument('--save_path', type=str, default='/home/leon/DeepLearning/Project/Dataset/result', help='Save path for the test imgs.')
    # Arguments for loading the trained networks
    parser.add_argument("--mem_model_path", type = str, default = '/home/leon/DeepLearning/Project/checkpoints/checkpoint_ep150_mem.pt')
    parser.add_argument("--mem_generator_model_path", type = str, default = '/home/leon/DeepLearning/Project/checkpoints/checkpoint_ep150_gen.pt')
    parser.add_argument("--feat_model_path", type = str, default = '/home/leon/DeepLearning/Project/checkpoints/checkpoint_ep50_feat.pt')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    # display args
    print_args(args)
    run_test(args)
