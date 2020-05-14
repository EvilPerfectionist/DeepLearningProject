import os
import argparse
import torch
from torch.utils.data import Dataset
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from torch.utils.data import DataLoader
from memory_network import Memory_Network
from networks import Generator, unet_generator, Discriminator, Discriminator2, weights_init_normal
from helpers import save_test_sample, print_args
import numpy as np
from util import NNEncode, encode_313bin
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize

class mydata(Dataset):
    def __init__(self, img_list, img_size, km_file_path, color_info, transform = None, NN = 20.0, sigma = 5.0):

        self.img_list = img_list
        self.img_size = img_size

        self.res_normalize_mean = [0.485, 0.456, 0.406]
        self.res_normalize_std = [0.229, 0.224, 0.225]
        self.transform = transform
        self.color_info = color_info
        if self.color_info == 'dist':
            self.nnenc = NNEncode(NN, sigma, km_filepath = km_file_path)

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, i):

        img_item = {}
        rgb_image = cv2.cvtColor(self.img_list[i], cv2.COLOR_BGR2RGB)
        # cv2.imshow('sample', rgb_image)
        # cv2.waitKey(0)
        rgb_image = Image.fromarray(rgb_image)
        w, h = rgb_image.size
        if w != h:
            min_val = min(w, h)
            rgb_image = rgb_image.crop((w // 2 - min_val // 2, h // 2 - min_val // 2, w // 2 + min_val // 2, h // 2 + min_val // 2))

        rgb_image = np.array(rgb_image.resize((self.img_size, self.img_size), Image.LANCZOS))

        lab_image = rgb2lab(rgb_image)
        l_image = lab_image[:,:,:1]
        ab_image = lab_image[:,:,1:]

        if self.color_info == 'dist':
            color_feat = encode_313bin(np.expand_dims(ab_image, axis = 0), self.nnenc)[0]
            color_feat = np.mean(color_feat, axis = (0, 1))

        elif self.color_info == 'RGB':
            color_thief = ColorThief(os.path.join(self.img_path, self.img[i]))

            ## color_feat shape : (10, 3)
            color_feat = np.array(color_thief.get_palette(11))

            ## color_feat shape : (3, 10)
            color_feat = np.transpose(color_feat)

            ## color_feat shape : (30)
            color_feat = np.reshape(color_feat, (30)) / 255.0
            del color_thief

        gray_image = [lab_image[:,:,:1]]
        h, w, c = lab_image.shape
        gray_image.append(np.zeros(shape = (h, w, 2)))
        gray_image = np.concatenate(gray_image, axis = 2)

        res_input = lab2rgb(gray_image)
        res_input = (res_input - self.res_normalize_mean) / self.res_normalize_std
        res_input = resize(res_input, (224, 224))

        index = i + 0.0

        img_item['l_channel'] = np.transpose(l_image, (2, 0, 1)).astype(np.float32)
        img_item['ab_channel'] = np.transpose(ab_image, (2, 0, 1)).astype(np.float32)
        img_item['color_feat'] = color_feat.astype(np.float32)
        img_item['res_input'] = np.transpose(res_input, (2, 0, 1)).astype(np.float32)
        img_item['index'] = np.array(([index])).astype(np.float32)[0]

        return img_item

def initialization():
    """Create the generators for testing purposes."""
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

    return generator, mem, device

def calculate_diff(frame1, frame2):
    diff = 0.0
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr1 = cv2.calcHist([frame1], [i], None, [256], [0,256])
        histr2 = cv2.calcHist([frame2], [i], None, [256], [0,256])
        diff += np.square(np.subtract(histr1.flatten(), histr2.flatten())).mean()
    #print(diff / 3)
    return diff

def create_gif(args):
    video_name = 'DogTrouble'
    cap = cv2.VideoCapture('/home/leon/DeepLearning/Project/' + video_name + '.avi')
    frame_list = []
    diff_list = []
    generator, mem, device = initialization()
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        dsize = (256, 256)
        frame = cv2.resize(frame, dsize)
        frame_list.append(frame)
        if len(frame_list) > 2:
            diff = calculate_diff(frame_list[-1], frame_list[-2])
            diff_list.append(diff)
            if sum(i > 5000.0 for i in diff_list) == 2:
                if len(frame_list) < 20:
                    frame_list = []
                    diff_list = []
                else:
                    dataset = mydata(frame_list, img_size = args.img_size, km_file_path = args.km_file_path, color_info = args.color_info)
                    print('Dataset len: {}'.format(len(dataset)))
                    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, drop_last = False)
                    generate_gif(args, generator, mem, device, dataloader, count)
                    count += 1
                    frame_list = []
                    diff_list = []

def generate_gif(args, generator, mem, device, dataloader, count):

    # run through the dataset and display the first few images of every batch
    for idx, batch in enumerate(dataloader):

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

        if idx == 0:
            real_images = real_image
            fake_images = fake_image
        else:
            real_images = torch.cat([real_images, real_image], dim = 0)
            fake_images = torch.cat([fake_images, fake_image], dim = 0)

        print('sample {}/{}'.format(idx + 1, len(dataloader) + 1))
        # save_test_sample(real_image, fake_image, args.img_size,
        #                  os.path.join(args.save_path, 'test_sample_{}.png'.format(idx)), show=False)
    save_gif(real_images, fake_images, args.img_size, args.save_path, count)

def postprocess(img_lab):
    # transpose back
    img_lab = img_lab.transpose((1, 2, 0))
    # transform back
    img_lab[:, :, 0] = img_lab[:, :, 0] * 100
    img_lab[:, :, 1] = img_lab[:, :, 1] * 110
    img_lab[:, :, 2] = img_lab[:, :, 2] * 110
    # transform to bgr
    img_rgb = lab2rgb(img_lab)
    # to int8
    img_rgb = (img_rgb * 255.0).astype(np.uint8)
    img_rgb = Image.fromarray(img_rgb)
    return img_rgb

def postprocess2(img_lab):
    # transpose back
    img_lab = img_lab.transpose((1, 2, 0))
    # transform back
    img_lab[:, :, 0] = img_lab[:, :, 0] * 100
    img_lab[:, :, 1] = img_lab[:, :, 1] * 110
    img_lab[:, :, 2] = img_lab[:, :, 2] * 110
    # transform to bgr
    img_rgb = lab2rgb(img_lab)
    # to int8
    img_rgb = (img_rgb * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr

def save_gif(real_imgs_lab, fake_imgs_lab, img_size, save_path, count):
    """Create a grid of ground truth, grayscale and colorized images and save + display it to the user."""
    batch_size = real_imgs_lab.size()[0]

    real_imgs_lab = real_imgs_lab.cpu().numpy()
    fake_imgs_lab = fake_imgs_lab.cpu().numpy()

    real_imgs_rgb = []
    fake_imgs_rgb = []

    for i in range(0, batch_size):
        # postprocess real and fake samples
        real_rgb = postprocess(real_imgs_lab[i])
        fake_rgb = postprocess(fake_imgs_lab[i])
        real_imgs_rgb.append(real_rgb)
        fake_imgs_rgb.append(fake_rgb)

    real_gif_path = os.path.join(save_path, 'real_sample_{}.gif'.format(count))
    fake_gif_path = os.path.join(save_path, 'fake_sample_{}.gif'.format(count))
    real_imgs_rgb[0].save(real_gif_path,
               save_all=True, append_images=real_imgs_rgb[1:], optimize=False, duration=40, loop=0)
    fake_imgs_rgb[0].save(fake_gif_path,
               save_all=True, append_images=fake_imgs_rgb[1:], optimize=False, duration=40, loop=0)

def save_test_sample(real_imgs_lab, fake_imgs_lab, img_size, save_path, plot_size=14, scale=1.6, show=False):
    """
    Create a grid of ground truth,
    grayscale and 2 colorized images (from different sources) and save + display it to the user.
    """
    batch_size = real_imgs_lab.size()[0]
    plot_size = min(plot_size, batch_size)

    # create white canvas
    canvas = np.ones((plot_size*img_size + (plot_size+1)*6, 3*img_size + 5*8, 3), dtype=np.uint8)*255

    real_imgs_lab = real_imgs_lab.cpu().numpy()
    fake_imgs_lab = fake_imgs_lab.cpu().numpy()

    for i in range(0, plot_size):
        # post-process real and fake samples
        real_bgr = postprocess2(real_imgs_lab[i])
        fake_bgr = postprocess2(fake_imgs_lab[i])
        grayscale = np.expand_dims(cv2.cvtColor(real_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY), 2)
        # paint
        x = (i+1)*6+i*img_size
        canvas[x:x+img_size, 8:8 + img_size, :] = real_bgr
        canvas[x:x+img_size, 16 + img_size:16 + 2 * img_size, :] = np.repeat(grayscale, 3, axis=2)
        canvas[x:x+img_size, 24 + 2 * img_size:24 + 3 * img_size, :] = fake_bgr

    # scale
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # save
    cv2.imwrite(os.path.join(save_path), canvas)

    if show:
        cv2.destroyAllWindows()
        cv2.imshow('sample', canvas)
        cv2.waitKey(10000)

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
    parser.add_argument("--mem_model", type = str, default = '/home/leon/DeepLearning/Project/checkpoints/checkpoint_ep199_mem.pt')
    parser.add_argument("--generator_model", type = str, default = '/home/leon/DeepLearning/Project/checkpoints/checkpoint_ep199_gen.pt')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    # display args
    print_args(args)
    create_gif(args)
