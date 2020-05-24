import numpy as np
import argparse
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from matplotlib import pyplot as plt
import os
from networks import Generator, Discriminator, Memory_Network
import torch
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from util import NNEncode, encode_313bin
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import copy

def print_args(args):
    """Display args."""
    arg_list = str(args)[10:-1].split(',')
    for arg in arg_list:
        print(arg.strip())
    print('')

class customed_dataset(Dataset):
    def __init__(self, img_list, img_size, km_file_path, transform = None, NN = 20.0, sigma = 5.0):

        self.img_list = img_list
        self.img_size = img_size

        self.res_normalize_mean = [0.485, 0.456, 0.406]
        self.res_normalize_std = [0.229, 0.224, 0.225]
        self.transform = transform
        self.nnenc = NNEncode(NN, sigma, km_filepath = km_file_path)

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, i):

        img_item = {}
        rgb_image = cv2.cvtColor(self.img_list[i], cv2.COLOR_BGR2RGB)
        rgb_image = Image.fromarray(rgb_image)
        w, h = rgb_image.size
        if w != h:
            min_val = min(w, h)
            rgb_image = rgb_image.crop((w // 2 - min_val // 2, h // 2 - min_val // 2, w // 2 + min_val // 2, h // 2 + min_val // 2))

        rgb_image = np.array(rgb_image.resize((self.img_size, self.img_size), Image.LANCZOS))

        lab_image = rgb2lab(rgb_image)
        l_image = lab_image[:,:,:1]
        ab_image = lab_image[:,:,1:]

        color_feat = encode_313bin(np.expand_dims(ab_image, axis = 0), self.nnenc)[0]
        color_feat = np.mean(color_feat, axis = (0, 1))

        gray_image = [lab_image[:,:,:1]]
        h, w, c = lab_image.shape
        gray_image.append(np.zeros(shape = (h, w, 2)))
        gray_image = np.concatenate(gray_image, axis = 2)

        res_input = lab2rgb(gray_image)
        res_input = (res_input - self.res_normalize_mean) / self.res_normalize_std
        res_input = resize(res_input, (224, 224))

        img_item['img_l'] = np.transpose(l_image, (2, 0, 1)).astype(np.float32)
        img_item['img_ab'] = np.transpose(ab_image, (2, 0, 1)).astype(np.float32)
        img_item['color_feat'] = color_feat.astype(np.float32)
        img_item['res_input'] = np.transpose(res_input, (2, 0, 1)).astype(np.float32)

        return img_item

def initialization(args):
    """Create the generators for testing purposes."""
    # check CUDA availability and set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    mem = Memory_Network(mem_size = args.mem_size, color_feat_dim = args.color_feat_dim, spatial_feat_dim = args.spatial_feat_dim, top_k = args.top_k, alpha = args.alpha)
    generator = Generator(args.color_feat_dim, args.img_size, args.gen_norm)

    if args.use_memory == True:
        ### Load the pre-trained model
        mem_checkpoint = torch.load(args.mem_model_path)
        mem.load_state_dict(mem_checkpoint['mem_model'])
        mem.sptial_key = mem_checkpoint['mem_key']
        mem.color_value = mem_checkpoint['mem_value']
        mem.age = mem_checkpoint['mem_age']
        mem.img_id = mem_checkpoint['img_id']

    print(mem.img_id.cpu().numpy())
    generator.load_state_dict(torch.load(args.generator_model_path))

    if args.use_memory == True:
        mem.to(device)
        mem.spatial_key = mem.sptial_key.to(device)
        mem.color_value = mem.color_value.to(device)
        mem.age = mem.age.to(device)
    generator.to(device)

    mem = mem.eval()
    generator = generator.eval()

    return generator, mem, device

def render_images(args, generator, mem, device, dataloader):
    for idx, batch in enumerate(dataloader):

        res_input = batch['res_input'].to(device)
        color_feat = batch['color_feat'].to(device)
        img_l = (batch['img_l'] / 100.0).to(device)
        img_ab = (batch['img_ab'] / 110.0).to(device)

        if args.use_memory == True:
            query = mem(res_input)
            top_features, ref_img_ids = mem.topk_feature(query, 3)
            top1_feature= top_features[:, 0, :]
            top2_feature= top_features[:, 1, :]
            top3_feature= top_features[:, 2, :]
        fake_img_ab_top1 = generator(img_l, top1_feature).detach()
        fake_img_ab_top2 = generator(img_l, top2_feature).detach()
        fake_img_ab_top3 = generator(img_l, top3_feature).detach()
        fake_img_ab_best = generator(img_l, color_feat).detach()

        real_image = torch.cat([img_l, img_ab], dim = 1)
        fake_image_top1 = torch.cat([img_l, fake_img_ab_top1], dim = 1)
        fake_image_top2 = torch.cat([img_l, fake_img_ab_top2], dim = 1)
        fake_image_top3 = torch.cat([img_l, fake_img_ab_top3], dim = 1)
        fake_image_best = torch.cat([img_l, fake_img_ab_best], dim = 1)

        if idx == 0:
            real_images = real_image
            fake_images_top1 = fake_image_top1
            fake_images_top2 = fake_image_top2
            fake_images_top3 = fake_image_top3
            fake_images_best = fake_image_best
        else:
            real_images = torch.cat([real_images, real_image], dim = 0)
            fake_images_top1 = torch.cat([fake_images_top1, fake_image_top1], dim = 0)
            fake_images_top2 = torch.cat([fake_images_top2, fake_image_top2], dim = 0)
            fake_images_top3 = torch.cat([fake_images_top3, fake_image_top3], dim = 0)
            fake_images_best = torch.cat([fake_images_best, fake_image_best], dim = 0)

    return real_images, fake_images_top1, fake_images_top2, fake_images_top3, fake_images_best, ref_img_ids

def get_imgs_from_id(img_ids, img_size):
    imgs_bgr = []
    for i in range(len(img_ids)):
        print(str(int(img_ids[i]))[:])
        img_path = '/home/leon/DeepLearning/Project/Dataset/train/raw_image_Romantic_' + str(int(img_ids[i]))[:] + '.png'
        print(img_path)
        img_bgr = cv2.imread(img_path)
        img_bgr = cv2.resize(img_bgr, (img_size, img_size))
        imgs_bgr.append(img_bgr)
    return imgs_bgr

def postprocess(img_lab, frame_width, frame_height):
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
    img_bgr = cv2.resize(img_bgr, (frame_width // 3, frame_height // 3))
    return img_bgr

def processing(args, frame_list, frame_width, frame_height, out, generator, mem, device, memory_canvas):
    dataset = customed_dataset(frame_list, img_size = args.img_size, km_file_path = args.km_file_path)
    print('Dataset len: {}'.format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, drop_last = False)
    real_images, fake_images_top1, fake_images_top2, fake_images_top3, fake_images_best, img_ids = render_images(args, generator, mem, device, dataloader)
    real_images = real_images.cpu().numpy()
    fake_images_top1 = fake_images_top1.cpu().numpy()
    fake_images_top2 = fake_images_top2.cpu().numpy()
    fake_images_top3 = fake_images_top3.cpu().numpy()
    fake_images_best = fake_images_best.cpu().numpy()
    img_ids = img_ids.cpu().numpy()

    for i in range(len(frame_list)):
        memory_canvas_copy = copy.copy(memory_canvas)
        indexes = get_index_from_id(img_ids[i], mem)
        frame = frame_list[i]
        real_bgr = postprocess(real_images[i], frame_width, frame_height)
        fake_bgr_top1 = postprocess(fake_images_top1[i], frame_width, frame_height)
        fake_bgr_top2 = postprocess(fake_images_top2[i], frame_width, frame_height)
        fake_bgr_top3 = postprocess(fake_images_top3[i], frame_width, frame_height)
        fake_bgr_best = postprocess(fake_images_best[i], frame_width, frame_height)
        dsize = (256, 256)
        frame = cv2.resize(frame, dsize)
        frame = cv2.resize(frame, (frame_width // 3, frame_height // 3))
        frame_gray = np.expand_dims(cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY), 2)
        cv2.imshow('Frame',frame)
        #canvas = np.ones((3 * frame_height + 4 * 6, 2 * frame_width + (2 + 1)*6, 3), dtype=np.uint8)*255
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        canvas[0: frame_height // 3, 0: frame_width // 3, :] = frame
        canvas[0: frame_height // 3, frame_width // 3: frame_width * 2 // 3, :] = np.repeat(frame_gray, 3, axis=2)
        canvas[frame_height // 3: frame_height * 2 // 3, 0: frame_width // 3, :] = fake_bgr_top1
        canvas[frame_height // 3: frame_height * 2 // 3, frame_width // 3: frame_width * 2 // 3, :] = fake_bgr_top2
        canvas[frame_height * 2 // 3: frame_height, 0: frame_width // 3, :] = fake_bgr_top3
        canvas[frame_height * 2 // 3: frame_height, frame_width // 3: frame_width * 2 // 3, :] = fake_bgr_best
        canvas = cv2.rectangle(canvas, (0, frame_height // 3), ( frame_width // 3, frame_height * 2 // 3), color = (255, 0, 0), thickness = 2)
        canvas = cv2.rectangle(canvas, (frame_width // 3, frame_height // 3), (frame_width * 2 // 3, frame_height * 2 // 3), color = (0, 255, 0), thickness = 2)
        canvas = cv2.rectangle(canvas, (0, frame_height * 2 // 3), (frame_width // 3, frame_height), color = (0, 0, 255), thickness = 2)
        memory_canvas_copy = cv2.rectangle(memory_canvas_copy, (indexes[0] % 20 * 32, indexes[0] // 20 * 32), (indexes[0] % 20 * 32 + 32, indexes[0] // 20 * 32 + 32), color = (255, 0, 0), thickness = 2)
        memory_canvas_copy = cv2.rectangle(memory_canvas_copy, (indexes[1] % 20 * 32, indexes[1] // 20 * 32), (indexes[1] % 20 * 32 + 32, indexes[1] // 20 * 32 + 32), color = (0, 255, 0), thickness = 2)
        memory_canvas_copy = cv2.rectangle(memory_canvas_copy, (indexes[2] % 20 * 32, indexes[2] // 20 * 32), (indexes[2] % 20 * 32 + 32, indexes[2] // 20 * 32 + 32), color = (0, 0, 255), thickness = 2)
        canvas[8: frame_height - 8, 1280: frame_width, :] = memory_canvas_copy
        canvas = cv2.resize(canvas, None, fx = 1.0, fy = 1.0, interpolation = cv2.INTER_NEAREST)
        # paint
        # canvas[6: 6 + frame_height, 6: 6 + frame_width, :] = frame
        # canvas[12 + frame_height: 12 + 2 * frame_height, 6: 6 + frame_width, :] = frame
        # canvas[18 + 2 * frame_height: 18 + 3 * frame_height, 6: 6 + frame_width, :] = frame
        out.write(canvas)

def get_index_from_id(img_ids, mem, img_size = 32):
    indexes = []
    mem_indexes = mem.img_id.cpu().numpy()
    for i in range(len(img_ids)):
        print(str(int(img_ids[i]))[:])
        index = np.where(mem_indexes == img_ids[i])[0][0]
        indexes.append(index)
    return indexes

def get_imgs_from_id(img_ids, img_size = 32):
    imgs_bgr = []
    for i in range(len(img_ids)):
        print(str(int(img_ids[i]))[:])
        img_path = '/home/leon/DeepLearning/Project/Dataset/train/raw_image_Romantic_' + str(int(img_ids[i]))[:] + '.png'
        print(img_path)
        img_bgr = cv2.imread(img_path)
        img_bgr = cv2.resize(img_bgr, (img_size, img_size))
        imgs_bgr.append(img_bgr)
    return imgs_bgr

def create_memory_canvas(mem):
    mem_imgs = mem.img_id.cpu().numpy()
    memory_canvas = np.zeros((800, 640, 3), dtype=np.uint8)
    ref_imgs = get_imgs_from_id(mem_imgs)
    count = 0
    for i in range(0, 25):
        for j in range(0, 20):
            row = i * 32
            column = j * 32
            memory_canvas[row: row + 32, column: column + 32, :] = ref_imgs[count]
            count += 1
    return memory_canvas

def create_video(args):
    video_name = 'Romantic'
    base_folder = '/home/leon/DeepLearning/Project/'
    demo_folder = base_folder + '/demo'
    if not os.path.exists(demo_folder):
        os.mkdir(demo_folder)

    generator, mem, device = initialization(args)
    if args.use_memory == True:
        memory_canvas = create_memory_canvas(mem)

    cap = cv2.VideoCapture('/home/leon/DeepLearning/Project/' + video_name + '.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('demo2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 23.98, (frame_width, frame_height))

    frame_list = []
    while(cap.isOpened()):
        ret, frame = cap.read()

        if np.shape(frame) == ():
            print(len(frame_list))
            print(frame_list[-1])
            processing(args, frame_list, frame_width, frame_height, out, generator, mem, device, memory_canvas)
            break

        frame_list.append(frame)
        if len(frame_list) == args.batch_size:
            processing(args, frame_list, frame_width, frame_height, out, generator, mem, device, memory_canvas)
            frame_list = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description='Animation Colorization with Memory-Augmented Networks and a Few Shots.')
    # Arguments for initializing dataset
    parser.add_argument('--data_path', type=str, default='/home/leon/DeepLearning/Project/Dataset/DogTrouble/test')
    parser.add_argument("--img_size", type = int, default = 128, help = 'Height and weight of the images the networks will process')
    parser.add_argument("--km_file_path", type = str, default = './pts_in_hull.npy', help = 'Extra file for mapping color pairs in ab channels into Q(313) categories')
    # Arguments for initializing dataLoader
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_workers', type = int, default = 4)
    # Arguments for initializing networks
    parser.add_argument('--use_memory', type = bool, default = True, help = 'Use memory or not')
    parser.add_argument("--mem_size", type = int, default = 360, help = 'The number of color and spatial features that will be stored in the memory_network respectively')
    parser.add_argument("--color_feat_dim", type = int, default = 313, help = 'Dimension of color feaures extracted from an image')
    parser.add_argument("--spatial_feat_dim", type = int, default = 512, help = 'Dimension of spatial feaures extracted from an image')
    parser.add_argument("--top_k", type = int, default = 256, help = 'Select the top k spatial feaures in memory_network which relate to input query')
    parser.add_argument("--alpha", type = float, default = 0.1, help = 'Bias term in the unsupervised loss')
    parser.add_argument('--gen_norm', type = str, default = 'adain', choices = ['batch', 'adain'], help = 'Defines the type of normalization used in the generator.')
    parser.add_argument('--save_path', type=str, default='/home/leon/DeepLearning/Project/Dataset/DogTrouble/result', help='Save path for the test imgs.')
    # Arguments for loading the trained networks
    parser.add_argument("--mem_model_path", type = str, default = '/home/leon/DeepLearning/Project/checkpoints/checkpoint_ep199_mem.pt')
    parser.add_argument("--generator_model_path", type = str, default = '/home/leon/DeepLearning/Project/checkpoints/checkpoint_ep199_gen.pt')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    # display args
    print_args(args)
    create_video(args)
