import torch
from torch.utils.data import Dataset
import os
import cv2
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from util import NNEncode, encode_313bin
from colorthief import ColorThief


class mydata(Dataset):
    def __init__(self, img_path, img_size, km_file_path, color_info, transform = None, NN = 10.0, sigma = 5.0):

        self.img_path = img_path
        self.img_size = img_size
        self.img = sorted(os.listdir(self.img_path))

        self.res_normalize_mean = [0.485, 0.456, 0.406]
        self.res_normalize_std = [0.229, 0.224, 0.225]
        self.transform = transform
        self.color_info = color_info
        if self.color_info == 'dist':
            self.nnenc = NNEncode(NN, sigma, km_filepath = km_file_path)

    def __len__(self):

        return len(self.img)

    def __getitem__(self, i):

        img_item = {}
        rgb_image = Image.open(os.path.join(self.img_path, self.img[i])).convert('RGB')
        test_image = cv2.imread(os.path.join(self.img_path, self.img[i]))
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
        res_input = cv2.resize(res_input, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        index = i + 0.0

        img_item['l_channel'] = np.transpose(l_image, (2, 0, 1)).astype(np.float32)
        img_item['ab_channel'] = np.transpose(ab_image, (2, 0, 1)).astype(np.float32)
        img_item['color_feat'] = color_feat.astype(np.float32)
        img_item['res_input'] = np.transpose(res_input, (2, 0, 1)).astype(np.float32)
        img_item['index'] = np.array(([index])).astype(np.float32)[0]


        #return img_item
        new_lab_image = preprocess(test_image)
        #new_lab_image = np.transpose(new_lab_image, (2, 0, 1)).astype(np.float32)
        real_bgr = postprocess(new_lab_image)
        cv2.imshow("image", real_bgr)
        cv2.waitKey(0)
        return new_lab_image

def preprocess(img_bgr):
    # to 32bit img
    img_bgr = img_bgr.astype(np.float32)/255.0
    # transform to lab
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    # normalize
    img_lab[:, :, 0] = img_lab[:, :, 0]/50 - 1
    img_lab[:, :, 1] = img_lab[:, :, 1]/127
    img_lab[:, :, 2] = img_lab[:, :, 2]/127
    # transpose
    img_lab = img_lab.transpose((2, 0, 1))
    return img_lab

def postprocess(img_lab):
    # transpose back
    img_lab = img_lab.transpose((1, 2, 0))
    # transform back
    img_lab[:, :, 0] = (img_lab[:, :, 0] + 1)*50
    img_lab[:, :, 1] = img_lab[:, :, 1]*127
    img_lab[:, :, 2] = img_lab[:, :, 2]*127
    # transform to bgr
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    # to int8
    img_bgr = (img_bgr*255.0).astype(np.uint8)
    return img_bgr
