import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob
from scipy.interpolate import interp2d
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

# def preprocess(img_bgr):
#     # to 32bit img
#     img_bgr = img_bgr.astype(np.float32)/255.0
#     # transform to lab
#     img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
#     # normalize
#     img_lab[:, :, 0] = img_lab[:, :, 0]/50 - 1
#     img_lab[:, :, 1] = img_lab[:, :, 1]/127
#     img_lab[:, :, 2] = img_lab[:, :, 2]/127
#     # transpose
#     img_lab = img_lab.transpose((2, 0, 1))
#     return img_lab
#
#
# def postprocess(img_lab):
#     # transpose back
#     img_lab = img_lab.transpose((1, 2, 0))
#     # transform back
#     img_lab[:, :, 0] = (img_lab[:, :, 0] + 1)*50
#     img_lab[:, :, 1] = img_lab[:, :, 1]*127
#     img_lab[:, :, 2] = img_lab[:, :, 2]*127
#     # transform to bgr
#     img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
#     # to int8
#     img_bgr = (img_bgr*255.0).astype(np.uint8)
#     return img_bgr
#
# folders = glob.glob('/home/leon/DeepLearning/Project/Dataset/*')
# for folder in folders[0:1]:
#     imagenames_list = []
#     for f in glob.glob(folder + '/*.png'):
#         imagenames_list.append(f)
#
#     read_images = []
#     for image in imagenames_list[0:1]:
#         image_bgr = cv2.imread(image)
#         #print(image_bgr)
#         #rgb_image = np.array(Image.open(image).convert('RGB'))
#         rgb_image = Image.open(image).convert('RGB')
#         print(rgb_image.size)
#         image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
#         # image_lab[:, :, 0] = image_lab[:, :, 0]/50 - 1
#         # image_lab[:, :, 1] = image_lab[:, :, 1]/127
#         # image_lab[:, :, 2] = image_lab[:, :, 2]/127
#         #print(image_lab)
#         lab_image = rgb2lab(rgb_image)
#         rgb_image2 = lab2rgb(lab_image)
#         #print(lab_image)
#         #image_bgr = postprocess(image_lab)
#         rgb_image = cv2.cvtColor(np.array((rgb_image2 * 255.0).astype(np.uint8)), cv2.COLOR_RGB2BGR)
#         read_images.append(rgb_image)
# cv2.imshow("image", read_images[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------------------------------#
# x = np.array([[1,2],[3,4]])
# print(x.repeat(4, axis = 0))
# print(int(128 / np.power(2, 5)))
x = "201988"
print(x[4:])
