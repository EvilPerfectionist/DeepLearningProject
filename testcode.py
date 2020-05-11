import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob
from scipy.interpolate import interp2d

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

folders = glob.glob('/home/leon/DeepLearning/Project/Dataset/*')
for folder in folders[0:1]:
    imagenames_list = []
    for f in glob.glob(folder + '/*.png'):
        imagenames_list.append(f)

    read_images = []
    for image in imagenames_list:
        image_bgr = cv2.imread(image)
        image_lab = preprocess(image_bgr)
        image_bgr = postprocess(image_lab)
        read_images.append(image_bgr)
cv2.imshow("image", read_images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
