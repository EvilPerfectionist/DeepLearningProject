import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob

folders = glob.glob('/home/leon/DeepLearning/Project/Dataset/*')
imagenames_list = []
for folder in folders:
    for f in glob.glob(folder + '/*.png'):
        print(f)
        imagenames_list.append(f)

read_images = []
for image in imagenames_list:
    image_bgr = cv2.imread(image)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    read_images.append(image_lab)

A = np.array([])
B = np.array([])
for single_image in read_images:
    a = single_image[:, :, 1] - 128
    b = single_image[:, :, 2] - 128
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    A = np.append(A, a)
    B = np.append(B, b)
nbins = 10
plt.hist2d(A, B, bins = nbins, norm = LogNorm())
plt.xlabel('A')
plt.ylabel('B')
plt.colorbar()
plt.xlim([-110, 110])
plt.ylim([-110 ,110])
plt.show()

cv2.imshow("Result BGR", read_images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
