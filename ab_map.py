import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob
from scipy.interpolate import interp2d

nbins = 32

folders = glob.glob('/home/leon/DeepLearning/Project/Dataset/*')
final_hist = np.zeros((nbins, nbins))
for folder in folders:
    imagenames_list = []
    for f in glob.glob(folder + '/*.png'):
        imagenames_list.append(f)

    read_images = []
    for image in imagenames_list:
        image_bgr = cv2.imread(image)
        image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        read_images.append(image_lab)

    print('Number of images: ' + str(len(read_images)))
    A = np.array([])
    B = np.array([])
    for single_image in read_images:
        a = single_image[:, :, 1]
        b = single_image[:, :, 2]
        a = a.reshape(a.shape[0] * a.shape[1])
        b = b.reshape(b.shape[0] * b.shape[1])
        A = np.append(A, a)
        B = np.append(B, b)

    hist, a_edges, b_edges, _ = plt.hist2d(A, B, bins = nbins, range = [[0, 255], [0, 255]], norm = LogNorm(), cmap = 'plasma')
    print(hist.shape)
    print('Number of cells: ' + str(np.count_nonzero(hist)))
    final_hist += hist

print(np.count_nonzero(final_hist))
x = np.arange(0, 255, 8)
y = np.arange(0, 255, 8)
h = plt.pcolormesh(x, y, final_hist, norm = LogNorm(), cmap = 'plasma')
plt.colorbar()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
