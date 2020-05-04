import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from matplotlib import pyplot as plt
import os

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

def calculate_diff(frame1, frame2):
    diff = 0.0
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr1 = cv2.calcHist([frame1], [i], None, [256], [0,256])
        histr2 = cv2.calcHist([frame2], [i], None, [256], [0,256])
        diff += np.square(np.subtract(histr1.flatten(), histr2.flatten())).mean()
    print(diff / 3)
    return diff

video_name = 'TheNightBeforChristmas'
os.mkdir('/home/leon/DeepLearning/Project/Dataset/' + video_name)
cap = cv2.VideoCapture('/home/leon/DeepLearning/Project/' + video_name + '.avi')

cropped_frame_list = []
diff_list = []
count = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    dsize = (256, 256)
    crop = cv2.resize(frame, dsize)

    cropped_frame_list.append(crop)

    if len(cropped_frame_list) > 2:
        diff = calculate_diff(cropped_frame_list[-1], cropped_frame_list[-2])
        diff_list.append(diff)
        if sum(i > 5000.0 for i in diff_list) == 2:
            if len(cropped_frame_list) < 6:
                cropped_frame_list = []
                diff_list = []
            else:
                count += 1
                cv2.imwrite('/home/leon/DeepLearning/Project/Dataset/' + video_name + '/raw_image_' + str(count) + '.png', cropped_frame_list[0])
                cropped_frame_list = []
                diff_list = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
