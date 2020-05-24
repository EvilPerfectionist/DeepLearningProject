import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from matplotlib import pyplot as plt
import os
import argparse

def calculate_diff(frame1, frame2):
    diff = 0.0
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr1 = cv2.calcHist([frame1], [i], None, [256], [0,256])
        histr2 = cv2.calcHist([frame2], [i], None, [256], [0,256])
        diff += np.square(np.subtract(histr1.flatten(), histr2.flatten())).mean()
    #print(diff / 3)
    return diff

def save_images(video_name, args):
    cap = cv2.VideoCapture(args.video_path + video_name + '.mp4')

    count_train = 0
    count_val = 0
    count_test = 0

    frame_list = []
    diff_list = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if np.shape(frame) == ():
            break
        #resize images to (256, 256)
        frame = cv2.resize(frame, (256, 256))
        #store images to a list
        frame_list.append(frame)

        if len(frame_list) > 2:
            #calculate the color distribution histogram difference between two images
            diff = calculate_diff(frame_list[-1], frame_list[-2])
            diff_list.append(diff)
            if sum(i > args.col_diff_thres for i in diff_list) == 2:
                if len(frame_list) >= args.shots_len_thres:
                    print(len(frame_list))
                    choice = np.random.choice(3, 3, p=[0.7, 0.1, 0.2])[0]
                    if choice == 0:
                        count_train += 1
                        cv2.imwrite(args.save_path + '/train/raw_image_' + video_name + '_' + str(count_train) + '.png', frame_list[0])
                        count_train += 1
                        cv2.imwrite(args.save_path + '/train/raw_image_' + video_name + '_' + str(count_train) + '.png', frame_list[-1])
                        count_train += 1
                        cv2.imwrite(args.save_path + '/train/raw_image_' + video_name + '_' + str(count_train) + '.png', frame_list[len(frame_list) // 2])
                    elif choice == 1:
                        count_test += 1
                        cv2.imwrite(args.save_path + '/test/raw_image_' + video_name + '_' + str(count_test) + '.png', frame_list[0])
                        count_test += 1
                        cv2.imwrite(args.save_path + '/test/raw_image_' + video_name + '_' + str(count_test) + '.png', frame_list[-1])
                        count_test += 1
                        cv2.imwrite(args.save_path + '/test/raw_image_' + video_name + '_' + str(count_test) + '.png', frame_list[len(frame_list) // 2])
                    elif choice == 2:
                        count_val += 1
                        cv2.imwrite(args.save_path + '/val/raw_image_' + video_name + '_' + str(count_val) + '.png', frame_list[0])
                        count_val += 1
                        cv2.imwrite(args.save_path + '/val/raw_image_' + video_name + '_' + str(count_val) + '.png', frame_list[-1])
                        count_val += 1
                        cv2.imwrite(args.save_path + '/val/raw_image_' + video_name + '_' + str(count_val) + '.png', frame_list[len(frame_list) // 2])
                frame_list = []
                diff_list = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def create_folders(args):
    base_folder = args.save_path
    train_folder = base_folder + '/train'
    val_folder = base_folder + '/val'
    test_folder = base_folder + '/test'
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

def define_arguments():
    parser = argparse.ArgumentParser(
        description='Animation Colorization with Memory-Augmented Networks and a Few Shots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video_path", type = str, default = '/home/leon/DeepLearning/Project/', help = 'Path to load the video')
    parser.add_argument("--video_names", type = str, default = 'Romantic')
    parser.add_argument("--save_path", type = str, default = '/home/leon/DeepLearning/Project/Dataset/', help = 'Path to save the images')
    parser.add_argument("--col_diff_thres", type = float, default = 7000.0, help = 'Threshold for splitting the videos to different shots')
    parser.add_argument("--shots_len_thres", type = int, default = 6, help = 'Filter the shots whose length are larger than the threshold ')
    return parser.parse_args()

if __name__ == '__main__':
    args = define_arguments()
    # create train val test folders
    create_folders(args)
    video_names = [args.video_names]
    for video_name in video_names:
        save_images(video_name, args)
