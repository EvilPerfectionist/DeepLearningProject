# DeepLearningProject: Animation Colorization with Memory-Augmented Networks

## Overview

The project is to reproduce the Memory-Augmented Networks proposed by [Coloring With Limited Data: Few-shot Colorization via Memory Augmented Networks](https://arxiv.org/abs/1906.11888).

## Requirement

* python-opencv
* matplotlib
* pytorch 1.1.0

## Outline

* How to generate dataset
* How to use generated dataset to train your model

## How to generate dataset

The file `dataset_generator.py` can be used to extract image frames from videos. I use VideoCapture from OpenCV to load the video and calculate the color distribution difference between two neighboring image frames. A series of image frames can be called a shot. I keep shots in which the difference between all the neighboring image frames is within a threshold and whose length is within another threshold. Finally, I use `np.random.choice` to randomly save images from these shots to train/val/test folders. The default proportion of assigning images to train/val/test folders is 0.7:0.2:0.1. You can change this proportion as you wish in the file. The images will be named as `raw_image_videoname_number.png`.

The file takes five parameters:
* video_path: The path of the videos
* video_names: The video names
* save_path: Where you want to save the extracted images. The file will create a base folder in this path and create train/val/test folders inside the base folder.
* col_diff_thres: threshold of the color distribution difference.
* shots_len_thres: threshold of the length of the shots. You can customize the last two parameters to control the number of images you want to obtain.

***Note:*** There is a conflict between ROS and OpenCV so that I add the following code to my file. If you don't use ROS, please remove them.
```
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
```

## How to use generated dataset to train your model

You can use `train.py` file to train your own model.

### Initialize Dataset and DataLoader

The train and validation datasets are initialized by loading the images from the train/val folders mentioned in the previous section and implementing some processing to these images.

The processing details can be found in the file `dataset.py`. If we see this processing as a black box, it inputs the images from the train/val folders and outputs python dicts which contain the 5 elements: L channel of the images, ab channels of the images, color features of the images, greyscale images and image identities or numbers.

Assume we have an image whose size is 256x256x3. Then the size of its ab channels is 256x256x2 and the number of total pixels is 256x256. The color feature of each pixel can be represented by a pair of ab values [a, b]. Therefore, the color features of the image can be represented by 256x256 pairs of [a, b] values. According to [Colorful Image Colorization](https://arxiv.org/abs/1603.08511), there are 313 possibilities of [a, b] pairs. Then we can map these 256x256 pairs of [a, b] values to a 313 possibility vector. This 313 possibility vector will be treated as our final color features of the image. 
