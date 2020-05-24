# DeepLearningProject: Animation Colorization with Memory-Augmented Networks

## Overview

The project is to reproduce the Memory-Augmented Networks proposed by [Coloring With Limited Data: Few-shot Colorization via Memory Augmented Networks](https://arxiv.org/abs/1906.11888).

## Requirement

* python-opencv
* matplotlib

## Outline

* How to generate dataset

## How to generate dataset

The file `dataset_generator.py` can be used to extract image frames from videos. I use VideoCapture from OpenCV to load the video and calculate the color distribution difference between two neighboring image frames. A series of image frames can be called a shot. I keep shots in which the difference between all the neighboring image frames is within a threshold and whose length is within another threshold. Finally, I use `np.random.choice` to randomly save images from these shots to train/val/test folders. The default proportion of assigning images to train/val/test folders is 0.7:0.2:0.1. You can change this proportion as you wish in the file.

The file takes five parameters:
* video_path: The path of the videos
* video_names: The video names
* save_path: Where you want to save the extracted images. The file will create a base folder in this path and create train/val/test folders inside the base folder.
* col_diff_thres: threshold of the color distribution difference.
* shots_len_thres: threshold of the length of the shots. You can customize the last two parameters to control the number of images you want to obtain.
