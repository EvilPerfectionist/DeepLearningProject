# DeepLearningProject: Animation Colorization with Memory-Augmented Networks

## Overview

The project is to reproduce the Memory-Augmented Networks proposed by [Coloring With Limited Data: Few-shot Colorization via Memory Augmented Networks](https://arxiv.org/abs/1906.11888).

## Requirement

* python-opencv
* matplotlib

## Outline

* How to generate dataset

## How to generate dataset

The file `dataset_generator.py` can be used to extract image frames from videos. I use VideoCapture from OpenCV to load the video and calculate the color distribution difference between two neighboring image frames. A series of image frames can be called a shot. I keep shots in which the difference between all the neighboring image frames is within a threshold and whose length is within another threshold. Finally, I use `np.random.choice` to randomly save images from these shots to train/val/test folders.
