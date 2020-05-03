import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride = 2, padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride = 2, padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride = 2, padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.block_6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 2, dilation = 2),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.block_7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.block_8 = nn.Sequential(
            nn.Conv2d(512, 256, 4, stride = 2, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(True)
        )

        self.pred_layer = nn.Sequential(
            nn.Conv2d(256, 313, 4, stride = 2, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(True)
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
