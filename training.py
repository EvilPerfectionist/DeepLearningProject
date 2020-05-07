import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_imagenet import TrainImageFolder
from model import Color_model
from training_layers import PriorBoostLayer, NNEncLayer, ClassRebalanceMultLayer, NonGrayMaskLayer

original_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # Image preprocessing, normalization for the pretrained resnet
    train_set = TrainImageFolder(args.image_dir, original_transform)
    # Build data loader
    data_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Build the models
    model = Color_model()
    model.to(device)
    #model.load_state_dict(torch.load('../model/models/model-171-216.ckpt'))
    encode_layer = NNEncLayer()
    boost_layer = PriorBoostLayer()
    nongray_mask = NonGrayMaskLayer()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    # Train the models
    total_step = len(data_loader)
    print(len(data_loader))
    for epoch in range(args.num_epochs):
        for i, (images, img_ab) in enumerate(data_loader):
            # Set mini-batch dataset
            # print(images.shape)
            # print(img_ab.shape)
            images = images.unsqueeze(1).float().to(device)
            img_ab = img_ab.float()
            # print(images.shape)
            # print(img_ab.shape)
            encode, max_encode = encode_layer.forward(img_ab)
            # print(encode.shape)
            targets = torch.Tensor(max_encode).long().to(device)
            #print('set_tar',set(targets[0].cpu().data.numpy().flatten()))
            boost = torch.Tensor(boost_layer.forward(encode)).float().to(device)
            mask = torch.Tensor(nongray_mask.forward(img_ab)).float().to(device)
            boost_nongray = boost * mask
            outputs = model(images)#.log()
            output = outputs[0].to('cpu').data.numpy()
            # print(outputs.shape)
            # print(output.shape)
            out_max = np.argmax(output,axis=0)
            #print(out_max.shape)

            #print('set',set(out_max.flatten()))
            loss = (criterion(outputs,targets)*(boost_nongray.squeeze(1))).mean()

            model.zero_grad()

            loss.backward()
            optimizer.step()
            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch, args.num_epochs, i, total_step, loss.item()))

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0 and epoch + 1 == args.num_epochs:
                print('Finished Training')
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = '../model/', help = 'path for saving trained models')
    parser.add_argument('--crop_size', type = int, default = 224, help = 'size for randomly cropping images')
    parser.add_argument('--image_dir', type = str, default = '/home/leon/DeepLearning/Project/Dataset', help = 'directory for resized images')
    parser.add_argument('--log_step', type = int, default = 1, help = 'step size for prining log info')
    parser.add_argument('--save_step', type = int, default = 20, help = 'step size for saving trained models')

    # Model parameters
    parser.add_argument('--num_epochs', type = int, default = 2)
    parser.add_argument('--batch_size', type = int, default = 5)
    parser.add_argument('--num_workers', type = int, default = 2)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    args = parser.parse_args()
    #print(args)
    main(args)
