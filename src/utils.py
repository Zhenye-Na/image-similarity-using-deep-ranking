"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import os
import sys
import shutil
import numpy as np

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

from numpy import linalg as LA
from torch.autograd import Variable

from imageloader import TripletImageLoader


def TinyImageNetLoader(root, batch_size_train, batch_size_test):
    """
    Tiny ImageNet Loader.

    Args:
        train_root:
        test_root:
        batch_size_train:
        batch_size_test:

    Return:
        trainloader:
        testloader:
    """
    # Normalize training set together with augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Loading Tiny ImageNet dataset
    print("==> Preparing Tiny ImageNet dataset ...")

    trainset = TripletImageLoader(
        base_path=root, triplets_filename="../triplets.txt", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, num_workers=32)

    testset = TripletImageLoader(
        base_path=root, triplets_filename="", transform=transform_test, train=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, num_workers=32)

    return trainloader, testloader


def train(net, criterion, optimizer, scheduler, trainloader,
          testloader, start_epoch, epochs, is_gpu):
    """
    Training process.
    Args:
        net: Triplet Net
        criterion: TripletMarginLoss
        optimizer: SGD with momentum optimizer
        scheduler: scheduler
        trainloader: training set loader
        testloader: test set loader
        start_epoch: checkpoint saved epoch
        epochs: training epochs
        is_gpu: whether use GPU
    """
    print("==> Start training ...")

    net.train()
    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        for batch_idx, (data1, data2, data3) in enumerate(trainloader):

            if is_gpu:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(
                data1), Variable(data2), Variable(data3)

            # compute output and loss
            embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
            loss = criterion(embedded_a, embedded_p, embedded_n)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

            if batch_idx % 30 == 0:
                print("mini Batch Loss: {}".format(loss.data[0]))

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        print("Training Epoch: {0} | Loss: {1}".format(epoch+1, running_loss))

        # remember best acc and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
        }, False)

    print('==> Finished Training ...')


def calculate_distance(i1, i2):
    """
    Calculate euclidean distance of the ranked results from the query image.

    Args:
        i1: query image
        i2: ranked result
    """
    return np.sum((i1 - i2) ** 2)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint."""
    directory = "../checkpoint"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')
