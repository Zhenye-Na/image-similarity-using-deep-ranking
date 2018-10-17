"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import sys
import numpy as np

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

from torch.autograd import Variable

from imageloader import TripletImageLoader


def TinyImageNetLoader(trainroot, test_root, batch_size_train, batch_size_test):
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loading Tiny ImageNet dataset
    print("==> Preparing Tiny ImageNet dataset ...")

    trainset = TripletImageLoader(base_path=trainroot, triplets_filename="../triplets.txt", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, num_workers=32)

    testset = TripletImageLoader(root=test_root, transform=transform_test, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, num_workers=32)

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

    # how many batches to wait before logging training status
    log_interval = 20
    net.train()
    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        for batch_idx, (data1, data2, data3) in enumerate(trainloader):

            if is_gpu:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

            # compute output and loss
            embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
            loss = criterion(embedded_a, embedded_p, embedded_n)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        # Calculate training/test set accuracy of the existing model
        # train_accuracy = calculate_accuracy(net, trainloader, is_gpu)
        # test_accuracy = calculate_accuracy(net, testloader, is_gpu)

        print("Training Epoch: {0} | Loss: {1}".format(epoch+1, running_loss))

        # save model in every epoch
        print('==> Saving model ...')
        state = {
            'net': net.module if is_gpu else net,
            'epoch': epoch,
        }
        if not os.path.isdir('../checkpoint'):
            os.mkdir('../checkpoint')
        torch.save(state, '../checkpoint/ckpt.t7')

    print('==> Finished Training ...')


def calculate_accuracy(net, trainloader, testloader is_gpu):
    """
    Calculate accuracy for TripletNet model.

    Try using numpy to do this efficiently (otherwise it'll be too slow).

    For example:
    1. Form 2d array: Number of training images * size of embedding
    2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
    3. Perform subtraction between the two 2D arrays
    4, Take L2 norm of the 2d array (after subtraction)
    5. Get the 30 min values (argmin might do the trick)
    6. Repeat for the rest of the embeddings in the test set

    I wouldn't think this would be too slow (10^4 comparisons), especially if you multi-threaded it.
    P.S. - Kd-trees sounds like a good idea as well.

    """
    embedded_features =

    for data1, data2, data3 in trainloader:

        if is_gpu:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

        # wrap in torch.autograd.Variable
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        embedded_a, _, _ = net(data1, data2, data3)





    # TODO: 1. Form 2d array: Number of training images * size of embedding
    matrix = np.zeros(900000, 4096))


    # TODO: 2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)



    # TODO: 3. Perform subtraction between the two 2D arrays



    # TODO: 4, Take L2 norm of the 2d array (after subtraction)


    # TODO: 5. Get the 30 min values (argmin might do the trick)


    # TODO: 6. Repeat for the rest of the embeddings in the test set




def calculate_distance(i1, i2):
    """
    Calculate euclidean distance of the ranked results from the query image.

    Args:
        i1: query image
        i2: ranked result
    """
    return np.sum((i1 - i2) ** 2)


# def preprocess(file="../tiny-imagenet-200/words.txt"):
#     """
#     Preprocess reading training images and labels.
#
#     Args:
#         file: txt file containing images directory and labels
#     Returns:
#         lines: list of lists which contains folder names and labels
#     """
#     with open(file, 'r') as fd:
#         lines = [ line.strip().split("\t") for line in fd.readlines() ]
#
#     # lines[0] is the directory name of images in classes included in lines[1]
#     for line in lines:
#         line[1] = line[1].split(", ")
#
#     return lines
