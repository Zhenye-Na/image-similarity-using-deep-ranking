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

    trainset = TripletImageLoader(base_path=root, triplets_filename="../triplets.txt", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, num_workers=32)

    testset = TripletImageLoader(base_path=root, triplets_filename="", transform=transform_test, train=False)
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
    best_acc = 0

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

            if batch_idx % 30 == 0:
                print("mini Batch Loss: {}".format(loss.data[0]))

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        # Calculate training/test set accuracy of the existing model
        # train_accuracy = calculate_accuracy(net, trainloader, is_gpu)
        # test_accuracy = calculate_accuracy(net, testloader, is_gpu)

        print("Training Epoch: {0} | Loss: {1}".format(epoch+1, running_loss))

        # remember best acc and save checkpoint
        # is_best = test_accuracy > best_acc
        # best_acc = max(test_accuracy, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            # 'best_prec1': best_acc,
        }, False)

    print('==> Finished Training ...')


def calculate_accuracy(net, trainloader, testloader, is_gpu):
    """
    Calculate accuracy for TripletNet model.

        1. Form 2d array: Number of training images * size of embedding
        2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
        3. Perform subtraction between the two 2D arrays
        4, Take L2 norm of the 2d array (after subtraction)
        5. Get the 30 min values (argmin might do the trick)
        6. Repeat for the rest of the embeddings in the test set

    """
    print('==> Retrieve model parameters ...')
    checkpoint = torch.load("../checkpoint/checkpoint.pth.tar")
    # args.start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    net.load_state_dict(checkpoint['state_dict'])

    # dictionary of test images with class
    # val_1788.JPEG ==> n04532670
    # val_8463.JPEG ==> n02917067
    class_dict = get_classes()

    # list of traning images names, e.g., "../tiny-imagenet-200/train/n01629819/images/n01629819_238.JPEG"
    training_images = []
    for line in open(triplets_filename):
        line_array = line.split(",")
        training_images.append(line_array[0])

    # get embedded features of training
    embedded_features = []
    for data1, data2, data3 in trainloader:

        if is_gpu:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

        # wrap in torch.autograd.Variable
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        embedded_a, _, _ = net(data1, data2, data3)
        embedded_a_numpy = embedded_a.data.cpu().numpy()

        embedded_features.append(embedded_a_numpy)

    # TODO: 1. Form 2d array: Number of training images * size of embedding
    embedded_features_train = np.concatenate(embedded_features, axis=0)

    # TODO: 2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
    for test_id, test_data in enumerate(testloader):

        if is_gpu:
            test_data = test_data.cuda()
        test_data = Variable(test_data)

        embedded_test, _, _ = net(test_data, test_data, test_data)
        embedded_test_numpy = embedded_test_numpy.data.cpu().numpy()

        embedded_features_test = np.tile(embedded_test_numpy, (embedded_features.shape[0], 1))

        # TODO: 3. Perform subtraction between the two 2D arrays
        embedding_diff = embedded_features_train - embedded_features_test

        # TODO: 4, Take L2 norm of the 2d array (after subtraction)
        embedding_norm = LA.norm(embedding_diff, axis=0)

        # TODO: 5. Get the 30 min values (argmin might do the trick)
        min_index = embedding_norm.argsort()[:30]

        # TODO: 6. Repeat for the rest of the embeddings in the test set
        accuracies = []

        # get test image class
        test_image_name = "val_" + str(test_id) + ".JPEG"
        test_image_class = class_dict[test_image_name]

        # for each image results in min distance
        for idx in min_index:
            correct = 0

            # get result image class
            top_result_image_name = training_images[idx]
            top_result_image_name_class = top_result_image_name.split("/")[3]

            if test_image_class == top_result_image_name_class:
                correct += 1

        acc = correct / 30
        accuracies.append(acc)

    avg_acc = sum(accuracies) / len(accuracies)
    print("Test accuracy {}%".format(avg_acc))


def get_classes(filename="../tiny-imagenet-200/val/val_annotations.txt"):
    """
    Get corresponding class name for each val image.

    Args:
        filename: txt file which contains image name and corresponding class name

    Returns:
        class_dict: A dictionary which maps from image name to class name
    """
    class_dict = {}
    for line in open(filename):
        line_array = line.rstrip("\n").split("\t")
        class_dict[line_array[0]] = line_array[1]

    return class_dict



# def calculate_accuracy(net, trainloader, testloader, is_gpu):
#     """
#     Calculate accuracy for TripletNet model.
#
#         1. Form 2d array: Number of training images * size of embedding
#         2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
#         3. Perform subtraction between the two 2D arrays
#         4, Take L2 norm of the 2d array (after subtraction)
#         5. Get the 30 min values (argmin might do the trick)
#         6. Repeat for the rest of the embeddings in the test set
#
#     """
#     embedded_features = []
#
#     for data1, data2, data3 in trainloader:
#
#         if is_gpu:
#             data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
#
#         # wrap in torch.autograd.Variable
#         data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)
#
#         # compute output
#         embedded_a, _, _ = net(data1, data2, data3)
#
#         embedded_a_numpy = embedded_a.data.cpu().numpy()
#         embedded_features.append(embedded_a_numpy)
#
#     # TODO: 1. Form 2d array: Number of training images * size of embedding
#     embedded_features_train = np.concatenate(embedded_features, axis=0)
#
#     # TODO: 2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
#     for test_data in testloader:
#
#         if is_gpu:
#             test_data = test_data.cuda()
#         test_data = Variable(test_data)
#
#         embedded_test, _, _ = net(test_data, test_data, test_data)
#         embedded_test_numpy = embedded_test_numpy.data.cpu().numpy()
#
#         embedded_features_test = np.tile(embedded_test_numpy, (embedded_features.shape[0], 1))
#
#     # TODO: 3. Perform subtraction between the two 2D arrays
#     embedding_diff = embedded_features_train - embedded_features_test
#
#     # TODO: 4, Take L2 norm of the 2d array (after subtraction)
#     embedding_norm = LA.norm(embedding_diff, axis=0)
#
#     # TODO: 5. Get the 30 min values (argmin might do the trick)
#     min_index = embedding_norm.argsort()[:30]
#
#     # TODO: 6. Repeat for the rest of the embeddings in the test set


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
