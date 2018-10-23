"""
Image Similarity using Deep Ranking.

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import time
import argparse

import torch
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from numpy import linalg as LA
from torch.autograd import Variable
from utils import TinyImageNetLoader

from net import *

from sklearn.neighbors import KNeighborsClassifier


def calculate_accuracy(trainloader, testloader, is_gpu):
    """
    Calculate accuracy for TripletNet model.

        1. Form 2d array: Number of training images * size of embedding
        2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
        3. Perform subtraction between the two 2D arrays
        4, Take L2 norm of the 2d array (after subtraction)
        5. Get the 30 min values (argmin might do the trick)
        6. Repeat for the rest of the embeddings in the test set

    """
    net = TripletNet(resnet101())

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    if is_gpu:
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    print('==> Retrieve model parameters ...')
    checkpoint = torch.load("../checkpoint/checkpointcheckpoint.pth.tar")
    # start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    net.load_state_dict(checkpoint['state_dict'])

    net.eval()

    t1 = time.time()
    # dictionary of test images with class
    class_dict = get_classes()
    t2 = time.time()
    print("Get all test image classes, Done ... | Time elapsed {}s".format(t2 - t1))

    # list of traning images names, e.g., "../tiny-imagenet-200/train/n01629819/images/n01629819_238.JPEG"
    # update to get class names
    training_images = []
    for line in open("../triplets.txt"):
        line_array = line.split(",")
        if line_array[0] not in training_images:
            training_images.append(line_array[0])
    t3 = time.time()
    print("Get all training images, Done ... | Time elapsed {}s".format(t3 - t2))

    embedded_features_train = np.fromfile(
        "../embedded_features_train.txt", dtype=np.float32)

    neigh = KNeighborsClassifier(
        n_neighbors=30, weights='distance', algorithm='kd_tree', n_jobs=-1)
    neigh.fit(embedded_features_train.reshape(-1, 4096),
              np.array(training_images).reshape, 1)

    # TODO: 2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
    embedded_features_t = []
    with torch.no_grad():
        for test_id, test_data in enumerate(testloader):

            if test_id % 5 == 0:
                print("Now processing {}th test image".format(test_id))

            if is_gpu:
                test_data = test_data.cuda()
            test_data = Variable(test_data)

            embedded_test, _, _ = net(test_data, test_data, test_data)
            embedded_test_numpy = embedded_test.data.cpu().numpy()

            embedded_features_t.append(embedded_test_numpy)

        embedded_features_test = np.concatenate(embedded_features_t, axis=0)

        print(neigh.score(embedded_features_test, training_images))


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


def main():
    """Main pipleine for image similarity using deep ranking."""
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', type=str, default="",
                        help='train/val data root')
    parser.add_argument('--batch_size_train', type=int,
                        default=25, help='training set input batch size')
    parser.add_argument('--batch_size_test', type=int,
                        default=1, help='test set input batch size')

    parser.add_argument('--is_gpu', type=bool, default=True,
                        help='whether training using GPU')

    # parse the arguments
    args = parser.parse_args()

    # load triplet dataset
    trainloader, testloader = TinyImageNetLoader(
        args.dataroot, args.batch_size_train, args.batch_size_test)

    # calculate test accuracy
    calculate_accuracy(trainloader, testloader, args.is_gpu)


if __name__ == '__main__':
    main()
