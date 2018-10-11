"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import sys

from sampler import TripletSampler


def preprocess(file="../tiny-imagenet-200/words.txt"):
    """
    Preprocess training images and labels.

    Args:
        file: txt file containing images directory and labels
    Returns:
        lines: list of lists which contains folder names and labels
    """
    with open(file, 'r') as fd:
        lines = [ line.strip().split("\t") for line in fd.readlines() ]

    # lines[0] is the directory name of images in classes included in lines[1]
    for line in lines:
        line[1] = line[1].split(", ")

    return lines



def TinyImageNetLoader(train_root, test_root, batch_size_train, batch_size_test):
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

    trainset = TinyImageNet(root=train_root, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, sampler=TripletSampler, num_workers=4)

    testset = TinyImageNet(root=test_root, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, sampler=TripletSampler, num_workers=4)

    return trainloader, testloader


def train():
    """
    Training process.

    Args:
        net: Network model
    """
    pass




# HACK:
