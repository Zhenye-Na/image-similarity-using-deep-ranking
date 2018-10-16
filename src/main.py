"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import os
import torch
import torchvision
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import argparse

from utils import *
from net import *


parser = argparse.ArgumentParser()

# directory
parser.add_argument('--dataroot', type=str, default="../data", help='path to dataset')
parser.add_argument('--ckptroot', type=str, default="../checkpoint/ckpt.t7", help='path to checkpoint')
parser.add_argument('--trainroot', type=str, default="../tiny-imagenet-200", help='path to dataset')

# hyperparameters settings
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor')
parser.add_argument('--nesterov', type=bool, default=True, help='enables Nesterov momentum')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 penalty)')
parser.add_argument('--epochs', type=int, default=35, help='number of epochs to train')
parser.add_argument('--batch_size_train', type=int, default=129, help='training set input batch size')
parser.add_argument('--batch_size_test', type=int, default=129, help='test set input batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')

# loss function settings
parser.add_argument('--g', type=float, default=1.0, help='gap parameter')
parser.add_argument('--p', type=int, default=2, help='norm degree for pairwise distance - Euclidean Distance')

# training settings
parser.add_argument('--resume', type=bool, default=False, help='whether re-training from ckpt')
parser.add_argument('--is_gpu', type=bool, default=False, help='whether training using GPU')

# model_urls
parser.add_argument('--model_url', type=str, default="https://download.pytorch.org/models/resnet18-5c106cde.pth", help='model url of resnet-18')

# parse the arguments
args = parser.parse_args()


def main():
    """Main pipeline of Image Similarity using Deep Ranking."""

    # resume training from the last time
    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint ...')
        assert os.path.isdir(
            '../checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.ckptroot)
        net = checkpoint['net']
        args.start_epoch = checkpoint['epoch']
    else:
        # start over
        print('==> Building new TripletNet model ...')
        net = TripletNet(EmbeddingNet(resnet101()))

    print("==> Initialize CUDA support for TripletNet model ...")

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    if args.is_gpu:
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    # Loss function, optimizer and scheduler
    criterion = nn.TripletMarginLoss(margin=args.g, p=args.p)
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=10,
                                                           verbose=True)

    trainloader, testloader = TinyImageNetLoader(args.trainroot, args.trainroot, args.batch_size_train, args.batch_size_test)

    train(net, criterion, optimizer, scheduler, trainloader, testloader, args.start_epoch, args.epochs, args.is_gpu)



if __name__ == '__main__':
    main()
