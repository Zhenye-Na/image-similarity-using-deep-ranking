"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable
from resnet import resnet_cifar

import argparse
parser = argparse.ArgumentParser()

# model_urls
parser.add_argument('--model_url', type=str, default="https://download.pytorch.org/models/resnet18-5c106cde.pth", help='model url for pretrained model')

# parse the arguments
args = parser.parse_args()


def resnet18(model_urls, pretrained=True):
    """Load pre-trained ResNet-18 model in Pytorch."""
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])

    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(model_urls, model_dir='../'))
        model = ConvNet(model)
    return model


class ConvNet(nn.Module):
    """Fine-tune pre-trained ResNet model."""

    def __init__(self, resnet):
        """Initialize Fine-tune ResNet model."""
        super(ConvNet, self).__init__()

        # Everything except the last linear layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # num_ftrs = resnet.fc.in_features
        # self.classifier = nn.Sequential(
        #     nn.Linear(num_ftrs, num_classes)
        # )
        self.maxpool3x3 = nn.MaxPool2d(kernel_size=3)
        self.maxpool7x7 = nn.MaxPool2d(kernel_size=7)
        self.dropout = nn.Dropout()

    def forward(self, x):
        """Forward pass of ResNet model."""
        out = self.features(x)
        # out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return out


def convnet():
    """ConvNet in multiscale network structure in Fig 3."""
    resnet = resnet18(model_urls)



    F.normalize(x, p=2, dim=1)


# TODO:
