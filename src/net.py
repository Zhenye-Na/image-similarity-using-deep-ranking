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
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable


def resnet18(model_urls, pretrained=True):
    """Load pre-trained ResNet-18 model in Pytorch."""
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])

    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(model_urls, model_dir='../resnet_model'))
        model = ConvNet(model)
    return model


def resnet101(pretrained=True):
    """Load pre-trained ResNet-101 model in Pytorch."""
    resnet101 = torchvision.models.resnet101(pretrained=pretrained)

    if pretrained:
        resnet101 = ConvNet(resnet101)

    return resnet101


class TripletNet(nn.Module):
    """Triplet Network."""
    def __init__(self, embeddingnet):
        """Triplet Network Builder."""
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, a, p, n):
        """Forward pass."""
        # anchor
        embedded_a = self.embeddingnet(a)

        # positive examples
        embedded_p = self.embeddingnet(p)

        # negative examples
        embedded_n = self.embeddingnet(n)

        # dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        # dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        # dist_a, dist_b,
        # return embedded_x, embedded_y, embedded_z
        return embedded_a, embedded_p, embedded_n


class EmbeddingNet(nn.Module):
    """EmbeddingNet."""

    def __init__(self, convnet):
        """Initialize Network model in Deep Ranking."""
        super(EmbeddingNet, self).__init__()

        self.convnet = convnet

        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, stride=16, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=4)
        )

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, stride=16, padding=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=4)

        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, stride=32, padding=1),
            nn.MaxPool2d(kernel_size=7, stride=2)
        )

        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, stride=32, padding=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=7, stride=2)

        self.embedding = nn.Linear(in_features=7170, out_features=4096)


    def forward(self, x):
        """Forward pass."""
        out1 = self.convnet.forward(x)
        norm1 = F.normalize(out1, p=2, dim=1)

        out2 = self.net1(x)
        out3 = self.net2(x)

        cat2 = torch.cat((out2, out3), 1)
        norm2 = F.normalize(cat2, p=2, dim=1)

        cat1 = torch.cat((norm1, norm2), 1)
        embedding = self.embedding(cat1)

        out = F.normalize(embedding, p=2, dim=1)

        return out


class ConvNet(nn.Module):
    """ConvNet using ResNet model."""

    def __init__(self, resnet):
        """Initialize ResNet model."""
        super(ConvNet, self).__init__()

        # Everything except the last linear layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 4096)

    def forward(self, x):
        """Forward pass of ResNet model."""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out
