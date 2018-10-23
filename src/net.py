"""
Image Similarity using Deep Ranking.

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import os
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18(model_urls, pretrained=True):
    """
    Construct a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(
            model_urls['resnet18'], model_dir='../resnet18'))
    return EmbeddingNet(model)


def resnet101(pretrained=False, **kwargs):
    """
    Construct a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.BasicBlock, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(
            model_urls['resnet101'], model_dir='../resnet101'))
    return EmbeddingNet(model)


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

        return embedded_a, embedded_p, embedded_n


class EmbeddingNet(nn.Module):
    """EmbeddingNet using ResNet-101."""

    def __init__(self, resnet):
        """Initialize EmbeddingNet model."""
        super(EmbeddingNet, self).__init__()

        # Everything except the last linear layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 4096)

    def forward(self, x):
        """Forward pass of EmbeddingNet."""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out
