"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from Pokemon import *

def gen_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std

if __name__=='__main__':
    trainset = Pokemon(root="../data/preprocessed_data", transform=train_transform)
    mean, std = gen_mean_std(trainset)
    print(mean, std)
