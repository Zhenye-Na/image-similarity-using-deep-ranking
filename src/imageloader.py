"""
Image Similarity using Deep Ranking.

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

from __future__ import print_function
from PIL import Image

import os
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


def image_loader(path):
    """Image Loader helper function."""
    return Image.open(path.rstrip("\n")).convert('RGB')


class TripletImageLoader(Dataset):
    """Image Loader for Tiny ImageNet."""

    def __init__(self, base_path, triplets_filename, transform=None,
                 train=True, loader=image_loader):
        """
        Image Loader Builder.

        Args:
            base_path: path to triplets.txt
            filenames_filename: text file with each line containing the path to an image e.g., `images/class1/sample.JPEG`
            triplets_filename: A text file with each line containing three images
            transform: torchvision.transforms
            loader: loader for each image
        """
        self.base_path = base_path
        self.transform = transform
        self.loader = loader

        self.train_flag = train

        # load training data
        if self.train_flag:
            triplets = []
            for line in open(triplets_filename):
                line_array = line.split(",")
                triplets.append((line_array[0], line_array[1], line_array[2]))
            self.triplets = triplets

        # load test data
        else:
            singletons = []
            test_images = os.listdir(os.path.join(
                "../tiny-imagenet-200", "val", "images"))
            for test_image in test_images:
                loaded_image = self.loader(os.path.join(
                    "../tiny-imagenet-200", "val", "images", test_image))
                singletons.append(loaded_image)
            self.singletons = singletons

    def __getitem__(self, index):
        """Get triplets in dataset."""
        # get trainig triplets
        if self.train_flag:
            path1, path2, path3 = self.triplets[index]
            a = self.loader(os.path.join(self.base_path, path1))
            p = self.loader(os.path.join(self.base_path, path2))
            n = self.loader(os.path.join(self.base_path, path3))
            if self.transform is not None:
                a = self.transform(a)
                p = self.transform(p)
                n = self.transform(n)
            return a, p, n

        # get test image
        else:
            img = self.singletons[index]
            if self.transform is not None:
                img = self.transform(img)
            return img

    def __len__(self):
        """Get the length of dataset."""
        if self.train_flag:
            return len(self.triplets)
        else:
            return len(self.singletons)
