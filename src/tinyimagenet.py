"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""


from __future__ import print_function
from PIL import Image
from skimage import io

import os
import numpy as np

import torch.utils.data

# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset

class TinyImageNet(Dataset):
    """TinyImageNet."""

    def __init__(self, root, transform=None):
        """TinyImageNet Class Builder."""
        self.transform = transform
        self.train_root = root + "train/"
        self.test_root = root + "val/"

        self.train_images = []
        self.test_images = []

        self.train_list = os.listdir(self.train_root)
        self.test_list = os.listdir(self.test_root)

        # preprocess
        self.info = preprocess(file=os.path.join(root, "words.txt"))


    def __getitem__(self, index):
        """Get item and label in dataset."""
        # TODO: 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).

        for file in self.train_list:
            imgdir = os.path.join(self.root, file)
            self.train_data.append(io.imread(imgdir))

        # TODO: 2. Preprocess the data (e.g. torchvision.Transform).
        if self.transform is not None:
            img = self.transform(img)


        # TODO: 3. Return a data pair (e.g. image and label).


        return (img, label)




    def __len__(self):
        """Get length of dataset."""
        return count # of how many examples(images?) you have







# ha
