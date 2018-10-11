"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import os

# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset

class TinyImageNet(Dataset):
    """TinyImageNet."""

    def __init__(self, root="/projects/training/bauh/tiny-imagenet-200/", transform=None):
        """TinyImageNet Class Builder."""
        # stuff
        self.transform = transform
        self.train_root = root + "train/"
        self.test_root = root + "val/"

    def __getitem__(self, index):
        # stuff

        if self.transform is not None:
            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return count # of how many examples(images?) you have
