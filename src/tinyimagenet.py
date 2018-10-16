"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""


from __future__ import print_function
from PIL import Image
from skimage import io

import os
import random
import numpy as np

import torch.utils.data

# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset

from utils import preprocess


def gen_idx(idx, mode):
    """Generate random index for negative/positive images."""
    if mode == "class":
        random_class_idx = random.randint(0, 200)
        while idx == random_class_idx:
            random_class_idx = random.randint(0, 200)
        return random_class_idx

    elif mode == "image":
        random_img_idx = random.randint(0, 200)
        while idx == random_img_idx:
            random_img_idx = random.randint(0, 200)
        return random_img_idx

    else:
        raise ValueError('We do not support this mode right now!')



class TinyImageNet(Dataset):
    """TinyImageNet."""

    def __init__(self, root, transform=None):
        """TinyImageNet Class Builder."""
        self.transform = transform
        self.root = root

        self.train_images = []
        self.test_images = []

        # preprocess to get folder list
        # [[n00001740], [entity]]
        # [[n00001930], [physical entity]]
        # self.info = preprocess(file=os.path.join(root, "words.txt"))
        self.info = os.listdir(os.path.join(root, "train"))

        # training list: list of lists of images names
        self.train_dict = {}
        for info in self.info:
            if info != ".DS_Store":
                self.train_dict[info] = os.listdir(os.path.join(root, "train", info, "images"))
            # self.train_list.append(os.listdir(os.path.join(root, "train", info[0], "images")))

        # self.train_list = os.listdir(os.path.join(self.train_root, "iamges"))
        self.test_list = os.listdir(os.path.join(root, "val", "images"))

        # read training images
        for folder_name, files in self.train_dict.items():
            training_images = []
            for file in files:
                imgdir = os.path.join(root, "train", folder_name, "images", file)
                training_images.append(io.imread(imgdir))
            # training_images = np.array(training_images)
            self.train_images.append(training_images)


        # read test images
        for file in self.test_list:
            path = os.path.join(root, "val", "images")
            imgs = os.listdir(os.path.join(root, "val", "images"))
            for img in imgs:
                self.test_images.append(io.imread(os.path.join(path, img)))

        # self.test_images = np.array(self.test_images)


    def __getitem__(self, index):
        """Get triplet in dataset."""
        # TODO: 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).

        # query images
        class_idx = index / 500
        img_idx = index % 500
        q_img = self.train_images[class_idx][img_idx]

        # positive images
        positive_class_idx = class_idx
        p_img_idx = gen_idx(img_idx, "image")
        p_img = self.train_images[positive_class_idx][p_img_idx]

        # negative images
        negative_class_idx = gen_idx(class_idx, "class")
        negative_img_idx = gen_idx(img_idx, "image")
        n_img = self.train_images[negative_class_idx][negative_img_idx]

        # Return a PIL Image for later agmentation
        q_img = Image.fromarray(q_img).convert('RGB')
        p_img = Image.fromarray(p_img).convert('RGB')
        n_img = Image.fromarray(n_img).convert('RGB')

        # TODO: 2. Preprocess the data (e.g. torchvision.Transform).
        if self.transform is not None:
            q_img = self.transform(q_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)


        # TODO: 3. Return a data pair (e.g. image and label).
        return q_img, p_img, n_img


    def __len__(self):
        """Get length of dataset."""
        count = len(self.train_images) * len(self.train_images[0])
        return count # of how many examples(images?) you have







# ha
