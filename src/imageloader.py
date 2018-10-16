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
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


def image_loader(path):
    """Image Loader helper function."""
    return Image.open(path.rstrip("\n")).convert('RGB')


class TripletImageLoader(Dataset):
    """Image Loader for Tiny ImageNet."""

    def __init__(self, base_path, triplets_filename, transform=None, loader=image_loader):
        """
        Image Loader Builder.

        Args:
            base_path:
            filenames_filename: text file with each line containing the path to an image e.g., `images/class1/sample.JPEG`
            triplets_filename: A text file with each line containing three images
            transform: torchvision.transforms
            loader: loader for each image
        """
        self.base_path = base_path

        # self.filenamelist = []
        # for line in open(filenames_filename):
        #     self.filenamelist.append(line.rstrip('\n'))

        triplets = []
        for line in open(triplets_filename):
            line_array = line.split(",")
            triplets.append((line_array[0], line_array[1], line_array[2]))
        self.triplets = triplets

        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """Get triplets in dataset."""
        # path1, path2, path3 = self.triplets[index]
        # img1 = self.loader(os.path.join(self.base_path, self.filenamelist[int(path1)]))
        # img2 = self.loader(os.path.join(self.base_path, self.filenamelist[int(path2)]))
        # img3 = self.loader(os.path.join(self.base_path, self.filenamelist[int(path3)]))
        path1, path2, path3 = self.triplets[index]
        a = self.loader(os.path.join(self.base_path, path1))
        p = self.loader(os.path.join(self.base_path, path2))
        n = self.loader(os.path.join(self.base_path, path3))

        if self.transform is not None:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)

        return a, p, n

    def __len__(self):
        return len(self.triplets)






def gen_idx(idx, mode):
    """Generate random index for negative/positive images."""
    if mode == "class":
        random_class_idx = random.randint(0, 200)
        while idx == random_class_idx:
            random_class_idx = random.randint(0, 200)
        return random_class_idx

    elif mode == "image":
        random_img_idx = random.randint(0, 500)
        while idx == random_img_idx:
            random_img_idx = random.randint(0, 500)
        return random_img_idx

    else:
        raise ValueError('We do not support this mode right now!')



class TinyImageNet(Dataset):
    """TinyImageNet."""

    def __init__(self, root, transform=None, train=False):
        """TinyImageNet Class Builder."""
        self.transform = transform
        self.root = root

        self.train_images = []
        self.test_images = []

        self.train = train

        # preprocess to get folder list
        # [[n00001740], [entity]]
        # [[n00001930], [physical entity]]
        # self.info = preprocess(file=os.path.join(root, "words.txt"))

        if self.train:

            self.info = os.listdir(os.path.join(root, "train"))

            # training list: list of lists of images names
            self.train_dict = {}
            for info in self.info:
                if info != ".DS_Store":
                    self.train_dict[info] = os.listdir(os.path.join(root, "train", info, "images"))
                # self.train_list.append(os.listdir(os.path.join(root, "train", info[0], "images")))

            # read training images
            for folder_name, files in self.train_dict.items():
                training_images = []
                for file in files:
                    imgdir = os.path.join(root, "train", folder_name, "images", file)
                    training_images.append(io.imread(imgdir))
                # training_images = np.array(training_images)
                self.train_images.append(training_images)


        else:

            # self.train_list = os.listdir(os.path.join(self.train_root, "iamges"))
            self.test_list = os.listdir(os.path.join(root, "val", "images"))


            # read test images
            for file in self.test_list:
                path = os.path.join(root, "val", "images")
                imgs = os.listdir(os.path.join(root, "val", "images"))
                for img in imgs:
                    self.test_images.append(io.imread(os.path.join(path, img)))

        # self.test_images = np.array(self.test_images)


    def __getitem__(self, index):
        """Get triplets in dataset."""
        # TODO: 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).

        if self.train:

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

        else:
            pass



            # Return a PIL Image for later agmentation
            q_img = Image.fromarray(q_img).convert('RGB')
            p_img = Image.fromarray(p_img).convert('RGB')
            n_img = Image.fromarray(n_img).convert('RGB')

            if self.transform is not None:
                q_img = self.transform(q_img)
                p_img = self.transform(p_img)
                n_img = self.transform(n_img)



        # TODO: 3. Return a data pair (e.g. image and label).
        return q_img, p_img, n_img


    def __len__(self):
        """Get length of dataset."""
        count = len(self.train_images) * len(self.train_images[0])
        return count




class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)










# ha
