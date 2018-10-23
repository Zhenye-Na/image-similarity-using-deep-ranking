"""
Image Similarity using Deep Ranking.

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import numpy as np
from PIL import Image


def gen_mean_std():
    """Generate mean and std for Tiny ImageNet dataset."""
    image_list = []
    with open("../triplets.txt") as f:
        lines = [line.rstrip('\n').split(",") for line in f]
        for line in lines:
            image_list.append(line[0])

    images = []

    for image in image_list:
        img = np.asarray(Image.open(image).convert('RGB'))
        images.append(img)

    images = np.array(images)
    mean = np.mean(images, axis=2)
    std = np.mean(images, axis=2)

    return mean, std


if __name__ == '__main__':
    mean, std = gen_mean_std()
    print(mean, std)
