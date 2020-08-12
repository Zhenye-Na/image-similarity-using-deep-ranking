"""
Image Similarity using Deep Ranking.

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import time
import argparse

import torch
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from PIL import Image
from joblib import dump, load
from numpy import linalg as LA
from torch.autograd import Variable
from utils import TinyImageNetLoader, tranform_test_img

from net import *

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

def load_train_embedding():
    embedding_space = np.fromfile("../embedded_features_train.txt", dtype=np.float32)
    embedding_space = embedding_space.reshape(-1, 4096)
    return embedding_space

def load_train_images():
    # list of traning images names, e.g., "../tiny-imagenet-200/train/n01629819/images/n01629819_238.JPEG"
    # update to get class names
    training_images = []
    for line in open("../triplets.txt"):
        line_array = line.split(",")
        if line_array[0] not in training_images:
            training_images.append(line_array[0])
    t3 = time.time()
    print("Get all training images, Done ... | Time elapsed {}s".format(t3 - t2))
    return training_images

def gen_embedding(net, data, is_gpu):
    if is_gpu:
        data = data.cuda()
    data = Variable(data)

    embedded, _, _ = net(data, data, data)
    embedded_numpy = embedded.data.cpu().numpy()
    return embedded_numpy

def load_net():
    net = TripletNet(resnet101())

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    if is_gpu:
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    print('==> Retrieve model parameters ...')
    checkpoint = torch.load("../checkpoint/checkpointcheckpoint.pth.tar")
    # start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    net.load_state_dict(checkpoint['state_dict'])

    net.eval()
    return net

def plot_results(query, results, images):
    """
    Plot the query images and top 5 similar images
    Args:
        query: path to query image
        results: predictions from neighbour model
        images: path to embedding space images
    """
  
    plt.subplot(1, 6, 1)
    plt.imshow(np.asarray(Image.open(query)))
    for indx, val in enumerate(results[1][0]):
        if indx == 6:
            break
        elif indx > 0:
            plt.subplot(1, 6, indx+1)
            a = np.asarray(Image.open(images[val]))
            plt.axis('off')
            plt.imshow(a)
    
    plt.show()
    
def calculate_accuracy(trainloader, testloader, is_gpu):
    """
    Calculate accuracy for TripletNet model.

        1. Form 2d array: Number of training images * size of embedding
        2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
        3. Perform subtraction between the two 2D arrays
        4, Take L2 norm of the 2d array (after subtraction)
        5. Get the 30 min values (argmin might do the trick)
        6. Repeat for the rest of the embeddings in the test set

    """
    net = load_net()

    t1 = time.time()
    # dictionary of test images with class
    class_dict = get_classes()
    t2 = time.time()
    print("Get all test image classes, Done ... | Time elapsed {}s".format(t2 - t1))
    
    training_images = load_train_images()

    embedded_features_train = load_train_embedding()

    neigh = KNeighborsClassifier(
        n_neighbors=30, weights='distance', algorithm='kd_tree', n_jobs=-1)
    neigh.fit(embedded_features_train,
              np.array(training_images).reshape, 1)

    # TODO: 2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
    embedded_features_t = []
    with torch.no_grad():
        for test_id, test_data in enumerate(testloader):

            if test_id % 5 == 0:
                print("Now processing {}th test image".format(test_id))
            
            embedded_test_numpy = gen_embedding(net, test_data, is_gpu)
            
            embedded_features_t.append(embedded_test_numpy)

        embedded_features_test = np.concatenate(embedded_features_t, axis=0)

        print(neigh.score(embedded_features_test, training_images))

def train_neighbor_model(embedding_data, K=500):
    """
    Chose nearest neighbour model becasue we can perform image similarity 
    analysis even without labeled data using embedding space as training data
    """
    # n_neighbors is 500 because there are 200 classes and images are 100,000 in training space
    neighbor_model = NearestNeighbors(n_neighbors=500, algorithm='kd_tree', n_jobs=-1)
    neighbor_model.fit(embedding_data)
    dump(neighbor_model, 'models/neighbor_model.joblib')
    return neighbor_model

def predict_unseen(test_query, is_gpu):
    """
    Predict similar images from embedding space
     Args:
            test_query: Search similar images as this image
    Returns: void
        plot 5 similar images as query image
    """
    training_images = load_train_images()
    embedding_space = load_train_embedding()
    
    try:
        neighbor_model = load('models/neighbor_model.joblib')
    except:
        neighbor_model = train_neighbor_model(embedding_space)
          
    query_img = tranform_test_img(Image.open(test_query).convert('RGB')).reshape(1, 3, 224, 224)
    
    net = load_net()
    
    query_embed = gen_embedding(net, query_img, is_gpu)
    
    predictions = neighbor_model.predict(query_embed)
    
    plot_results(test_query, predictions, training_images)
    
def get_classes(filename="../tiny-imagenet-200/val/val_annotations.txt"):
    """
    Get corresponding class name for each val image.

    Args:
        filename: txt file which contains image name and corresponding class name

    Returns:
        class_dict: A dictionary which maps from image name to class name
    """
    class_dict = {}
    for line in open(filename):
        line_array = line.rstrip("\n").split("\t")
        class_dict[line_array[0]] = line_array[1]

    return class_dict


def main():
    """Main pipleine for image similarity using deep ranking."""
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', type=str, default="",
                        help='train/val data root')
    parser.add_argument('--batch_size_train', type=int,
                        default=25, help='training set input batch size')
    parser.add_argument('--batch_size_test', type=int,
                        default=1, help='test set input batch size')

    parser.add_argument('--is_gpu', type=bool, default=True,
                        help='whether training using GPU')
    
    parser.add_argument('--predict_similar_images', type=str, default="",
                        help='path to query image')

    # parse the arguments
    args = parser.parse_args()

    # load triplet dataset
    trainloader, testloader = TinyImageNetLoader(
        args.dataroot, args.batch_size_train, args.batch_size_test)

    # calculate test accuracy
    calculate_accuracy(trainloader, testloader, args.is_gpu)
    
    if args.predict_similar_images != "":
        predict_unseen(args.predict_similar_images, args.is_gpu)
        

if __name__ == '__main__':
    main()
