# Image Similarity using Deep Ranking

## Table of contents

- [Introduction](#introduction)
- [Project Description](#project-description)
    - [Overview](#overview)
    - [Ranking Layer](#ranking-layer)
    - [Testing stage](#testing-stage)
    - [Triplet Sampling Layer](#triplet-sampling-layer)
    - [Tips](#tips)
- [References](#references)


## Introduction

The goal of this project is to get hands-on experience concerning the computer vision task of image similarity. Like most tasks in this field, it's been aided by the ability of deep networks to extract image features.

The task of image similarity is retrieve a set of `N` images closest to the query image. One application of this task could involve visual search engine where we provide a query image and want to find an image closest that image in the database.


<p align="center">
<img src="./fig/query-image.png" width="80%">
</p>


## Project Description

### Overview

You will design a simplified version of the deep ranking model as discussed in the paper. Your network architecture will look exactly the same, but the details of the triplet sampling layer will be a lot simpler. The architecture consists of $3$ identical networks $(Q,P,N)$. Each of these networks take a single image denoted by $p_i$ , $p_i^+$ , $p_i^-$ respectively.

- $p_i$: Input to the $Q$ (Query) network. This image is randomly sampled across any class.
- $p_i^+$: Input to the $P$ (Positive) network. This image is randomly sampled from the **SAME** class as the query image.
- $p_i^-$: Input to the $N$ (Negative) network. This image is randomly sample from any class **EXCEPT** the class of $p_i$.


<p align="center">
<img src="./fig/model.png" width="80%">
</p>


The output of each network, denoted by $f(p_i)$, $f(p_i^+)$, $f(p_i^-)$ is the feature embedding of an image. This gets fed to the ranking layer.


### Ranking Layer

The ranking layer just computes the triplet loss. It teaches the network to produce similar feature embeddings for images from the same class (and different embeddings for images from different classes). $g$ is a gap parameter used for regularization purposes.

$$ l(p_i, p_i^+, p_i^-) = \max \{ 0, g + D \big(f(p_i), f(p_i^+) \big) - D \big( f(p_i), f(p_i^-) \big)  \} $$

$D$ is the **Euclidean Distance** between $f(p_i)$ and $f(p_i^{+/-})$.


$$ D(p, q) = \sqrt{(q_1 − p_1)^2 + (q_2 − p_2)^2 + \dots + (q_n − p_n)^2} $$


$g$ is the gap parameter. We use the default value of $1.0$, but you can tune it if you’d like (make sure it's positive).


### Testing stage

The testing (inference) stage only has one network and accepts only one image. To retrieve the top $n$ similar results of a query image during inference, the following procedure is followed:

1. Compute the feature embedding of the query image.
2. Compare (euclidean distance) the feature embedding of the query image to all the feature embeddings in the training data (i.e. your database).
3. Rank the results - sort the results based on Euclidean distance of the feature embeddings.


### Triplet Sampling Layer

One of the main contributions of the paper is the triplet sampling layer. Sampling the query image (randomly) and the positive sample image (randomly from the same class as the query image) are quite straightforward.

Negative samples are composed of two different types of samples: in-class and out-of-class. For this project, we will implement out-of-class samples only. Again, out-of-class samples are images sampled randomly from any class except the class of the query image.


### Tips

1. We recommend you use the ResNet architecture you implemented in your previous homework.
2. Use the data loader - it'll help a lot in loading the images in parallel (there is a `num_workers` option)
3. Sample your triplets beforehand, so during training all you’re doing is reading images.
4. Make sure you load your model with pre-trained weights. This will greatly reduce the time to train your ranking network.
5. Blue Waters training time is approximate 24-36 hours, so please start early.


## References

[1] bla [*"Learning Fine-grained Image Similarity with Deep Ranking"*](https://arxiv.org/abs/1404.4661). arXiv:1404.4661  
[2] Akarsh Zingade [*"Image Similarity using Deep Ranking"*](https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978)