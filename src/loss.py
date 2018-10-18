"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import torch
import torch.nn as nn


class TripletLoss(nn.Module):

    def __init__(self, batch_size):
        super(TripletLoss, self).__init__()

        self.batch_size = batch_size

    def forward(self, x, y):
        loss = torch.cuda.FloatTensor([[0.0]])
        g = torch.cuda.FloatTensor([[1.0]])

        for i in range(0, self.batch_size, 3):
            D_q_p = torch.sqrt(torch.sum((q_embedding - p_embedding) ** 2))
            D_q_n = torch.sqrt(torch.sum((q_embedding - n_embedding) ** 2))
            loss += torch.max(g + D_q_p - D_q_n, 0)

        loss = loss / (self.batch_size / 3)
        zero = torch.cuda.FloatTensor([[0.0]])

        return torch.max(loss,zero)
