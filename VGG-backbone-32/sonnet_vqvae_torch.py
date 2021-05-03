# modified from the tensorflow implementation of
# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
import pdb

from config import *

class VectorQuantizer(nn.Module):
    """ Modified from tensorflow Sonnet module representing the VQ-VAE layer.
      Implements the algorithm presented in
      'Neural Discrete Representation Learning' by van den Oord et al.
      https://arxiv.org/abs/1711.00937
      Input any tensor to be quantized. Last dimension will be used as space in
      which to quantize. All other dimensions will be flattened and will be seen
      as different examples to quantize.
      The output tensor will have the same shape as the input.
      For example a tensor with shape [16, 32, 32, 64] will be reshaped into
      [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
      independently.
      But this version is to quatize the value of each pixel.
      For example a tensor with shape [16, 1, 32, 32] will be reshaped into
      [16384, 1] and all 16384 pixel values (each scalar)  will be quantized
      independently.
      Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
        commitment_cost: scalar which controls the weighting of the loss terms
          (see equation 4 in the paper - this variable is Beta).
      """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25, rd_init=True):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        if rd_init:
            self.weight = nn.Parameter(torch.rand(self.embedding_dim, self.num_embeddings))
        else:
            # self.weight = nn.Parameter(torch.linspace(0.0, 1.0, num_embeddings).unsqueeze(0).expand(embeddings_dim, num_embeddings))
            self.weight = nn.Parameter(torch.linspace(lin_min, lin_max,
                                    self.num_embeddings).unsqueeze(0).expand(self.embedding_dim, num_embeddings))

        print('Init emb weight:', self.weight.data)

    def forward(self, input):
        # input size (batch_size, emb_dim, *)
        # assert input.size(1)==self.embedding_dim
        # pdb.set_trace()
        dims = list(range(len(input.size())))
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        # self.weight = torch.sigmoid(self.weight) # _sigw
        # if num_arbitrary_dims:
        #     emb_expanded = self.weight.view(self.weight.shape[0], *([1] * num_arbitrary_dims), self.weight.shape[1])
        # else:
        #     emb_expanded = self.weight

        # find nearest neighbors
        # dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        # dist = torch.pow(x_expanded - emb_expanded, 2)  # (batch_size, input.size(1), *, self.weight.size(1))  # new2
        dist = torch.pow(x_expanded - self.weight, 2)  # (batch_size, input.size(1), *, self.weight.size(1))  # new2
        # dist = torch.pow(x_expanded - torch.sigmoid(self.weight), 2)  # (batch_size, input.size(1), *, self.weight.size(1)) # _sigw
        _, argmin = dist.min(-1) # argmin.size() == input.size()
        shifted_shape = [input.shape[0], *list(input.shape[2:]), input.shape[1]] # == input.size()
        quantized  = self.weight.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])
        # quantized  = torch.sigmoid(self.weight).t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1]) # _sigw
        quantized = quantized.contiguous()
        # e_latent_loss = F.l1_loss(input, quantized.detach())
        # q_latent_loss = F.l1_loss(quantized, input.detach()) # _l1
        e_latent_loss = F.mse_loss(input, quantized.detach())
        q_latent_loss = F.mse_loss(quantized, input.detach()) # default
        # e_latent_loss = F.cosine_similarity(input, quantized.detach(), dim=1).mean()
        # q_latent_loss = F.cosine_similarity(quantized, input.detach(), dim=1).mean() # _cos
        # loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # pdb.set_trace()
        quantized = input + (quantized - input).detach()

        # return quantized, loss
        return quantized, q_latent_loss, e_latent_loss


