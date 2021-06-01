# adapted from https://github.com/nadavbh12/VQ-VAE

import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F

from config import *

import pdb


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        # if input.size(1) != emb.size(0):
        #     raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
        #                        format(input.size(1), emb.size(0)))

        # emb = emb.expand(input.size(1), emb.size(1))
        emb_ex = emb.expand(input.size(1), emb.size(1)) # new2

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        # ctx.emb_dim = emb.size(0)
        # ctx.num_emb = emb.size(1)
        ctx.emb_dim = emb_ex.size(0)
        ctx.num_emb = emb_ex.size(1) # new2
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            # emb_expanded = emb.view(emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
            emb_expanded = emb_ex.view(emb_ex.shape[0], *([1] * num_arbitrary_dims), emb_ex.shape[1])  # new2
        else:
            # emb_expanded = emb
            emb_expanded = emb_ex  # new2

        # find nearest neighbors
        # dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        dist = torch.pow(x_expanded - emb_expanded, 2) # (batch_size, emb_dim, *, num_emb)  # new2
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *list(input.shape[2:]) ,input.shape[1]]
        # pdb.set_trace()
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])  # new2

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        # pdb.set_trace()
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            # pdb.set_trace()
            # grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
            #                      idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
            grad_emb = torch.sum(grad_output.data.view(-1, 1) *
                                 idx_avg_choices.view(-1, ctx.num_emb), 0, keepdim=True) # new2
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, rd_init=True):
        super(NearestEmbed, self).__init__()
        if rd_init:
            self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))
        else:
            # self.weight = nn.Parameter(torch.linspace(0.0, 1.0, num_embeddings).unsqueeze(0).expand(embeddings_dim, num_embeddings))
            self.weight = nn.Parameter(torch.linspace(lin_min, lin_max, num_embeddings).unsqueeze(0).expand(embeddings_dim, num_embeddings))

        print('Init emb weight:', self.weight.data)

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)


# adapted from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py#L25
# that adapted from https://github.com/deepmind/sonnet


class NearestEmbedEMA(nn.Module):
    def __init__(self, n_emb, emb_dim, decay=0.99, eps=1e-5, rd_init=True):
        super(NearestEmbedEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        self.embeddings_dim = emb_dim
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        if rd_init:
            embed = torch.rand(emb_dim, n_emb)
        else:
            # embed = torch.linspace(0.0, 1.0, n_emb).unsqueeze(0).expand(emb_dim, n_emb)
            embed = torch.linspace(lin_min, lin_max, n_emb).unsqueeze(0).expand(emb_dim, n_emb)
        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.zeros(n_emb))
        self.register_buffer('embed_avg', embed.clone())

        print('Init emb weight ema:', self.weight.data)

    def forward(self, x, weight_sg=None):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        emb_ex = self.weight.expand(x.size(1), self.weight.size(1))  # new2
        #emb_avg_ex = self.embed_avg.expand(x.size(1), self.weight.size(1))  # new2

        dims = list(range(len(x.size())))
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        # if num_arbitrary_dims:
        #     emb_expanded = self.weight.view(self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
        # else:
        #     emb_expanded = self.weight

        emb_size = x.size(1)
        if num_arbitrary_dims:
            #emb_expanded = self.weight.expand(emb_size, self.n_emb).view(self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
            emb_expanded = emb_ex.expand(emb_size, self.n_emb).view(self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
        else:
            #emb_expanded = self.weight.expand(emb_size, self.n_emb)
            emb_expanded = emb_ex.expand(emb_size, self.n_emb)

        # find nearest neighbors
        # dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        dist = torch.pow(x_expanded - emb_expanded, 2)  # (batch_size, emb_dim, *, num_emb) # new2
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        # result = emb_ex.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1]) # (batch_size, emb_dim, *, num_emb) # new2
        result = self.weight.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1]) # (batch_size, emb_dim, *, num_emb) # new2
        # result = self.weight.expand(emb_size, self.n_emb).t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])

        if self.training:
            latent_indices = torch.arange(self.n_emb).type_as(argmin)
            emb_onehot = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(x.data)
            n_idx_choice = emb_onehot.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            # pdb.set_trace()
            # flatten = x.permute(1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)
            num_arbitrary_dims = len(dims) - 2
            if num_arbitrary_dims:
                # flatten = x.permute(1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)
                # flatten = x.permute(1, 0, *dims[-2:]).contiguous().view(1, -1)
                flatten = x.view(1, -1)
            else:
                # flatten = x.permute(1, 0).contiguous()
                # flatten = x.permute(1, 0).contiguous().view(1, -1)
                flatten = x.view(1, -1)

            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, n_idx_choice
            )
            # pdb.set_trace()
            embed_sum = flatten @ emb_onehot # -----dc0.99
            # embed_sum = torch.pow(flatten.t() - emb_onehot, 2).mean(0) # ----dc0.99_s
            #pdb.set_trace()
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            #emb_avg_ex.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            #pdb.set_trace()

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_emb * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.weight.data.copy_(embed_normalized) # ---dc0.99
            # self.weight.data.copy_(self.embed_avg) # -------dc0.99_s

            #embed_normalized = emb_avg_ex / cluster_size.unsqueeze(0)
            #self.weight.data.copy_(embed_normalized.mean(0, keepdim=True))
            #self.embed_avg.data.copy_(emb_avg_ex.mean(0, keepdim=True))

        return result, argmin
