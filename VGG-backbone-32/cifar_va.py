"""
    This code is adapted from PyTorch training code for
    "Paying More Attention to Attention: Improving the Performance of
                Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    2017 Sergey Zagoruyko
"""


import argparse
import os
import json
import numpy as np
import math
import scipy.misc
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as T
from torchvision import datasets, models
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import utils_my
import pdb
import random
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
from sonnet_vqvae_torch import VectorQuantizer
from sonnet_vqvae_torch import VectorQuantizer
from config import *
from pathlib import Path
from numbers import Number
from initialize import *
from blocks import *
from cifar_dataset import CIFAR10, CIFAR100

# $tensorboard --bind_all --port=6001 --logdir=<log_dir>

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
parser.add_argument('--phase', default='train', type=str, help='running phase')

# Model options
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--dset_dir', default='../dataset', type=str, help='dataset directory path')
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=2, type=int)
parser.add_argument('--env_name', default='', type=str, help='visdom env name of model')
parser.add_argument('--filename', default='best_acc.tar', type=str, help='filename of model')
parser.add_argument('--summary_dir', default='tf_log', type=str, help='path of tensorboard logs')
parser.add_argument('--tensorboard', default=False, type=utils_my.str2bool, help='enable tensorboard')


# Training options
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_ratio', default=1e-4, type=float, help='ratio of emb weight learning rate')
# parser.add_argument('--epochs', default=200, type=int, metavar='N',
#                     help='number of total epochs to run')
parser.add_argument('--epoch', default=200, type=int, metavar='N', help='epoch size')
parser.add_argument('--weight_decay', default=0.0005, type=float)
# parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
#                     help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.5, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)
parser.add_argument('--temperature', default=4, type=float)
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--beta', default=0, type=float)

# Device options
parser.add_argument('--cuda', action='store_true')
# parser.add_argument('--save', default='', type=str,
#                     help='save parameters and logs in this folder')
parser.add_argument('--ckpt_dir', default='VGG_32', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# VA param
parser.add_argument('--va_beta', default=1e-2, type=float, help='beta')
parser.add_argument('--va_beta_anneal', default=1e-2, type=float, help='beta anneal')
parser.add_argument('--K', default=256, type=int, help='dimension of encoding Z')
parser.add_argument('--att_K', default=256, type=int, help='dimension of attention A')
parser.add_argument('--num_sample', default=num_sample, type=int, help='the number of samples')
parser.add_argument('--seed', default=random.randint(2, 10000), type=int, help='random seed')
parser.add_argument('--num_avg', default=4, type=int, help='the number of samples when\
               perform multi-shot prediction')

parser.add_argument('--att_dir', default='att_maps', type=str, help='att_mask directory path')
parser.add_argument('--return_att', default=False, type=utils_my.str2bool, help='enable return att_mask')
parser.add_argument('--qt_num', default=20, type=int, help='number of quantum')
parser.add_argument('--vq_coef', default=0.4, type=float, help='weight for vq_loss')
parser.add_argument('--comit_coef', default=0.1, type=float, help='weight for commit_loss')
parser.add_argument('--rd_init', default=False, type=utils_my.str2bool, help='randomly init quantum layer or not.')
parser.add_argument('--decay', default=None, type=float, help='decay quantum layer if not set as None')
parser.add_argument('--qt_trainable', default=True, type=utils_my.str2bool, help='quantum is trainable or not.')

parser.add_argument('--model_init', default=3, type=int, help='model init: 0 kaimingNormal, 1 kaimingUniform, 2 xavierNormal, 3 xavierUniform')



def create_dataset(opt, train):

    if 'CIFAR10' == opt.dataset or 'CIFAR100' == opt.dataset:
        input_size = 32
        pad_size = 4

        if train:
            transform = T.Compose([
                T.Resize((input_size, input_size)),
                # T.RandomCrop(input_size, padding=pad_size), # comment for _nopad
                T.RandomCrop(input_size, padding=pad_size, padding_mode='reflect'), # for _rft
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

        else:
            transform = T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        return getattr(datasets, opt.dataset)(opt.dset_dir, train=train, download=True, transform=transform)

    elif 'SVHN' == opt.dataset:
        # def target_transform(target):
        #     return int(target) - 1 # no need to -1; already 0~9
        # input_size = 32
        # pad_size = 4
        # transform = T.Compose([
        #     T.Resize((input_size, input_size)),
        #     T.RandomCrop(input_size, padding=pad_size), # _aug
        #     # T.RandomCrop(input_size, padding=pad_size, padding_mode='reflect'), # _aug_rft
        #     T.RandomHorizontalFlip(),  # ========================
        #     T.ToTensor(),
        #     # T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
        #     #             np.array([63.0, 62.1, 66.7]) / 255.0),
        #     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ], )
        if train:
            input_size = 32
            pad_size = 4
            transform = T.Compose([
                T.Resize((input_size, input_size)),
                # T.RandomCrop(input_size, padding=pad_size),  # _aug
                T.RandomCrop(input_size, padding=pad_size, padding_mode='reflect'), # _aug_rft
                T.RandomHorizontalFlip(),  # ========================
                T.ToTensor(),
                # T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                #             np.array([63.0, 62.1, 66.7]) / 255.0),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ], )
            return getattr(datasets, opt.dataset)(opt.dset_dir, split='train', download=True, transform=transform) # target_transform=target_transform
        else:
            transform = T.Compose([
                T.ToTensor(),
                # T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                #             np.array([63.0, 62.1, 66.7]) / 255.0),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ], )
            return getattr(datasets, opt.dataset)(opt.dset_dir, split='test', download=True, transform=transform)


# ---------- va definition --------------
# conv part, MLP part, deconv part
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


''' Vgg_bn_lrn '''
class Vgg16_bn_lrn_VANet(nn.Module):

    def __init__(self, K=K_dim, att_K=A_dim, return_att=False,
                 n_cls = n_class, init='xavierUniform'):
        super(Vgg16_bn_lrn_VANet, self).__init__()
        print('vgg_bn_lrn_single')
        self.K = K
        self.att_K = att_K
        self.return_att = return_att
        ''''''
        # im_size = 64                                # [bs, 3, 32, 32] # mu_s2
        im_size = 32                                # [bs, 3, 32, 32] # mxp2_1, mxp2_2, mxp4_3, mxp_d2_3, ap_3
        conv_block1 = ConvBlock(3, 64, 2)           # [bs, 64, 32, 32]
        conv_block2 = ConvBlock(64, 128, 2)         # [bs, 128, 32, 32]
        conv_block3 = ConvBlock(128, 256, 3)        # [bs, 256, 32, 32] --> [bs, 256, 16, 16]
        conv_block4 = ConvBlock(256, 512, 3)        # [bs, 512, 16, 16] --> [bs, 512, 8, 8]
        conv_block5 = ConvBlock(512, 512, 3)        # [bs, 512, 8, 8] --> [bs, 512, 4, 4]
        conv_block6 = ConvBlock(512, 512, 2, pool=True)       # [bs, 512, 1, 1]
        dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size / 32), padding=0, bias=True) # [bs, 512, 1, 1]

        self.features = nn.Sequential(
            conv_block1,
            conv_block2,
            conv_block3,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            # conv_block4,
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            # conv_block5,
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            # conv_block6,
            # dense,
        )

        self.channel = 512

        self.encode = nn.Sequential(
            # conv_block1,
            # conv_block2,
            # conv_block3,
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            conv_block4,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            conv_block5,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            conv_block6,
            dense,
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(self.channel, 2 * self.K)
            )

        self.att_dim = 16
        self.att_channel = 256
        self.att_module_mu = nn.Sequential(
            # nn.Conv2d(self.att_channel, self.att_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(self.att_channel),
            # nn.ReLU(inplace=True),  # mu_m1
            # nn.Conv2d(self.att_channel, self.att_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(self.att_channel),
            # nn.ReLU(inplace=True),  # mu_m2
            # nn.Conv2d(self.att_channel, self.att_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(self.att_channel),
            # nn.ReLU(inplace=True),  # =================== # _mu_m3
            nn.Conv2d(self.att_channel, 1, 3, stride=1, padding=1),
            nn.Sigmoid()  # sigmu
        )

        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1), #conv2
        )

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, n_cls))

        # self.decode = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.Linear(self.K, self.K),
        #     nn.ReLU(True),
        #     nn.Linear(self.K, n_cls))  # v2

        # self.decode = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.Linear(self.K, n_cls))  # v3

        # self.decode = nn.Sequential(
        #     nn.Linear(self.K, n_cls)) # v4


        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    # f4 f7 f11
    def forward(self, x, num_sample=1, train=True, return_all=False):
        # if x.dim() > 2 : x = x.view(x.size(0),-1) #(bs, X_dim) # x is in [-1, 1]
        f = self.features(x)
        # pdb.set_trace()
        att_mu = self.att_module_mu(f)
        #att_mu = att_mu.view(x.size(0), -1) # conv

        att_std = self.att_module_std(att_mu) # conv, conv2


        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1) # conv2


        #att_std = self.att_module_std(f)
        #att_std = att_std.view(x.size(0), -1)  # conv3

        att_std = F.softplus(att_std - 5, beta=1) # comment this for new5, new6; best choice?
        # att_std = F.softplus(att_std, beta=1) # new3_2_sigmu
        # att_std = F.softplus(att_std - 10, beta=1) # new3_2_sigmu

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample) # (num_sample, bs, self.att_K)
        # in the new version, is (num_sample, bs, self.att_dim * self.att_dim)
        # pdb.set_trace()
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)
        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)
        # pdb.set_trace()
        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[f.size(2), f.size(3)]) # add for tinyim model
        # pdb.set_trace()
        # pdb.set_trace()

        # f = self.encode_pre(f)
        if num_sample > 1:
            f_rpt = f.repeat(num_sample, 1, 1, 1)
            f_att = torch.mul(f_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            f_att = f_rpt + f_att
        else:
            f_att = torch.mul(f, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            f_att = f + f_att

        statistics = self.encode(f_att)
        mu = statistics[:,:self.K]
        std = F.softplus(statistics[:,self.K:]-5, beta=1)

        encoding = self.reparametrize_n(mu,std,num_sample) # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K) # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        # pdb.set_trace()

        # if num_sample == 1 : logit = logit.squeeze(0).squeeze(0)
        # # elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)
        # elif num_sample > 1 : logit = logit.mean(0).mean(0)

        logit = logit.mean(0).mean(0)

        # # another encoding p(z|x)
        # statistics_ori = self.encode_ori(f)
        # mu_ori = statistics_ori[:, :self.K]
        # std_ori = F.softplus(statistics_ori[:, self.K:] - 5, beta=1)
        #
        # encoding_ori = self.reparametrize_n(mu_ori, std_ori, num_sample)  # (num_sample, num_sample*bs, self.K)
        # # pdb.set_trace()
        # encoding_ori = encoding_ori.view(num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        # logit_ori = self.decode(encoding_ori)
        # logit_ori = logit_ori.mean(0)

        if train:
            # # normal_prior = Independent(Normal(torch.tensor([0.0]), torch.tensor([1.0])), 1)
            # normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            # # ori_latent_prior = Independent(Normal(loc=mu_ori, scale=torch.exp(std_ori)).expand(normal_prior.batch_shape), 1)
            # ori_latent_prior = Independent(Normal(loc=mu_ori, scale=torch.exp(std_ori)), 1)
            # latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1) # Independent(base_distribution, reinterpreted_batch_ndims, validate_args=None)
            #
            # latent_loss1 = torch.mean(self.kl_divergence(ori_latent_prior, normal_prior))
            # latent_loss2 = torch.mean(self.kl_divergence(latent_prior, ori_latent_prior))

            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            # latent_loss = torch.zeros((num_sample, 1)).to(mu.device)
            # batch_size = mu.size(0)
            # if num_sample==12:
            #     pdb.set_trace()
            # for nidx in range(num_sample):
            #     mu_cur = mu[nidx * batch_size:(nidx + 1) * batch_size]
            #     std_cur = std[nidx * batch_size:(nidx + 1) * batch_size]
            #     # pdb.set_trace()
            #     latent_prior = Independent(Normal(loc=mu_cur, scale=torch.exp(std_cur)), 1)
            #     # latent_prior = Independent(Normal(loc=mu[nidx:(nidx+1)*batch_size], scale=torch.exp(std[nidx:(nidx+1)*batch_size])), 1)
            #     latent_loss[nidx] = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                # 1) directly return att_mask
                # return logit, latent_loss, (att_mask, att_mask, att_mask)
                return logit, latent_loss, att_mask

                # 2) return distribution of attention map
                # att_dist = Independent(Normal(loc=att_muscale=torch.exp(att_std)), 1)
                # return logit, logit_ori, latent_loss1, latent_loss2.mean(), (g0, g1, att_dist)
            return logit, latent_loss

            # ---------old-------------
            # normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            # latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            # latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))
            #
            # return logit, logit_ori, latent_loss, latent_loss

            # ---------old2-----------
            # return logit, logit_ori, mu, std

            # ---------dcal-----------

        else:
            return logit, att_mask
            # return logit, logit_ori, (att_mask, att_mask, att_mask)

        # pdb.set_trace()

        # if self.return_att:
        #     # return logit, att_mask.mean(0)
        #     return logit, att_mask
        #
        # else:
        #     return (att_mu, att_std), (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(utils_my.cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def kl_divergence(self, latent_space1, latent_space2):
        kl_div = kl.kl_divergence(latent_space1, latent_space2)
        return kl_div

''' Vgg_bn_lrn_qt '''
class Vgg16_bn_lrn_VANet_QT(nn.Module):
    # qt specific params: qt_num, vq_coef, comit_coef, rd_init, decay, qt_trainable,
    def __init__(self, K=K_dim, att_K=A_dim, qt_num=2, vq_coef=0.2, comit_coef=0.4, rd_init=False, decay=None,
                 qt_trainable=False, return_att=False, n_cls = n_class, init='xavierUniform'):
        super(Vgg16_bn_lrn_VANet_QT, self).__init__()
        print('vgg_bn_lrn_single')
        self.K = K
        self.att_K = att_K
        self.qt_num = qt_num
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.rd_init = rd_init
        self.decay = decay
        self.qt_trainable = qt_trainable
        self.return_att = return_att

        if decay is None:
            self.emb = VectorQuantizer(embedding_dim=1, num_embeddings=self.qt_num, rd_init=self.rd_init)  # new nearest_embed
            # self.emb = NearestEmbed(num_embeddings=self.qt_num, embeddings_dim=1, rd_init=self.rd_init)  # new nearest_embed
            # self.emb = NearestEmbed(num_embeddings=self.qt_num, embeddings_dim=X_dim, rd_init=self.rd_init)
        else:
            raise NotImplementedError
            # self.emb = NearestEmbedEMA(n_emb=self.qt_num, emb_dim=1, decay=self.decay, rd_init=self.rd_init)  # new nearest_embed
            # self.emb = NearestEmbedEMA(n_emb=self.qt_num, emb_dim=X_dim, decay=self.decay, rd_init=self.rd_init)

        if not self.qt_trainable:
            for param in self.emb.parameters():
                param.requires_grad = False

        ''''''
        im_size = 32                                # [bs, 3, 32, 32] # mxp2_1, mxp2_2
        # im_size = 64                                # [bs, 3, 32, 32]
        conv_block1 = ConvBlock(3, 64, 2)           # [bs, 64, 32, 32]
        conv_block2 = ConvBlock(64, 128, 2)         # [bs, 128, 32, 32]
        conv_block3 = ConvBlock(128, 256, 3)        # [bs, 256, 32, 32] --> [bs, 256, 16, 16]
        conv_block4 = ConvBlock(256, 512, 3)        # [bs, 512, 16, 16] --> [bs, 512, 8, 8]
        conv_block5 = ConvBlock(512, 512, 3)        # [bs, 512, 8, 8] --> [bs, 512, 4, 4]
        conv_block6 = ConvBlock(512, 512, 2, pool=True)       # [bs, 512, 1, 1]
        dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size / 32), padding=0, bias=True) # [bs, 512, 1, 1]

        self.features = nn.Sequential(
            conv_block1,
            conv_block2,
            conv_block3,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            # conv_block4,
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            # conv_block5,
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            # conv_block6,
            # dense,
        )

        self.channel = 512

        self.encode = nn.Sequential(
            # conv_block1,
            # conv_block2,
            # conv_block3,
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            conv_block4,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            conv_block5,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            conv_block6,
            dense,
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(self.channel, 2 * self.K)
            )

        self.att_dim = 16
        self.att_channel = 256
        self.att_module_mu = nn.Sequential(
            # nn.Conv2d(self.att_channel, self.att_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(self.att_channel),
            # nn.ReLU(inplace=True),  # mu_m1
            # nn.Conv2d(self.att_channel, self.att_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(self.att_channel),
            # nn.ReLU(inplace=True),  # mu_m2
            # nn.Conv2d(self.att_channel, self.att_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(self.att_channel),
            # nn.ReLU(inplace=True),  # =================== # _mu_m3
            nn.Conv2d(self.att_channel, 1, 3, stride=1, padding=1),
            nn.Sigmoid()  # sigmu
        )

        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1), #conv2
        )

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, n_cls))

        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    # f4 f7 f11
    def forward(self, x, num_sample=1, train=True, return_all=False):
        # if x.dim() > 2 : x = x.view(x.size(0),-1) #(bs, X_dim) # x is in [-1, 1]
        f = self.features(x)

        att_mu = self.att_module_mu(f)
        #att_mu = att_mu.view(x.size(0), -1) # conv

        att_std = self.att_module_std(att_mu) # conv, conv2


        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1) # conv2


        #att_std = self.att_module_std(f)
        #att_std = att_std.view(x.size(0), -1)  # conv3

        att_std = F.softplus(att_std - 5, beta=1) # comment this for new5, new6; best choice?
        # att_std = F.softplus(att_std, beta=1) # new3_2_sigmu
        # att_std = F.softplus(att_std - 10, beta=1) # new3_2_sigmu

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample) # (num_sample, bs, self.att_K)
        # in the new version, is (num_sample, bs, self.att_dim * self.att_dim)
        # pdb.set_trace()
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)
        # if att_mode in ['sigmoid', 'sig_sft']:
        #     att_mask = torch.sigmoid(att_mask)
        # if att_mode in ['softmax', 'sig_sft']:
        #     att_mask = F.softmax(att_mask, dim=-1)
        # if att_mode in ['relu']:
        #     att_mask = F.relu(att_mask)
        # # pdb.set_trace()
        # att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        # att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        # # pdb.set_trace()
        # # pdb.set_trace()

        att_mask_flt = att_mask.view(num_sample * x.size(0), self.att_dim * self.att_dim)
        att_mask_q, vq_loss, commit_loss = self.emb(att_mask_flt)
        att_mask_reshape = att_mask_q.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_q = att_mask_q.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        # f = self.encode_pre(f)
        if num_sample > 1:
            f_rpt = f.repeat(num_sample, 1, 1, 1)
            f_att = torch.mul(f_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            f_att = f_rpt + f_att
        else:
            f_att = torch.mul(f, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            f_att = f + f_att

        statistics = self.encode(f_att)
        mu = statistics[:,:self.K]
        std = F.softplus(statistics[:,self.K:]-5, beta=1)

        encoding = self.reparametrize_n(mu,std,num_sample) # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K) # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        # pdb.set_trace()

        # if num_sample == 1 : logit = logit.squeeze(0).squeeze(0)
        # # elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)
        # elif num_sample > 1 : logit = logit.mean(0).mean(0)

        logit = logit.mean(0).mean(0)

        # # another encoding p(z|x)
        # statistics_ori = self.encode_ori(f)
        # mu_ori = statistics_ori[:, :self.K]
        # std_ori = F.softplus(statistics_ori[:, self.K:] - 5, beta=1)
        #
        # encoding_ori = self.reparametrize_n(mu_ori, std_ori, num_sample)  # (num_sample, num_sample*bs, self.K)
        # # pdb.set_trace()
        # encoding_ori = encoding_ori.view(num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        # logit_ori = self.decode(encoding_ori)
        # logit_ori = logit_ori.mean(0)

        if train:
            # # normal_prior = Independent(Normal(torch.tensor([0.0]), torch.tensor([1.0])), 1)
            # normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            # # ori_latent_prior = Independent(Normal(loc=mu_ori, scale=torch.exp(std_ori)).expand(normal_prior.batch_shape), 1)
            # ori_latent_prior = Independent(Normal(loc=mu_ori, scale=torch.exp(std_ori)), 1)
            # latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1) # Independent(base_distribution, reinterpreted_batch_ndims, validate_args=None)
            #
            # latent_loss1 = torch.mean(self.kl_divergence(ori_latent_prior, normal_prior))
            # latent_loss2 = torch.mean(self.kl_divergence(latent_prior, ori_latent_prior))

            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            # latent_loss = torch.zeros((num_sample, 1)).to(mu.device)
            # batch_size = mu.size(0)
            # if num_sample==12:
            #     pdb.set_trace()
            # for nidx in range(num_sample):
            #     mu_cur = mu[nidx * batch_size:(nidx + 1) * batch_size]
            #     std_cur = std[nidx * batch_size:(nidx + 1) * batch_size]
            #     # pdb.set_trace()
            #     latent_prior = Independent(Normal(loc=mu_cur, scale=torch.exp(std_cur)), 1)
            #     # latent_prior = Independent(Normal(loc=mu[nidx:(nidx+1)*batch_size], scale=torch.exp(std[nidx:(nidx+1)*batch_size])), 1)
            #     latent_loss[nidx] = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                # 1) directly return att_mask
                # return logit, latent_loss, (att_mask, att_mask, att_mask)
                return logit, latent_loss, vq_loss, commit_loss, att_mask, att_mask_q

                # 2) return distribution of attention map
                # att_dist = Independent(Normal(loc=att_muscale=torch.exp(att_std)), 1)
                # return logit, logit_ori, latent_loss1, latent_loss2.mean(), (g0, g1, att_dist)
            return logit, latent_loss, vq_loss, commit_loss

            # ---------old-------------
            # normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            # latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            # latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))
            #
            # return logit, logit_ori, latent_loss, latent_loss

            # ---------old2-----------
            # return logit, logit_ori, mu, std

            # ---------dcal-----------

        else:
            return logit, att_mask, att_mask_q
            # return logit, logit_ori, (att_mask, att_mask, att_mask)

        # pdb.set_trace()

        # if self.return_att:
        #     # return logit, att_mask.mean(0)
        #     return logit, att_mask
        #
        # else:
        #     return (att_mu, att_std), (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(utils_my.cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def kl_divergence(self, latent_space1, latent_space2):
        kl_div = kl.kl_divergence(latent_space1, latent_space2)
        return kl_div


def main(opt):
    # opt = parser.parse_args()
    print('parsed options:', vars(opt))

    if '100' in opt.dataset:
        num_classes = 100
    else:
        num_classes = 10

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    use_cuda = torch.cuda.is_available()

    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)

    test_loader = create_iterator(False)

    CFVaNet = Vgg16_bn_lrn_VANet

    print('Model init %s' % ModelInit[opt.model_init])
    net = CFVaNet(opt.K, opt.att_K, opt.return_att, num_classes, init=ModelInit[opt.model_init])

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    params = net.state_dict()

    ckpt_dir = os.path.join(opt.ckpt_dir, opt.env_name)

    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD(net.parameters(), lr,
                   momentum=0.9, weight_decay=opt.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        checkpoint = torch.load(os.path.join(opt.ckpt_dir, opt.resume, opt.filename))
        epoch = checkpoint['epoch']
        state_dict = checkpoint['model_states']['net']
        net.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optim_states']['optim'])

    n_parameters = sum(p.numel() for p in list(params.values()))
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(topk=[1,3,5], accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]


    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    def h_train(sample):
        inputs = sample[0].detach()
        targets = sample[1]

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.to(inputs.device)

        inputs, targets = Variable(inputs), Variable(targets)

        net.train()
        logit, latent_loss  = net(inputs, opt.num_sample)
        class_loss = F.cross_entropy(logit, targets).div(math.log(2))
        info_loss = 0.5 * latent_loss.mean().div(math.log(2))
        total_loss = class_loss + opt.va_beta * info_loss
        return total_loss, logit

    def h_test(sample):
        inputs = sample[0].detach()
        targets = sample[1]

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.to(inputs.device)

        inputs, targets = Variable(inputs), Variable(targets)

        net.eval()
        logit, latent_loss  = net(inputs, opt.num_avg)
        class_loss = F.cross_entropy(logit, targets).div(math.log(2))
        info_loss = 0.5 * latent_loss.mean().div(math.log(2))
        total_loss = class_loss + opt.va_beta * info_loss
        return total_loss, logit

    def log(t, state):
        model_states = {
            # 'net': {k: v.data for k, v in params.items()},  # because the param matrix are defined manually
            # 'net': net.module.state_dict() if torch.cuda.device_count()>1 else net.state_dict(),  # because the param matrix are defined manually
            'net': net.state_dict(),  # because the param matrix are defined manually
        }
        optim_states = {
            'optim': state['optimizer'].state_dict(),
        }

        states = {
            'epoch': t['epoch'],
            'args': opt,  # self.args
            'model_states': model_states,
            'optim_states': optim_states,
            'test_acc': t['test_acc'],
            'test_loss': t['test_loss']
        }

        save_path = os.path.join(opt.ckpt_dir, opt.env_name, opt.filename)
        torch.save(states, save_path)

        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(ckpt_dir, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])
        state['iter_num'] += 1
        state['iter_num_epoch'] += 1

    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(state['loss'].item())
        if state['train']:
            # timer_train.add()
            timer_train.add(opt.batch_size)
        else:
            # timer_test.add()
            timer_test.add(opt.batch_size)

    def on_start(state):
        state['epoch'] = epoch
        if opt.resume != '':
            state['test_acc'] = checkpoint['test_acc']
        else:
            state['test_acc'] = 0.0

        if state['train']:
            state['optimizer'] = optimizer
            if opt.lr==0.01:
                lr_lambda = lambda epoch: 1.0 if epoch==1 else 10.0*np.power(0.5, int(epoch / 25)) # e2 lr=0.01***
            elif opt.lr==0.001:
                lr_lambda = lambda epoch: 0.1*np.power(10, epoch) if epoch<3 else 100.0*np.power(0.5, int(epoch / 25)) # e2e3 lr=0.001***
            else:
                lr_lambda = lambda epoch: np.power(0.5, int(epoch / 25)) # ***

            state['scheduler'] = lr_scheduler.LambdaLR(state['optimizer'], lr_lambda=lr_lambda)

            # state['optimizer'] = Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            # state['scheduler'] = lr_scheduler.ExponentialLR(state['optimizer'], gamma=0.97)

        state['iter_num'] = 0
        summary_dir = os.path.join(opt.summary_dir, opt.env_name)
        if not os.path.exists(summary_dir): os.makedirs(summary_dir)
        state['tf'] = SummaryWriter(log_dir=summary_dir)
        state['tf'].add_text(tag='argument', text_string=str(opt), global_step=state['iter_num'])
        state['iter_num_epoch'] = 0

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]
        state['iter_num_epoch'] = 0

        state['scheduler'].step()
        print("epoch %d learning rate %f" % (state['epoch'], state['optimizer'].param_groups[0]['lr']))

        state['iterator'] = tqdm(train_loader)

    def on_update(state):
        state['iterator'].set_description(
            '[{epoch}]({batch}/{size})  Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                epoch=state['epoch'],
                batch=state['iter_num_epoch'],
                size=len(state['iterator']),
                loss=meter_loss.mean,
                top1=classacc.value()[0],
                top5=classacc.value()[2],
            ))
        if opt.tensorboard:
            if state['iter_num'] % 50 == 0:
                # state['tf'].add_scalar('Train/loss', train_loss, state['iter_num'])
                # pdb.set_trace()
                if torch.cuda.device_count()>1:
                    state['tf'].add_scalars(main_tag='Train/grad',
                                            tag_scalar_dict={
                                                # 'emb': net.module.emb.weight.grad.abs().mean().item(),
                                                'encode': net.module.encode[-1].weight.grad.abs().mean().item(),
                                                'decode': net.module.decode[-1].weight.grad.abs().mean().item(),
                                                'att_mu': net.module.att_module_mu[0].weight.grad.abs().mean().item(),
                                                'att_std': net.module.att_module_std[0].weight.grad.abs().mean().item(),
                                            }, global_step=state['iter_num'])
                    state['tf'].add_scalars(main_tag='Train/loss',
                                            tag_scalar_dict={
                                                'loss': meter_loss.mean, # state['train_loss'],
                                            }, global_step=state['iter_num'])
                else:
                    state['tf'].add_scalars(main_tag='Train/grad',
                                            tag_scalar_dict={
                                                # 'emb': net.emb.weight.grad.abs().mean().item(),
                                                'encode': net.encode[-1].weight.grad.abs().mean().item(),
                                                'decode': net.decode[-1].weight.grad.abs().mean().item(),
                                                'att_mu': net.att_module_mu[0].weight.grad.abs().mean().item(),
                                                'att_std': net.att_module_std[0].weight.grad.abs().mean().item(),
                                            }, global_step=state['iter_num'])
                    state['tf'].add_scalars(main_tag='Train/loss',
                                            tag_scalar_dict={
                                                'loss': meter_loss.mean, # state['train_loss'],
                                            }, global_step=state['iter_num'])

    def on_end_epoch(state):
        train_loss = meter_loss.mean
        train_acc = classacc.value()
        train_time = timer_train.value()
        train_at_losses = [m.value() for m in meters_at]
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()
        [m.reset() for m in meters_at]
        state['iter_num_epoch'] = 0

        engine.test(h_test, test_loader)

        test_acc = classacc.value()[0]
        if test_acc > state['test_acc']:
            state['test_acc'] = test_acc
            print(log({
                "train_loss": train_loss,
                "train_acc": train_acc[0],
                "test_loss": meter_loss.mean,
                "test_acc": test_acc,
                "test_acc_top3": classacc.value()[1],
                "test_acc_top5": classacc.value()[2],
                "epoch": state['epoch'],
                "num_classes": num_classes,
                "n_parameters": n_parameters,
                "train_time": train_time,
                "test_time": timer_test.value(),
                # "at_losses_train": train_at_losses,
                # "at_losses_test": [m.value() for m in meters_at],
               }, state))
            print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                           (ckpt_dir, state['epoch'], opt.epoch, test_acc))


    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.hooks['on_update'] = on_update
    engine.train(h_train, train_loader, opt.epoch, optimizer)
    # engine.test(h_test, test_loader)
    # test_acc1, test_acc3, test_acc5 = classacc.value()
    # print('==> id: %s (%d/%d), test_acc1: \33[91m%.2f\033[0m, test_acc3: \33[91m%.2f\033[0m, test_acc5: \33[91m%.2f\033[0m' % \
    #     (ckpt_dir, epoch, opt.epoch, test_acc1, test_acc3, test_acc5))

def main_qt(opt):
    # opt = parser.parse_args()
    print('parsed options:', vars(opt))

    if '100' in opt.dataset:
        num_classes = 100
    else:
        num_classes = 10

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    use_cuda = torch.cuda.is_available()

    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    CFVaNet = Vgg16_bn_lrn_VANet_QT

    print('Model init %s' % ModelInit[opt.model_init])

    net = CFVaNet(opt.K, opt.att_K, opt.qt_num, opt.vq_coef, opt.comit_coef,
                    opt.rd_init, opt.decay, opt.qt_trainable, opt.return_att,
                    n_cls=num_classes, init=ModelInit[opt.model_init])

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    params = net.state_dict()


    ckpt_dir = os.path.join(opt.ckpt_dir, opt.env_name)

    def create_optimizer(opt, lr):
        print('creating optimizer with lr %.6f for base and %.6f for emb' % (lr, lr*opt.lr_ratio))
        my_list = ['emb.weight']
        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, net.named_parameters()))))
        base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, net.named_parameters()))))
        optimizer = SGD([{'params': base_params}, {'params': params, 'lr': lr*opt.lr_ratio}],
                        lr=lr, momentum=0.9, weight_decay=opt.weight_decay)
        return optimizer

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        checkpoint = torch.load(os.path.join(opt.ckpt_dir, opt.resume, opt.filename))
        epoch = checkpoint['epoch']
        state_dict = checkpoint['model_states']['net']
        net.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optim_states']['optim'])

    n_parameters = sum(p.numel() for p in list(params.values()))
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(topk=[1,3,5], accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]


    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)


    def h_train(sample):  # network feed into engine should have 'loss, output'
        inputs = sample[0].detach()
        targets = sample[1]

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.to(inputs.device)

        inputs, targets = Variable(inputs), Variable(targets)

        net.train()
        logit, latent_loss, vq_loss, commit_loss  = net(inputs, opt.num_sample)
        # logit, logit_ori, latent_loss1, latent_loss2, vq_loss, commit_loss  = net(inputs, opt.num_avg)
        class_loss = F.cross_entropy(logit, targets).div(math.log(2))
        # class_loss_ori = F.cross_entropy(logit_ori, targets).div(math.log(2))
        info_loss = 0.5 * latent_loss.mean().div(math.log(2))
        qt_loss = opt.vq_coef * vq_loss.mean() + opt.comit_coef * commit_loss.mean()
        total_loss = class_loss + opt.va_beta * info_loss + qt_loss
        return total_loss, logit  # only train the student network with gt labels?

    def h_test(sample):  # network feed into engine should have 'loss, output'
        inputs = sample[0].detach()
        targets = sample[1]

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.to(inputs.device)

        inputs, targets = Variable(inputs), Variable(targets)

        net.eval()
        # logit, logit_ori, latent_loss1, latent_loss2, vq_loss, commit_loss  = net(inputs, opt.num_sample)
        logit, latent_loss, vq_loss, commit_loss  = net(inputs, opt.num_avg)
        class_loss = F.cross_entropy(logit, targets).div(math.log(2))
        # class_loss_ori = F.cross_entropy(logit_ori, targets).div(math.log(2))
        info_loss = 0.5 * latent_loss.mean().div(math.log(2))
        qt_loss = opt.vq_coef * vq_loss.mean() + opt.comit_coef * commit_loss.mean()
        total_loss = class_loss + opt.va_beta * info_loss + qt_loss
        return total_loss, logit  # only train the student network with gt labels?

    def log(t, state):
        model_states = {
            'net': net.state_dict(),
        }
        optim_states = {
            'optim': state['optimizer'].state_dict(),
        }


        states = {
            # 'iter': self.global_iter,
            'epoch': t['epoch'],
            # 'history': self.history,
            'args': opt,  # self.args
            'model_states': model_states,
            'optim_states': optim_states,
            'test_acc': t['test_acc'],
            'test_loss': t['test_loss']
        }

        save_path = os.path.join(opt.ckpt_dir, opt.env_name, opt.filename)
        torch.save(states, save_path)

        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(ckpt_dir, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])
        state['iter_num'] += 1
        state['iter_num_epoch'] += 1

    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(state['loss'].item())
        if state['train']:
            # timer_train.add()
            timer_train.add(opt.batch_size)
        else:
            # timer_test.add()
            timer_test.add(opt.batch_size)

    def on_start(state):
        state['epoch'] = epoch
        if opt.resume != '':
            state['test_acc'] = checkpoint['test_acc']
        else:
            state['test_acc'] = 0.0

        if state['train']:
            state['optimizer'] = optimizer
            if opt.lr==0.01:
                lr_lambda = lambda epoch: 1.0 if epoch==1 else 10.0*np.power(0.5, int(epoch / 25)) # e2 lr=0.01***
            elif opt.lr==0.001:
                lr_lambda = lambda epoch: 0.1*np.power(10, epoch) if epoch<3 else 100.0*np.power(0.5, int(epoch / 25)) # e2e3 lr=0.001***
            else:
                lr_lambda = lambda epoch: np.power(0.5, int(epoch / 25)) # ***
            state['scheduler'] = lr_scheduler.LambdaLR(state['optimizer'], lr_lambda=lr_lambda)

        state['iter_num'] = 0
        summary_dir = os.path.join(opt.summary_dir, opt.env_name)
        if not os.path.exists(summary_dir): os.makedirs(summary_dir)
        state['tf'] = SummaryWriter(log_dir=summary_dir)
        state['tf'].add_text(tag='argument',text_string=str(opt),global_step=state['iter_num'])
        state['iter_num_epoch'] = 0

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]

        state['scheduler'].step()
        print("epoch %d lr %f, emb lr %f" % (state['epoch'], state['optimizer'].param_groups[0]['lr'],
                                                state['optimizer'].param_groups[1]['lr']))

        state['iterator'] = tqdm(train_loader)
        state['iter_num_epoch'] = 0

    def on_end_epoch(state):
        train_loss = meter_loss.mean
        # state['train_loss'] = train_loss
        train_acc = classacc.value()
        train_time = timer_train.value()
        train_at_losses = [m.value() for m in meters_at]
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()
        [m.reset() for m in meters_at]
        state['iter_num_epoch'] = 0

        engine.test(h_test, test_loader)
        test_acc = classacc.value()[0]
        if test_acc > state['test_acc']:
            state['test_acc'] = test_acc
            print(log({
                "train_loss": train_loss,
                "train_acc": train_acc[0],
                "test_loss": meter_loss.mean,
                "test_acc": test_acc,
                "test_acc_top3": classacc.value()[1],
                "test_acc_top5": classacc.value()[2],
                "epoch": state['epoch'],
                "num_classes": num_classes,
                "n_parameters": n_parameters,
                "train_time": train_time,
                "test_time": timer_test.value(),
                # "at_losses_train": train_at_losses,
                # "at_losses_test": [m.value() for m in meters_at],
               }, state))
            print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                           (ckpt_dir, state['epoch'], opt.epoch, test_acc))
            if hasattr(net, 'emb'):
                print(net.emb.weight.data)
            else:
                print(net.module.emb.weight.data)

    def on_update(state):
        state['iterator'].set_description('[{epoch}]({batch}/{size})  Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    epoch=state['epoch'],
                    batch=state['iter_num_epoch'],
                    size=len(state['iterator']),
                    loss=meter_loss.mean,
                    top1=classacc.value()[0],
                    top5=classacc.value()[2],
                    ))
        if opt.tensorboard:
            if state['iter_num'] % 50 == 0:
                # state['tf'].add_scalar('Train/loss', train_loss, state['iter_num'])
                # pdb.set_trace()

                if torch.cuda.device_count()>1:
                    state['tf'].add_scalars(main_tag='Train/grad',
                                            tag_scalar_dict={
                                                'emb': net.module.emb.weight.grad.abs().mean().item(),
                                                'encode': net.module.encode[-1].weight.grad.abs().mean().item(),
                                                'decode': net.module.decode[-1].weight.grad.abs().mean().item(),
                                                'att_mu': net.module.att_module_mu[0].weight.grad.abs().mean().item(),
                                                'att_std': net.module.att_module_std[0].weight.grad.abs().mean().item(),
                                            }, global_step=state['iter_num'])
                    state['tf'].add_scalars(main_tag='Train/loss',
                                            tag_scalar_dict={
                                                'loss': meter_loss.mean, # state['train_loss'],
                                            }, global_step=state['iter_num'])
                else:
                    state['tf'].add_scalars(main_tag='Train/grad',
                                            tag_scalar_dict={
                                                'emb': net.emb.weight.grad.abs().mean().item(),
                                                'encode': net.encode[-1].weight.grad.abs().mean().item(),
                                                'decode': net.decode[-1].weight.grad.abs().mean().item(),
                                                'att_mu': net.att_module_mu[0].weight.grad.abs().mean().item(),
                                                'att_std': net.att_module_std[0].weight.grad.abs().mean().item(),
                                            }, global_step=state['iter_num'])
                    state['tf'].add_scalars(main_tag='Train/loss',
                                            tag_scalar_dict={
                                                'loss': meter_loss.mean, # state['train_loss'],
                                            }, global_step=state['iter_num'])


    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.hooks['on_update'] = on_update
    engine.train(h_train, train_loader, opt.epoch, optimizer)
    # engine.test(h_test, test_loader)
    # test_acc1, test_acc3, test_acc5  = classacc.value()
    # print('==> id: %s (%d/%d), test_acc1: \33[91m%.2f\033[0m, test_acc3: \33[91m%.2f\033[0m, test_acc5: \33[91m%.2f\033[0m' % \
    #                        (ckpt_dir, epoch, opt.epoch, test_acc1, test_acc3, test_acc5 ))

def save_attention(opt):
    print('parsed options:', vars(opt))

    if '100' in opt.dataset or 'CF100_' in opt.dataset:
        num_classes = 100
    else:
        num_classes = 10

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    cuda_available = torch.cuda.is_available()
    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('main:', random.random())


    att_dir = Path(opt.att_dir).joinpath(opt.env_name)

    CFVaNet = Vgg16_bn_lrn_VANet
    ImgSize = CIFAR_RESIZE

    print('Model init %s' % ModelInit[opt.model_init])
    net = CFVaNet(opt.K, opt.att_K, opt.return_att, num_classes, init=ModelInit[opt.model_init])

    params = net.state_dict()
    n_parameters = sum(p.numel() for p in list(params.values()))
    print('\nTotal number of parameters:', n_parameters)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

    AttSize = [net.att_dim, net.att_dim]

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    model_data = torch.load(os.path.join(opt.ckpt_dir, opt.env_name, opt.filename))  # 'best_acc.tar'
    params = model_data['model_states']['net']
    pdb.set_trace()
    net.load_state_dict(params)

    net.eval()

    # ------- load data ----------
    if 'CIFAR' in opt.dataset:
        va_trans = T.Compose([
            T.Normalize(mean=[-125.3 / 63.0, -123.0 / 62.1, -113.9 / 66.7],
                        std=[255.0 / 63.0, 255.0 / 62.1, 255.0 / 66.7]),
            T.ToPILImage()])
    else:
        va_trans = T.Compose([
            T.Normalize(mean=[-1.0, -1.0, -1.0],
                        std=[1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5]),
            T.ToPILImage()])

    mode=False
    test_loader = DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())


    cnt = 0
    if not att_dir.exists(): att_dir.mkdir(parents=True, exist_ok=True)

    for idx, (images, labels) in enumerate(test_loader):

        x = Variable(utils_my.cuda(images, cuda_available))
        y = Variable(utils_my.cuda(labels, cuda_available))
        # y_one_hot = F.one_hot(y, n_class).type_as(x)
        pred_logits, att_maps = net(x, opt.num_sample, train=False)
        _, predicted = torch.max(pred_logits.data, 1)
        flag = (predicted == y)
        # _, predicted_ori = torch.max(pred_logits_ori.data, 1)
        # flag_ori = (predicted_ori == y)

        # att_maps = att_maps.view(-1, self.batch_size, att_maps.size(2)) # wrong!
        att_maps_list = []
        for d_i in range(torch.cuda.device_count()):
            att_maps_list.append(att_maps[d_i * opt.num_sample:(d_i + 1) * opt.num_sample, :, :])
        att_maps = torch.cat(att_maps_list, dim=1)


        for b_num in range(x.size(0)):
            cnt = cnt + 1
            if cnt > va_num:
                break

            ori_img = va_trans(images[b_num])
            # scipy.misc.imsave(os.path.join(out_folder, 'test_{:04d}_{}.jpg'.format(cnt, flag[b_num])),
            #                  ori_img.detach().cpu().numpy())
            ori_img.save(os.path.join(att_dir,
                                      'test_{:04d}_cls{:02d}_{}.jpg'.format(cnt, labels[b_num], flag[b_num])))

            # if return att_mask.mean(0)
            # att_map = att_maps[b_num].view(MNIST_RESIZE).detach().cpu().numpy()
            # tmp_map = postprocess_prediction(att_map)
            # scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_{}_att.png'.format(
            #     cnt, flag[b_num])), tmp_map)
            # pdb.set_trace()

            # if return att_mask
            for a_num in range(att_maps.size(0)):
                # for a_num in range(att_maps.size(0)//torch.cuda.device_count()):
                att_map = att_maps[a_num, b_num].view(AttSize).detach().cpu().numpy()
                tmp_map = utils_my.postprocess_prediction(att_map, size=ImgSize)
                scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_cls{:02d}_{}_att{:02d}.png'.format(
                    cnt, labels[b_num], flag[b_num], a_num)), tmp_map)

        if cnt > va_num:
            break

def save_attention_qt(opt):
    print('parsed options:', vars(opt))

    if '100' in opt.dataset:
        num_classes = 100
    else:
        num_classes = 10


    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    cuda_available = torch.cuda.is_available()

    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('main:', random.random())

    att_dir = Path(opt.att_dir).joinpath(opt.env_name)

    CFVaNet = Vgg16_bn_lrn_VANet_QT
    ImgSize = CIFAR_RESIZE

    print('Model init %s' % ModelInit[opt.model_init])
    net = CFVaNet(opt.K, opt.att_K, opt.qt_num, opt.vq_coef, opt.comit_coef,
                    opt.rd_init, opt.decay, opt.qt_trainable, opt.return_att,
                    n_cls=num_classes, init=ModelInit[opt.model_init])
    AttSize = [net.att_dim, net.att_dim]

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    model_data = torch.load(os.path.join(opt.ckpt_dir, opt.env_name, opt.filename))  # 'best_acc.tar'
    params = model_data['model_states']['net']
    net.load_state_dict(params)
    if 'module.emb.weight' in params.keys():
        print(params['module.emb.weight'].data)
    else:
        print(params['emb.weight'].data)

    net.eval()

    # ------- load data ----------
    if 'CIFAR' in opt.dataset:
        va_trans = T.Compose([
            T.Normalize(mean=[-125.3 / 63.0, -123.0 / 62.1, -113.9 / 66.7],
                        std=[255.0 / 63.0, 255.0 / 62.1, 255.0 / 66.7]),
            T.ToPILImage()])
    else:
        va_trans = T.Compose([
            T.Normalize(mean=[-1.0, -1.0, -1.0],
                        std=[1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5]),
            T.ToPILImage()])

    mode = False
    test_loader = DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    cnt = 0
    if not att_dir.exists(): att_dir.mkdir(parents=True, exist_ok=True)

    for idx, (images, labels) in enumerate(test_loader):

        x = Variable(utils_my.cuda(images, cuda_available))
        y = Variable(utils_my.cuda(labels, cuda_available))
        pred_logits, att_maps, att_maps_q = net(x, opt.num_sample, train=False)
        _, predicted = torch.max(pred_logits.data, 1)
        flag = (predicted == y)

        att_maps_list = []
        for d_i in range(torch.cuda.device_count()):
            att_maps_list.append(att_maps[d_i * opt.num_sample:(d_i + 1) * opt.num_sample, :, :])
        att_maps = torch.cat(att_maps_list, dim=1)

        att_maps_q_list = []
        for d_i in range(torch.cuda.device_count()):
            att_maps_q_list.append(att_maps_q[d_i * opt.num_sample:(d_i + 1) * opt.num_sample, :, :])
        att_maps_q = torch.cat(att_maps_q_list, dim=1)

        for b_num in range(x.size(0)):
            cnt = cnt + 1
            if cnt > va_num:
                break

            ori_img = va_trans(images[b_num])
            # scipy.misc.imsave(os.path.join(out_folder, 'test_{:04d}_{}.jpg'.format(cnt, flag[b_num])),
            #                  ori_img.detach().cpu().numpy())
            ori_img.save(os.path.join(att_dir,
                                      'test_{:04d}_cls{:02d}_{}.jpg'.format(cnt, labels[b_num], flag[b_num])))

            # if return att_mask.mean(0)
            # att_map = att_maps[b_num].view(MNIST_RESIZE).detach().cpu().numpy()
            # tmp_map = postprocess_prediction(att_map)
            # scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_{}_att.png'.format(
            #     cnt, flag[b_num])), tmp_map)
            # pdb.set_trace()

            # if return att_mask
            for a_num in range(att_maps.size(0)):
                # for a_num in range(att_maps.size(0)//torch.cuda.device_count()):
                att_map = att_maps[a_num, b_num].view(AttSize).detach().cpu().numpy()
                tmp_map = utils_my.postprocess_prediction(att_map, size=ImgSize)
                scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_cls{:02d}_{}_att{:02d}.png'.format(
                    cnt, labels[b_num], flag[b_num], a_num)), tmp_map)

            # if return att_mask_q
            for a_num in range(att_maps_q.size(0)):
                att_map = att_maps_q[a_num, b_num].view(AttSize).detach().cpu().numpy()
                # tmp_map = postprocess_prediction(att_map, size=ImgSize)
                tmp_map = cv2.resize(att_map, (ImgSize[1], ImgSize[0]), interpolation=cv2.INTER_NEAREST)
                print('max %.4f min %.4f' % (np.max(tmp_map), np.min(tmp_map)))
                scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_cls{:02d}_{}_att{:02d}_q.png'.format(
                    cnt, labels[b_num], flag[b_num], a_num)), tmp_map.astype('float') * 255.)

            # # if return att_mask_emb
            # for a_num in range(att_maps_emb.size(0)):
            #     att_map = att_maps_emb[a_num, b_num].view(AttSize).detach().cpu().numpy()
            #     # tmp_map = postprocess_prediction(att_map, size=ImgSize)
            #     tmp_map = cv2.resize(att_map, (ImgSize[1], ImgSize[0]), interpolation=cv2.INTER_NEAREST)
            #     # print('max %.4f min %.4f' % (np.max(tmp_map), np.min(tmp_map)))
            #     scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_cls{:02d}_{}_{}_att{:02d}_emb.png'.format(
            #         cnt, labels[b_num], flag[b_num], flag_ori[b_num], a_num)), tmp_map.astype('float') * 255.)

        if cnt > va_num:
            break


if __name__ == '__main__':
    opt = parser.parse_args()
    if opt.phase == 'train':
        main(opt)
    elif opt.phase == 'train_qt':
        main_qt(opt)
    elif opt.phase == 'save_att':
        save_attention(opt)
    elif opt.phase == 'save_att_qt':
        save_attention_qt(opt)
