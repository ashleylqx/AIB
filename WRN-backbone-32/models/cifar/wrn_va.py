import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils_my
from utils import utils_my
from utils.config import *
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
from utils.sonnet_vqvae_torch import VectorQuantizer
from numbers import Number

__all__ = ['wrn_va', 'wrn_va_mu_m1', 'wrn_va_g3', 'wrn_va_g3_mu_m1', 'wrn_va_g1', 'wrn_va_qt']

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

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

# g2 s1 d1 version
class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        self.bn_att = nn.BatchNorm2d(nChannels[3])
        self.att_conv   = nn.Conv2d(nChannels[3], num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        self.att_module_mu = nn.Sequential(
            self.att_block3,
            self.bn_att,
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        ) # version before 20210102
        # self.att_module_mu = nn.Sequential(
        #     self.att_block3,
        #     self.bn_att,
        #     self.relu, # add one more relu layer here
        #     self.att_conv,
        #     self.bn_att2,
        #     self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.K = K
        self.encode = nn.Sequential(
            self.block3,
            self.bn1,
            self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        # self.decode = nn.Sequential(
        #     nn.Linear(self.K, self.K),
        #     nn.ReLU(True),
        #     nn.Linear(self.K, num_classes))

        # self.decode = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.Linear(self.K, self.K),
        #     nn.ReLU(True),
        #     nn.Linear(self.K, num_classes))  # v2

        self.decode = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(self.K, num_classes)) # v3

        # self.decode = nn.Sequential(
        #     nn.Linear(self.K, num_classes)) # v4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

class WideResNet_fcn(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet_fcn, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        self.bn_att = nn.BatchNorm2d(nChannels[3])
        self.att_conv   = nn.Conv2d(nChannels[3], num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        self.att_module_mu = nn.Sequential(
            self.att_block3,
            self.bn_att,
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        ) # version before 20210102
        # self.att_module_mu = nn.Sequential(
        #     self.att_block3,
        #     self.bn_att,
        #     self.relu, # add one more relu layer here
        #     self.att_conv,
        #     self.bn_att2,
        #     self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        fc_num = 7
        self.nChannels = nChannels[3] * fc_num * fc_num

        self.K = K
        self.encode = nn.Sequential(
            self.block3,
            self.bn1,
            self.relu,
            nn.AdaptiveAvgPool2d((fc_num, fc_num)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        # self.decode = nn.Sequential(
        #     nn.Linear(self.K, self.K),
        #     nn.ReLU(True),
        #     nn.Linear(self.K, num_classes))

        # self.decode = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.Linear(self.K, self.K),
        #     nn.ReLU(True),
        #     nn.Linear(self.K, num_classes))  # v2

        self.decode = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(self.K, num_classes)) # v3

        # self.decode = nn.Sequential(
        #     nn.Linear(self.K, num_classes)) # v4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

class WideResNet_mu_m1(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet_mu_m1, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        # channel_num = nChannels[3] # _mu_m1
        channel_num = 100 # _mu_m1_cn100
        # channel_num = nChannels[2] # _mu_m1_2
        # channel_num = nChannels[1] # _mu_m1_1
        # channel_num = nChannels[0] # _mu_m1_0
        self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        self.bn_att = nn.BatchNorm2d(nChannels[3])
        self.att_conv   = nn.Conv2d(nChannels[3], channel_num, kernel_size=1, padding=0, bias=False) # modify here
        self.bn_att2 = nn.BatchNorm2d(channel_num)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(channel_num, 1, kernel_size=3, padding=1, bias=False) # similar to mu_m1 setting
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        # self.att_module_mu = nn.Sequential(
        #     self.att_block3,
        #     self.bn_att,
        #     self.att_conv,
        #     self.bn_att2,
        #     self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # ) # version before 20210102

        # _mu_m1
        self.att_module_mu = nn.Sequential(
            self.att_block3,
            self.bn_att,
            # self.relu, # add one more relu layer here
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        )
        # _mu_m0
        # self.att_module_mu = nn.Sequential(
        #     self.att_block3,
        #     self.bn_att,
        #     self.relu, # add one more relu layer here
        #     # self.att_conv,
        #     # self.bn_att2,
        #     # self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # )
        # _noatbl
        # self.att_module_mu = nn.Sequential(
        #     # self.att_block3,
        #     # self.bn_att,
        #     # self.relu, # add one more relu layer here
        #     self.att_conv,
        #     self.bn_att2,
        #     self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.K = K
        self.encode = nn.Sequential(
            self.block3,
            self.bn1,
            self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

class WideResNet_mu_m0(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet_mu_m0, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        channel_num = nChannels[3] # _mu_m1
        # channel_num = nChannels[2] # _mu_m1_2
        # channel_num = nChannels[1] # _mu_m1_1
        # channel_num = nChannels[0] # _mu_m1_0
        self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        self.bn_att = nn.BatchNorm2d(nChannels[3])
        # self.att_conv   = nn.Conv2d(nChannels[3], channel_num, kernel_size=1, padding=0, bias=False) # modify here
        # self.bn_att2 = nn.BatchNorm2d(channel_num)
        # # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(channel_num, 1, kernel_size=3, padding=1, bias=False) # similar to mu_m1 setting
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        # _mu_m0
        self.att_module_mu = nn.Sequential(
            self.att_block3,
            self.bn_att,
            self.relu, # add one more relu layer here
            # self.att_conv,
            # self.bn_att2,
            # self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        )

        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.K = K
        self.encode = nn.Sequential(
            self.block3,
            self.bn1,
            self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

class WideResNet_noatbl(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet_noatbl, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        # channel_num = nChannels[3] # _mu_m1
        channel_num = nChannels[2] # _mu_m1_2
        # channel_num = nChannels[1] # _mu_m1_1
        # channel_num = nChannels[0] # _mu_m1_0
        # self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        # self.bn_att = nn.BatchNorm2d(nChannels[3])
        self.att_conv   = nn.Conv2d(nChannels[2], channel_num, kernel_size=1, padding=0, bias=False) # modify here
        self.bn_att2 = nn.BatchNorm2d(channel_num)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(channel_num, 1, kernel_size=3, padding=1, bias=False) # similar to mu_m1 setting
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        # _noatbl
        self.att_module_mu = nn.Sequential(
            # self.att_block3,
            # self.bn_att,
            # self.relu, # add one more relu layer here
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.K = K
        self.encode = nn.Sequential(
            self.block3,
            self.bn1,
            self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

class WideResNet_g3(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet_g3, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        # self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        # self.bn_att = nn.BatchNorm2d(nChannels[3])
        self.att_conv   = nn.Conv2d(nChannels[3], num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        # self.att_module_mu = nn.Sequential(
        #     self.att_block3,
        #     self.bn_att,
        #     self.att_conv,
        #     self.bn_att2,
        #     self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # ) # version before 20210102
        self.att_module_mu = nn.Sequential(
            # self.att_block3,
            # self.bn_att,
            # self.relu, # add one more relu layer here
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.K = K
        self.encode = nn.Sequential(
            # self.block3,
            # self.bn1,
            # self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        out = self.block3(out) # out.size [64, 640, 16, 16]
        out = self.relu(self.bn1(out))  # [64, 640, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

class WideResNet_g3_bn(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet_g3_bn, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        # self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        self.bn_att = nn.BatchNorm2d(nChannels[3])
        self.att_conv   = nn.Conv2d(nChannels[3], num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        # self.att_module_mu = nn.Sequential(
        #     self.att_block3,
        #     self.bn_att,
        #     self.att_conv,
        #     self.bn_att2,
        #     self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # ) # version before 20210102
        self.att_module_mu = nn.Sequential(
            # self.att_block3,
            self.bn_att,
            self.relu, # add one more relu layer here
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.K = K
        self.encode = nn.Sequential(
            # self.block3,
            self.bn1,
            self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        out = self.block3(out) # out.size [64, 640, 16, 16]
        # out = self.relu(self.bn1(out))  # [64, 640, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

class WideResNet_g3_bl4(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet_g3_bl4, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        # 4th block
        self.block4 = NetworkBlock(n, nChannels[3], nChannels[2], block, 2, dropRate)
        # global average pooling and classifier
        # self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.bn1 = nn.BatchNorm2d(nChannels[2])
        self.relu = nn.ReLU(inplace=True)

        # self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        self.bn_att = nn.BatchNorm2d(nChannels[3])
        self.att_conv   = nn.Conv2d(nChannels[3], num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        # self.att_module_mu = nn.Sequential(
        #     self.att_block3,
        #     self.bn_att,
        #     self.att_conv,
        #     self.bn_att2,
        #     self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # ) # version before 20210102
        self.att_module_mu = nn.Sequential(
            # self.att_block3,
            self.bn_att,
            self.relu, # add one more relu layer here
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[2]

        self.K = K
        self.encode = nn.Sequential(
            # self.block3,
            # self.bn1,
            # self.relu,
            self.block4,
            self.bn1,
            self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        out = self.block3(out) # out.size [64, 640, 16, 16]
        # out = self.relu(self.bn1(out))  # [64, 640, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

class WideResNet_g3_atbl4(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet_g3_atbl4, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.nChannels = nChannels[2]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        # 4th block
        self.block4 = NetworkBlock(n, nChannels[3], self.nChannels, block, 2, dropRate)
        # global average pooling and classifier
        # self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.bn1 = nn.BatchNorm2d(self.nChannels)
        self.relu = nn.ReLU(inplace=True)

        # self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        self.att_block4 = NetworkBlock(n, nChannels[3], self.nChannels, block, 1, dropRate)
        self.bn_att = nn.BatchNorm2d(self.nChannels)
        self.att_conv   = nn.Conv2d(self.nChannels, num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        # self.att_module_mu = nn.Sequential(
        #     self.att_block3,
        #     self.bn_att,
        #     self.att_conv,
        #     self.bn_att2,
        #     self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # ) # version before 20210102
        self.att_module_mu = nn.Sequential(
            self.att_block4,
            self.bn_att,
            self.relu, # add one more relu layer here
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        # self.nChannels = nChannels[3]

        self.K = K
        self.encode = nn.Sequential(
            # self.block3,
            # self.bn1,
            # self.relu,
            self.block4,
            self.bn1,
            self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        out = self.block3(out) # out.size [64, 640, 16, 16]
        # out = self.relu(self.bn1(out))  # [64, 640, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

class WideResNet_g3_mu_m1(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet_g3_mu_m1, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        # self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        # self.bn_att = nn.BatchNorm2d(nChannels[3])
        self.att_conv   = nn.Conv2d(nChannels[3], nChannels[3], kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(nChannels[3])
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(nChannels[3], 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        # self.att_module_mu = nn.Sequential(
        #     self.att_block3,
        #     self.bn_att,
        #     self.att_conv,
        #     self.bn_att2,
        #     self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # ) # version before 20210102
        self.att_module_mu = nn.Sequential(
            # self.att_block3,
            # self.bn_att,
            # self.relu, # add one more relu layer here
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.K = K
        self.encode = nn.Sequential(
            # self.block3,
            # self.bn1,
            # self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        out = self.block3(out) # out.size [64, 640, 16, 16]
        out = self.relu(self.bn1(out))  # [64, 640, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

class WideResNet_g1(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
        super(WideResNet_g1, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        self.att_block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        self.bn_att = nn.BatchNorm2d(nChannels[3])
        self.att_conv   = nn.Conv2d(nChannels[3], num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        # self.att_module_mu = nn.Sequential(
        #     self.att_block3,
        #     self.bn_att,
        #     self.att_conv,
        #     self.bn_att2,
        #     self.relu,
        #     self.att_conv3,
        #     self.bn_att3,
        #     self.sigmoid
        # ) # version before 20210102
        self.att_module_mu = nn.Sequential(
            self.att_block2,
            self.att_block3,
            self.bn_att,
            self.relu, # add one more relu layer here
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.K = K
        self.encode = nn.Sequential(
            self.block2,
            self.block3,
            self.bn1,
            self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        # out = self.block2(out) # out.size [64, 320, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx
        # pdb.set_trace()
        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, att_mask # for test

            return logit, latent_loss # for train

        else:
            return logit, att_mask  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

# class WideResNet_g0(nn.Module):
#     def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim):
#         super(WideResNet_g0, self).__init__()
#         nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
#         assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
#         n = (depth - 4) // 6
#         block = BasicBlock
#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         # 1st block
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         # 2nd block
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         # 3rd block
#         self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(nChannels[3])
#         self.relu = nn.ReLU(inplace=True)
#
#         self.att_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         self.att_block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
#         self.bn_att = nn.BatchNorm2d(nChannels[3])
#         self.att_conv   = nn.Conv2d(nChannels[3], num_classes, kernel_size=1, padding=0, bias=False)
#         self.bn_att2 = nn.BatchNorm2d(num_classes)
#         # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
#         self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
#         self.bn_att3 = nn.BatchNorm2d(1)
#         # self.att_gap = nn.AvgPool2d(16)
#         self.sigmoid = nn.Sigmoid()
#
#         # self.att_module_mu = nn.Sequential(
#         #     self.att_block3,
#         #     self.bn_att,
#         #     self.att_conv,
#         #     self.bn_att2,
#         #     self.relu,
#         #     self.att_conv3,
#         #     self.bn_att3,
#         #     self.sigmoid
#         # ) # version before 20210102
#         self.att_module_mu = nn.Sequential(
#             self.att_block1,
#             self.att_block2,
#             self.att_block3,
#             self.bn_att,
#             self.relu, # add one more relu layer here
#             self.att_conv,
#             self.bn_att2,
#             self.relu,
#             self.att_conv3,
#             self.bn_att3,
#             self.sigmoid
#         )
#         self.att_dim = 16
#         self.att_module_std = nn.Sequential(
#             nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
#         )
#
#
#         # self.fc = nn.Linear(nChannels[3], num_classes)
#         self.nChannels = nChannels[3]
#
#         self.K = K
#         self.encode = nn.Sequential(
#             self.block1,
#             self.block2,
#             self.block3,
#             self.bn1,
#             self.relu,
#             nn.AdaptiveAvgPool2d((1, 1)),
#             Flatten(),
#             nn.Linear(self.nChannels, 2 * self.K))  #
#
#         self.decode = nn.Sequential(
#             nn.Linear(self.K, self.K),
#             nn.ReLU(True),
#             nn.Linear(self.K, num_classes))
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#
#     def forward(self, x, num_sample=1, train=True, return_all=False):
#         out = self.conv1(x) # x.size [64, 3, 32, 32]
#         # out = self.block1(out)
#         # out = self.block2(out) # out.size [64, 320, 16, 16]
#         # pdb.set_trace()
#         # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
#         # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
#         # bs, cs, ys, xs = ax.shape
#         # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
#         # # self.att = self.att.view(bs, 1, ys, xs)
#         # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
#         # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
#         # # ax = ax.view(ax.size(0), -1) # [64, 200]
#
#         att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
#         att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
#         att_mu = att_mu.view(x.size(0), -1)
#         att_std = att_std.view(x.size(0), -1)  # conv2
#         att_std = F.softplus(att_std - 5, beta=1)
#
#         att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
#         att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)
#
#         if att_mode in ['sigmoid', 'sig_sft']:
#             att_mask = torch.sigmoid(att_mask)
#         if att_mode in ['softmax', 'sig_sft']:
#             att_mask = F.softmax(att_mask, dim=-1)
#         if att_mode in ['relu']:
#             att_mask = F.relu(att_mask)
#
#         att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
#         att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
#         att_mask_reshape = F.interpolate(att_mask_reshape, size=[out.size(2), out.size(3)])
#
#         if num_sample > 1:
#             out_rpt = out.repeat(num_sample, 1, 1, 1)
#             rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
#             rx = out_rpt + rx
#         else:
#             rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
#             rx = out + rx
#         # pdb.set_trace()
#         statistics = self.encode(rx)
#         mu = statistics[:, :self.K]
#         std = F.softplus(statistics[:, self.K:] - 5, beta=1)
#
#         encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
#         encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
#         logit = self.decode(encoding)
#
#         logit = logit.mean(0).mean(0)
#
#         if train:
#             # -----------------
#             normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
#             latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
#             latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))
#
#             if return_all:
#                 return logit, latent_loss, att_mask # for test
#
#             return logit, latent_loss # for train
#
#         else:
#             return logit, att_mask  # for save_attention
#
#         # # rx = out * self.att # [64, 320, 16, 16]
#         # # rx = rx + out
#         # rx = self.block3(rx) # [64, 640, 8, 8]
#         # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
#         # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
#         # rx = rx.view(-1, self.nChannels) # [64, 640]
#         # rx = self.fc(rx) # [64, 200]
#
#         # return ax, rx, self.att
#
#     def reparametrize_n(self, mu, std, n=1):
#         # reference :
#         # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
#         def expand(v):
#             if isinstance(v, Number):
#                 return torch.Tensor([v]).expand(n, 1)
#             else:
#                 return v.expand(n, *v.size())
#
#         if n != 1 :
#             mu = expand(mu)
#             std = expand(std)
#
#         eps = Variable(utils_my.cuda(std.data.new(std.size()).normal_(), std.is_cuda))
#
#         return mu + eps * std
#
#     def kl_divergence(self, latent_space1, latent_space2):
#         kl_div = kl.kl_divergence(latent_space1, latent_space2)
#         return kl_div


class WideResNet_QT(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=K_dim,
                 qt_num=2, vq_coef=0.2, comit_coef=0.4, rd_init=False, decay=None, qt_trainable=False):
        super(WideResNet_QT, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        self.att_block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        self.bn_att = nn.BatchNorm2d(nChannels[3])
        self.att_conv   = nn.Conv2d(nChannels[3], num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        self.att_module_mu = nn.Sequential(
            self.att_block3,
            self.bn_att,
            self.att_conv,
            self.bn_att2,
            self.relu,
            self.att_conv3,
            self.bn_att3,
            self.sigmoid
        )
        self.att_dim = 16
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # conv2
        )


        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.K = K
        self.encode = nn.Sequential(
            self.block3,
            self.bn1,
            self.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.nChannels, 2 * self.K))  #

        # self.decode = nn.Sequential(
        #     nn.Linear(self.K, self.K),
        #     nn.ReLU(True),
        #     nn.Linear(self.K, num_classes))

        # self.decode = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.Linear(self.K, self.K),
        #     nn.ReLU(True),
        #     nn.Linear(self.K, num_classes))  # v2

        self.decode = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(self.K, num_classes))  # v3

        # self.decode = nn.Sequential(
        #     nn.Linear(self.K, num_classes)) # v4

        self.qt_num = qt_num
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.rd_init = rd_init
        self.decay = decay
        self.qt_trainable = qt_trainable

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, num_sample=1, train=True, return_all=False):
        out = self.conv1(x) # x.size [64, 3, 32, 32]
        out = self.block1(out)
        out = self.block2(out) # out.size [64, 320, 16, 16]
        # pdb.set_trace()
        # ax = self.bn_att(self.att_block3(out)) # ax.size [64, 640, 16, 16]
        # ax = self.relu(self.bn_att2(self.att_conv(ax))) # ax.size [64, 200, 16, 16]
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax))) # self.att.size [64, 1, 16, 16]
        # # self.att = self.att.view(bs, 1, ys, xs)
        # # ax = self.att_conv2(ax) # [64, 200, 16, 16]
        # # ax = self.att_gap(ax) # [64, 200, 1, 1] *** not 1, 1 so cause errors?
        # # ax = ax.view(ax.size(0), -1) # [64, 200]

        att_mu = self.att_module_mu(out) # [64, 1, 16, 16]
        att_std = self.att_module_std(att_mu)  # [64, 1, 16, 16]
        att_mu = att_mu.view(x.size(0), -1)
        att_std = att_std.view(x.size(0), -1)  # conv2
        att_std = F.softplus(att_std - 5, beta=1)

        att_mask = self.reparametrize_n(att_mu, att_std, num_sample)  # (num_sample, bs, self.att_K)
        att_mask = att_mask.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if att_mode in ['sigmoid', 'sig_sft']:
            att_mask = torch.sigmoid(att_mask)
        if att_mode in ['softmax', 'sig_sft']:
            att_mask = F.softmax(att_mask, dim=-1)
        if att_mode in ['relu']:
            att_mask = F.relu(att_mask)

        att_mask_flt = att_mask.view(num_sample * x.size(0), self.att_dim * self.att_dim)
        att_mask_q, vq_loss, commit_loss = self.emb(att_mask_flt)
        att_mask_reshape = att_mask_q.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
        att_mask_q = att_mask_q.view(num_sample, x.size(0), self.att_dim * self.att_dim)

        if num_sample > 1:
            out_rpt = out.repeat(num_sample, 1, 1, 1)
            rx = torch.mul(out_rpt, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out_rpt + rx
        else:
            rx = torch.mul(out, att_mask_reshape)  # (num_sample, bs, h, w)-->(num_sample*bs, h, w)
            rx = out + rx

        statistics = self.encode(rx)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)  # (num_sample, num_sample*bs, self.K)
        encoding = encoding.view(num_sample, num_sample, x.size(0), self.K)  # (num_sample*num_sample*bs, self.K)
        logit = self.decode(encoding)

        logit = logit.mean(0).mean(0)

        if train:
            # -----------------
            normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
            latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
            latent_loss = torch.mean(self.kl_divergence(latent_prior, normal_prior))

            if return_all:
                return logit, latent_loss, vq_loss, commit_loss, att_mask, att_mask_q # for test

            return logit, latent_loss, vq_loss, commit_loss # for train

        else:
            return logit, att_mask, att_mask_q  # for save_attention

        # # rx = out * self.att # [64, 320, 16, 16]
        # # rx = rx + out
        # rx = self.block3(rx) # [64, 640, 8, 8]
        # rx = self.relu(self.bn1(rx)) # [64, 640, 8, 8]
        # rx = F.avg_pool2d(rx, 8) # [64, 640, 2, 2]
        # rx = rx.view(-1, self.nChannels) # [64, 640]
        # rx = self.fc(rx) # [64, 200]

        # return ax, rx, self.att

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

def wrn_va(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    # model = WideResNet_mu_m1(**kwargs)
    return model

def wrn_va_mu_m1(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    # model = WideResNet_mu_m1(**kwargs)
    # model = WideResNet_mu_m0(**kwargs)
    # model = WideResNet_noatbl(**kwargs)
    model = WideResNet_fcn(**kwargs)
    return model

def wrn_va_g3(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    # model = WideResNet_g3(**kwargs)
    # model = WideResNet_g3_bl4(**kwargs)
    # model = WideResNet_g3_bn(**kwargs)
    model = WideResNet_g3_atbl4(**kwargs)
    return model

def wrn_va_g3_mu_m1(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet_g3_mu_m1(**kwargs)
    return model

def wrn_va_g1(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet_g1(**kwargs)
    # model = WideResNet_g0(**kwargs)
    return model

def wrn_va_qt(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet_QT(**kwargs)
    return model


