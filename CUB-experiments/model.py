import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import cuda

import time
from numbers import Number
import pdb

from torchvision import models
from torchvision.models.resnet import Bottleneck, conv1x1, BasicBlock

from config import *

from torch.distributions import Normal, Independent, kl

def xavier_init(ms):
    for m in ms:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()


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


class VaNet_vgg(nn.Module):

    def __init__(self, K=K_dim, att_K=A_dim, return_att=False, backbone='vgg'):
        super(VaNet_vgg, self).__init__()
        print('VaNet_vgg')
        self.K = K
        self.att_K = att_K
        self.return_att = return_att
        assert backbone in ['vgg16', 'wrn_50_2'], 'backbone must be vgg16 or wrn_50_2'

        self.backbone = backbone
        ''''''

        '''------- vgg16 ------'''
        blocks = models.vgg16(pretrained=False)

        self.features = blocks.features[:-1]      # f29  vgg16 (512, 6, 6)


        self.channel = 512 * 3 * 3

        self.encode = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),  # the first model
            Flatten(),
            nn.Linear(self.channel, 2 * self.K))  # train0714_CF10


        self.att_dim = 14  # wol for f29

        self.att_module_mu = nn.Sequential(
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # [bs, 512, 6, 6]
            # nn.ReLU(inplace=True),  # [bs, 512, 6, 6] mu_m1
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # [bs, 512, 6, 6]
            # nn.ReLU(inplace=True),  # [bs, 512, 6, 6] mu_m2
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # [bs, 512, 6, 6]
            # nn.ReLU(inplace=True),  # ===================                         # [bs, 512, 6, 6] _mu_m3
            nn.Conv2d(512, 1, 3, stride=1, padding=1),  # f11 original
            nn.Sigmoid()  # sigmu
        )


        '''------ common part ---------'''
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1), #conv2
        )

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, n_class))

    # f4 f7 f11
    def forward(self, x, num_sample=1, train=True):

        f = self.features(x) # for vgg16


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
        # pdb.set_trace()
        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
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

            return logit, logit, latent_loss, latent_loss # repeat to fulfil the format

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
            return logit, logit, att_mask # repeat to fulfil the format

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

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

        if pre_train:
            print('Loading pretrained weights ...')
            ckpt_file = base_path + 'DataSets/GazeFollow/checkpoints/vgg16.pth'

            pretrained_dict = torch.load(ckpt_file)
            model_dict = self.state_dict()
            # pdb.set_trace()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def kl_divergence(self, latent_space1, latent_space2):
        kl_div = kl.kl_divergence(latent_space1, latent_space2)
        return kl_div


class VaNet_wrn(nn.Module):

    def __init__(self, K=K_dim, att_K=A_dim, return_att=False, backbone='vgg'):
        super(VaNet_wrn, self).__init__()
        print('VaNet_wrn')
        self.K = K
        self.att_K = att_K
        self.return_att = return_att
        assert backbone in ['vgg16', 'wrn_50_2'], 'backbone must be vgg16 or wrn_50_2'

        self.backbone = backbone
        ''''''

        '''------- wrn_50_2 ------'''
        blocks = models.wide_resnet50_2(pretrained=False)

        self.conv1 = blocks.conv1
        self.bn1 = blocks.bn1
        self.relu = blocks.relu
        self.maxpool = blocks.maxpool
        self.layer1 = blocks.layer1
        self.layer2 = blocks.layer2
        self.layer3 = blocks.layer3
        self.layer4 = blocks.layer4

        block = Bottleneck
        layers = [3,4,6,3]
        width_per_group = 64 * 2
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = width_per_group
        self.tmp_layer = self._make_layer(block, 64, layers[0])
        self.tmp_layer = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.tmp_layer = self._make_layer(block, 256, layers[2])
        self.att_layer4 = self._make_layer(block, 512, layers[3])


        self.channel = 2048

        self.encode = nn.Sequential(
            # self.layer3,
            self.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),  # the first model
            Flatten(),
            nn.Linear(self.channel, 2 * self.K))  # train0714_CF10


        self.att_dim = 14  # wol for layer2

        self.att_module_mu = nn.Sequential(
            self.att_layer4,
            nn.Conv2d(self.channel, 1, 3, stride=1, padding=1),
            nn.Sigmoid()  # sigmu
        )


        '''------ common part ---------'''
        self.att_module_std = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1), #conv2
        )

        self.decode = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, n_class))

    # f4 f7 f11
    def forward(self, x, num_sample=1, train=True):

        f = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
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
        # pdb.set_trace()
        att_mask_reshape = att_mask.view(-1, self.att_dim * self.att_dim)
        att_mask_reshape = att_mask_reshape.view(-1, self.att_dim, self.att_dim).unsqueeze(1)
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

            return logit, logit, latent_loss, latent_loss # repeat to fulfil the format

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
            return logit, logit, att_mask # repeat to fulfil the format

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

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        zero_init_residual = False
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        if pre_train:
            print('Loading pretrained weights ...')

            ckpt_file = base_path + 'DataSets/GazeFollow/checkpoints/wide_resnet50_2.pth'

            pretrained_dict = torch.load(ckpt_file)
            model_dict = self.state_dict()
            # pdb.set_trace()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def kl_divergence(self, latent_space1, latent_space2):
        kl_div = kl.kl_divergence(latent_space1, latent_space2)
        return kl_div


