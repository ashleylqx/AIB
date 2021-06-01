import os
import cv2
import numpy as np
import pickle
import torch
import argparse
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import torchnet as tnt
from tensorboardX import SummaryWriter
from utils import cuda, Weight_EMA_Update, postprocess_prediction
from datasets import return_data
# from model import VaNet
from model import VaNet_vgg, VaNet_wrn

from tqdm import tqdm
from pathlib import Path
from apex import amp
import pdb

import scipy.misc
from config import *

from torchnet.meter import mAPMeter


# ---- VA -----
class SolverVA(object):

    def __init__(self, args):
        self.args = args

        self.cuda = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.train_batch = args.train_batch * torch.cuda.device_count()
        self.test_batch = args.test_batch * torch.cuda.device_count()
        self.lr = args.lr
        self.eps = 1e-9
        self.K = args.K
        self.beta = args.beta
        self.num_avg = args.num_avg
        self.global_iter = 0
        self.global_epoch = 0

        # Network & Optimizer
        self.att_K = args.att_K
        self.return_att = args.return_att
        self.num_sample = args.num_sample
        self.min_zax = args.min_zax

        self.dataset = args.dataset

        self.backbone = args.backbone
        if self.backbone == 'vgg16':
            VaNet = VaNet_vgg
        else:
            VaNet = VaNet_wrn
        ''''''
        self.model = cuda(VaNet(self.K, self.att_K, backbone=self.backbone), self.cuda)
        self.model.weight_init()
        print('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        # --- 2nd ----------
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model_ema = Weight_EMA_Update(torch.nn.DataParallel(
                cuda(VaNet(self.K, self.att_K, backbone=self.backbone), self.cuda)), \
                                               self.model.state_dict(), decay=0.999)
        else:
            self.model_ema = Weight_EMA_Update(
                cuda(VaNet(self.K, self.att_K, backbone=self.backbone), self.cuda), \
                self.model.state_dict(), decay=0.999)


        self.optim = optim.Adam(self.model.parameters(),lr=self.lr,betas=(0.5,0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)

        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        if not self.ckpt_dir.exists() : self.ckpt_dir.mkdir(parents=True,exist_ok=True)
        self.load_ckpt = args.load_ckpt
        if self.load_ckpt != '' : self.load_checkpoint(self.load_ckpt)

        self.att_dir = Path(args.att_dir).joinpath(args.env_name)
        # if not self.att_dir.exists() : self.att_dir.mkdir(parents=True,exist_ok=True)

        # History
        self.history = dict()
        self.history['avg_acc']=0.
        self.history['avg_acc_ema']=0.
        self.history['info_loss']=0.
        self.history['latent_loss1']=0.
        self.history['latent_loss2']=0.
        self.history['class_loss']=0.
        self.history['class_loss_ori']=0.
        self.history['total_loss']=0.
        self.history['epoch']=0
        self.history['iter']=0

        # Tensorboard
        self.tensorboard = args.tensorboard
        if self.tensorboard :
            self.env_name = args.env_name
            self.summary_dir = Path(args.summary_dir).joinpath(args.env_name)
            if not self.summary_dir.exists() : self.summary_dir.mkdir(parents=True,exist_ok=True)
            self.tf = SummaryWriter(log_dir=self.summary_dir)
            self.tf.add_text(tag='argument',text_string=str(args),global_step=self.global_epoch)

        # Dataset
        self.dataset = args.dataset
        self.data_loader = return_data(args)

        # Classfication loss function
        self.class_lossfn = nn.CrossEntropyLoss()

    def set_mode(self,mode='train'):
        if mode == 'train' :
            self.model.train()
            self.model_ema.model.train()
        elif mode == 'eval' :
            self.model.eval()
            self.model_ema.model.eval()
        else : raise('mode error. It should be either train or eval')

    def train_anneal(self):
        self.set_mode('train')
        N = len(self.data_loader['train'])

        iter_delta = 0 # anl
        # iter_delta = 1000 * 10  # anl_2

        if self.beta == 1e-3:
            base_iter = 1100
        elif self.beta == 1e-2:
            base_iter = 2200
        elif self.beta == 5e-4:
            base_iter = 700

        if self.global_iter == 0:
            beta_anneal = 0.0
        elif self.global_iter > base_iter + iter_delta:
            # beta_anneal = 0.009952
            beta_anneal = self.beta
        else:
            beta_anneal = np.round((np.tanh((self.global_iter - 4500 - iter_delta) / 1000) + 1) / 2, decimals=6)

        for e in range(self.epoch):
            self.global_epoch += 1

            for idx, (images, labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1

                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))
                logit, logit_ori, latent_loss1, latent_loss2= self.model(x, self.num_sample)

                # class_loss = F.cross_entropy(logit, y).div(math.log(2))
                class_loss = self.class_lossfn(logit, y).div(math.log(2))
                class_loss_ori = self.class_lossfn(logit_ori, y).div(math.log(2))
                # info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
                info_loss = 0.5 * (latent_loss1 + latent_loss2).mean().div(math.log(2))  #div
                # info_loss = (latent_loss1 + latent_loss2).mean()
                # if self.min_zax:
                #     info_loss += -0.5 * (1 + 2 * att_std.log() - att_mu.pow(2) - att_std.pow(2)).sum(1).mean().div(math.log(2))
                # total_loss = class_loss + self.beta * info_loss
                total_loss = class_loss + beta_anneal * info_loss
                # if self.min_zax:
                #     info_att_loss = -0.5 * (1 + 2 * att_std.log() - att_mu.pow(2) - att_std.pow(2)).sum(1).mean().div(math.log(2))
                #     total_loss += gamma * info_att_loss

                # pdb.set_trace()

                izy_bound = math.log(10, 2) - class_loss
                izx_bound = info_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                self.model_ema.update(self.model.state_dict())

                prediction = F.softmax(logit, dim=1).max(1)[1]
                accuracy = torch.eq(prediction, y).float().mean()
                prediction_ori = F.softmax(logit_ori, dim=1).max(1)[1]
                accuracy_ori = torch.eq(prediction_ori, y).float().mean()

                if self.num_avg != 0:
                    avg_soft_logit, avg_soft_logit_ori, _, _ = self.model(x, self.num_avg)
                    avg_prediction = avg_soft_logit.max(1)[1]
                    avg_accuracy = torch.eq(avg_prediction, y).float().mean()
                    avg_prediction_ori = avg_soft_logit_ori.max(1)[1]
                    avg_accuracy_ori = torch.eq(avg_prediction_ori, y).float().mean()

                else:
                    avg_accuracy = Variable(cuda(torch.zeros(accuracy.size()), self.cuda))
                    avg_accuracy_ori = Variable(cuda(torch.zeros(accuracy_ori.size()), self.cuda))

                if self.global_iter % 150 == 0:
                    # print('i:{} IZY:{:.2f} IZX:{:.2f}'
                    #       .format(idx + 1, izy_bound.data.item(), izx_bound.data.item()), end=' ')
                    print('Train [{}][{}/{}]\tIZY:{:.2f} IZX:{:.2f}'
                          .format(e, idx, int(N), izy_bound.data.item(), izx_bound.data.item()), end=' ')
                    print('latent1:{:.2f} latent2:{:.2f}'
                          .format(latent_loss1.mean().data.item(), latent_loss2.mean().data.item()), end=' ')
                    # if self.min_zax:
                    #     print('IAX:{:.2f}'
                    #           .format(info_att_loss.data.item()), end=' ')
                    print('acc:{:.4f} avg_acc:{:.4f}'
                          .format(accuracy.data.item(), avg_accuracy.data.item()), end=' ')
                    print('acc_ori:{:.4f} avg_acc_ori:{:.4f}'
                          .format(accuracy_ori.data.item(), avg_accuracy_ori.data.item()))
                    #print('err:{:.4f} avg_err:{:.4f}'
                    #      .format(1 - accuracy.data.item(), 1 - avg_accuracy.data.item()))

                if self.global_iter % 50 == 0:
                    if self.tensorboard:
                        self.tf.add_scalars(main_tag='performance/accuracy',
                                            tag_scalar_dict={
                                                'train_one-shot': accuracy.data.item(),
                                                'train_multi-shot': avg_accuracy.data.item(),
                                                'train_one-shot_ori': accuracy_ori.data.item(),
                                                'train_multi-shot_ori': avg_accuracy_ori.data.item(),
                                            },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/error',
                                            tag_scalar_dict={
                                                'train_one-shot': 1 - accuracy.data.item(),
                                                'train_multi-shot': 1 - avg_accuracy.data.item(),
                                                'train_one-shot_ori': 1 - accuracy_ori.data.item(),
                                                'train_multi-shot_ori': 1 - avg_accuracy_ori.data.item(),
                                            },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={
                                                'train_one-shot_class': class_loss.data.item(),
                                                'train_one-shot_class_ori': class_loss_ori.data.item(),
                                                'train_one-shot_latent1': latent_loss1.mean().data.item(),
                                                'train_one-shot_latent2': latent_loss2.mean().data.item(),
                                                'train_one-shot_info': info_loss.data.item(),
                                                'train_one-shot_total': total_loss.data.item()},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='mutual_information/train',
                                            tag_scalar_dict={
                                                'I(Z;Y)': izy_bound.data.item(),
                                                'I(Z;X)': izx_bound.data.item()},
                                            global_step=self.global_iter)
                        # self.tf.add_scalars(main_tag='parameters/train',
                        #                     tag_scalar_dict={
                        #                         'att_mu': att_mu[0].data.item(),
                        #                         'att_std': att_std[0].data.item(),
                        #                         'std': std[0].data.item(),
                        #                         'mu': mu[0].data.item()},
                        #                     global_step=self.global_iter)

                        # self.tf.add_scalars('grad/att_mask',
                        #                     self.model.att_module[-1].weight.grad.abs().mean().item(), self.global_iter)
                        # self.tf.add_scalars('grad/encoder',
                        #                     self.model.encoder[-1].weight.grad.abs().mean().item(), self.global_iter)
                        if torch.cuda.device_count() > 1:
                            self.tf.add_scalars(main_tag='grad/train',
                                                tag_scalar_dict={
                                                    # 'att_mask': self.model.module.att_module[-1].weight.grad.abs().mean().item(),
                                                    'att_mask': self.model.module.att_module_mu[0].weight.grad.abs().mean().item(),
                                                    'encode': self.model.module.encode[-1].weight.grad.abs().mean().item(),
                                                    'decode': self.model.module.decode[-1].weight.grad.abs().mean().item()},
                                                global_step=self.global_iter)
                        else:
                            self.tf.add_scalars(main_tag='grad/train',
                                                tag_scalar_dict={
                                                    # 'att_mask': self.model.att_module[-1].weight.grad.abs().mean().item(),
                                                    'att_mask': self.model.att_module_mu[0].weight.grad.abs().mean().item(),
                                                    'encode': self.model.encode[-1].weight.grad.abs().mean().item(),
                                                    'decode': self.model.decode[-1].weight.grad.abs().mean().item()},
                                                global_step=self.global_iter)

            if self.global_iter <= base_iter + iter_delta: # about 0.009952
                beta_anneal = np.round((np.tanh((self.global_iter - 4500 - iter_delta) / 1000) + 1) / 2, decimals=6) # anl (anneal)

            if (self.global_epoch % 2) == 0: self.scheduler.step()
            print('--------------------\n')
            # if 'MS_COCO' in self.dataset:
            #     self.test_multicls(True)
            #     self.test_ema_multicls(True)
            # else:
            self.test(True)
            self.test_ema(True)

        print(" [*] Training Finished!")

    def train(self):
        self.set_mode('train')
        N = len(self.data_loader['train'])

        for e in range(self.epoch):
            self.global_epoch += 1
            bar = tqdm(self.data_loader['train'])
            for idx, (images, labels) in enumerate(bar):
            # for idx, (images, labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1

                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))
                logit, logit_ori, latent_loss1, latent_loss2= self.model(x, self.num_sample)
                # logit, logit_ori, mu, std= self.model(x, self.num_sample)

                # class_loss = F.cross_entropy(logit, y).div(math.log(2))
                class_loss = self.class_lossfn(logit, y).div(math.log(2))
                class_loss_ori = self.class_lossfn(logit_ori, y).div(math.log(2))
                # info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
                # info_loss = -0.5 * (1 + 2 * latent_loss2.log() - latent_loss1.pow(2) - latent_loss2.pow(2)).sum(1).mean().div(math.log(2)) #old2
                info_loss = 0.5 * (latent_loss1 + latent_loss2).mean().div(math.log(2)) #div
                # info_loss = (latent_loss1 + latent_loss2).mean()
                # if self.min_zax:
                #     info_loss += -0.5 * (1 + 2 * att_std.log() - att_mu.pow(2) - att_std.pow(2)).sum(1).mean().div(math.log(2))
                total_loss = class_loss + self.beta * info_loss # one
                # total_loss = class_loss + class_loss_ori + self.beta * info_loss # two
                # total_loss = 0.5 * (class_loss + class_loss_ori) + self.beta * info_loss # two_2
                # pdb.set_trace()
                # if self.min_zax:
                #     info_att_loss = -0.5 * (1 + 2 * att_std.log() - att_mu.pow(2) - att_std.pow(2)).sum(1).mean().div(math.log(2))
                #     total_loss += gamma * info_att_loss

                # pdb.set_trace()

                izy_bound = math.log(10, 2) - class_loss
                izx_bound = info_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                self.model_ema.update(self.model.state_dict())

                prediction = F.softmax(logit, dim=1).max(1)[1]
                accuracy = torch.eq(prediction, y).float().mean()
                prediction_ori = F.softmax(logit_ori, dim=1).max(1)[1]
                accuracy_ori = torch.eq(prediction_ori, y).float().mean()

                if self.num_avg != 0:
                    avg_soft_logit, avg_soft_logit_ori, _, _ = self.model(x, self.num_avg)
                    avg_prediction = avg_soft_logit.max(1)[1]
                    avg_accuracy = torch.eq(avg_prediction, y).float().mean()
                    avg_prediction_ori = avg_soft_logit_ori.max(1)[1]
                    avg_accuracy_ori = torch.eq(avg_prediction_ori, y).float().mean()

                else:
                    avg_accuracy = Variable(cuda(torch.zeros(accuracy.size()), self.cuda))
                    avg_accuracy_ori = Variable(cuda(torch.zeros(accuracy_ori.size()), self.cuda))

                # if self.global_iter % 150 == 0:
                #     # print('i:{} IZY:{:.2f} IZX:{:.2f}'
                #     #       .format(idx + 1, izy_bound.data.item(), izx_bound.data.item()), end=' ')
                #     print('Train [{}][{}/{}]\tIZY:{:.2f} IZX:{:.2f}'
                #           .format(e, idx, int(N), izy_bound.data.item(), izx_bound.data.item()), end=' ')
                #     print('latent1:{:.2f} latent2:{:.2f}'
                #           .format(latent_loss1.mean().data.item(), latent_loss2.mean().data.item()), end=' ')
                #     # if self.min_zax:
                #     #     print('IAX:{:.2f}'
                #     #           .format(info_att_loss.data.item()), end=' ')
                #     print('acc:{:.4f} avg_acc:{:.4f}'
                #           .format(accuracy.data.item(), avg_accuracy.data.item()), end=' ')
                #     print('acc_ori:{:.4f} avg_acc_ori:{:.4f}'
                #           .format(accuracy_ori.data.item(), avg_accuracy_ori.data.item()))
                #     #print('err:{:.4f} avg_err:{:.4f}'
                #     #      .format(1 - accuracy.data.item(), 1 - avg_accuracy.data.item()))

                bar.set_description('Train [{}][{}/{}]|IZY:{:.2f} IZX:{:.2f} '
                                    'latent1:{:.2f} latent2:{:.2f} '
                                    'acc:{:.4f} avg_acc:{:.4f} '
                                    .format(e, idx, int(N), izy_bound.data.item(), izx_bound.data.item(),
                                            latent_loss1.mean().data.item(), latent_loss2.mean().data.item(),
                                            accuracy.data.item(), avg_accuracy.data.item()))

                if self.global_iter % 50 == 0:
                    if self.tensorboard:
                        self.tf.add_scalars(main_tag='performance/accuracy',
                                            tag_scalar_dict={
                                                'train_one-shot': accuracy.data.item(),
                                                'train_multi-shot': avg_accuracy.data.item(),
                                                'train_one-shot_ori': accuracy_ori.data.item(),
                                                'train_multi-shot_ori': avg_accuracy_ori.data.item(),
                                            },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/error',
                                            tag_scalar_dict={
                                                'train_one-shot': 1 - accuracy.data.item(),
                                                'train_multi-shot': 1 - avg_accuracy.data.item(),
                                                'train_one-shot_ori': 1 - accuracy_ori.data.item(),
                                                'train_multi-shot_ori': 1 - avg_accuracy_ori.data.item(),
                                            },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={
                                                'train_one-shot_class': class_loss.data.item(),
                                                'train_one-shot_class_ori': class_loss_ori.data.item(),
                                                'train_one-shot_latent1': latent_loss1.mean().data.item(),
                                                'train_one-shot_latent2': latent_loss2.mean().data.item(),
                                                'train_one-shot_info': info_loss.data.item(),
                                                'train_one-shot_total': total_loss.data.item()},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='mutual_information/train',
                                            tag_scalar_dict={
                                                'I(Z;Y)': izy_bound.data.item(),
                                                'I(Z;X)': izx_bound.data.item()},
                                            global_step=self.global_iter)
                        # self.tf.add_scalars(main_tag='parameters/train',
                        #                     tag_scalar_dict={
                        #                         'att_mu': att_mu[0].data.item(),
                        #                         'att_std': att_std[0].data.item(),
                        #                         'std': std[0].data.item(),
                        #                         'mu': mu[0].data.item()},
                        #                     global_step=self.global_iter)

                        # self.tf.add_scalars('grad/att_mask',
                        #                     self.model.att_module[-1].weight.grad.abs().mean().item(), self.global_iter)
                        # self.tf.add_scalars('grad/encoder',
                        #                     self.model.encoder[-1].weight.grad.abs().mean().item(), self.global_iter)
                        if torch.cuda.device_count()>1:
                            self.tf.add_scalars(main_tag='grad/train',
                                                tag_scalar_dict={
                                                    # 'att_mask': self.model.module.att_module[-1].weight.grad.abs().mean().item(),
                                                    'att_mask': self.model.module.att_module_mu[0].weight.grad.abs().mean().item(),
                                                    'encode': self.model.module.encode[-1].weight.grad.abs().mean().item(),
                                                    'encode_ori': self.model.module.encode_ori[-1].weight.grad.abs().mean().item(),
                                                    'decode': self.model.module.decode[-1].weight.grad.abs().mean().item()},
                                                global_step=self.global_iter)
                        else:
                            self.tf.add_scalars(main_tag='grad/train',
                                                tag_scalar_dict={
                                                    # 'att_mask': self.model.att_module[-1].weight.grad.abs().mean().item(),
                                                    'att_mask': self.model.att_module_mu[0].weight.grad.abs().mean().item(),
                                                    'encode': self.model.encode[-1].weight.grad.abs().mean().item(),
                                                    'encode_ori': self.model.encode_ori[-1].weight.grad.abs().mean().item(),
                                                    'decode': self.model.decode[-1].weight.grad.abs().mean().item()},
                                                global_step=self.global_iter)


            if (self.global_epoch % 2) == 0: self.scheduler.step()
            print('--------------------\n')
            # if 'MS_COCO' in self.dataset:
            #     self.test_multicls(True)
            #     self.test_ema_multicls(True)
            # else:
            self.test(True)
            self.test_ema(True)

        print(" [*] Training Finished!")

    def test(self, save_ckpt=False, save_att=False):
        self.set_mode('eval')

        class_loss = 0
        class_loss_ori = 0
        latent_loss1 = 0
        latent_loss2 = 0
        info_loss = 0
        correct = 0
        avg_correct = 0
        correct_ori = 0
        avg_correct_ori = 0
        total_num = 0
        #if self.min_zax:
        #    info_att_loss = 0
        # bar = tqdm(self.data_loader['test'])
        # for idx, (images, labels) in enumerate(bar):
        for idx, (images, labels) in enumerate(self.data_loader['test']):

            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            logit, logit_ori, l_loss1, l_loss2 = self.model(x, self.num_sample)
            #pdb.set_trace()
            current_num = y.size(0)
            total_num += current_num

            # c_loss = F.cross_entropy(logit, y).div(math.log(2))
            c_loss = self.class_lossfn(logit, y).div(math.log(2))
            class_loss += current_num * 1.0 / total_num * (c_loss.item() - class_loss)
            c_loss_ori = self.class_lossfn(logit_ori, y).div(math.log(2))
            class_loss_ori += current_num * 1.0 / total_num * (c_loss_ori.item() - class_loss_ori)
            #i_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
            # i_loss = -0.5 * (1 + 2 * latent_loss2.log() - latent_loss1.pow(2) - latent_loss2.pow(2)).sum(1).mean().div(math.log(2))  #old2
            i_loss = 0.5 * (l_loss1.mean().item() + l_loss2.mean().item())/(math.log(2)) #div
            # i_loss = (latent_loss1 + latent_loss2).mean()
            # if self.min_zax:
            #     i_loss += -0.5 * (1 + 2 * att_std.log() - att_mu.pow(2) - att_std.pow(2)).sum(1).mean().div(math.log(2))
            info_loss += current_num * 1.0 / total_num * (i_loss - info_loss)

            latent_loss1 += current_num * 1.0 / total_num * (l_loss1.mean().item() - latent_loss1)
            latent_loss2 += current_num * 1.0 / total_num * (l_loss2.mean().item() - latent_loss2)

            #if self.min_zax:
            #    i_a_loss = -0.5 * (1 + 2 * att_std.log() - att_mu.pow(2) - att_std.pow(2)).sum(1).mean().div(math.log(2))
            #    info_att_loss += current_num * 1.0 / total_num * (i_a_loss.item() - info_att_loss)

            prediction = F.softmax(logit, dim=1).max(1)[1]
            correct += torch.eq(prediction, y).float().sum().item()
            prediction_ori = F.softmax(logit_ori, dim=1).max(1)[1]
            correct_ori += torch.eq(prediction_ori, y).float().sum().item()

            if self.num_avg != 0:
                avg_soft_logit, avg_soft_logit_ori, _, _ = self.model(x, self.num_avg)
                avg_prediction = avg_soft_logit.max(1)[1]
                avg_correct += torch.eq(avg_prediction, y).float().sum().item()
                avg_prediction_ori = avg_soft_logit_ori.max(1)[1]
                avg_correct_ori += torch.eq(avg_prediction_ori, y).float().sum().item()

            else:
                avg_correct = Variable(cuda(torch.zeros(correct.size()), self.cuda)).item()
                avg_correct_ori = Variable(cuda(torch.zeros(correct_ori.size()), self.cuda)).item()

        accuracy = correct / total_num
        avg_accuracy = avg_correct / total_num
        accuracy_ori = correct_ori / total_num
        avg_accuracy_ori = avg_correct_ori / total_num

        total_loss = class_loss + self.beta * info_loss # one
        # total_loss = class_loss + class_loss_ori + self.beta * info_loss  # two
        # total_loss = 0.5 * (class_loss + class_loss_ori) + self.beta * info_loss  # two_2

        #if self.min_zax:
        #    total_loss += gamma * info_att_loss
        izy_bound = math.log(10, 2) - class_loss
        izx_bound = info_loss

        print('[TEST RESULT]')
        print('e:{} IZY:{:.2f} IZX:{:.2f}'
              .format(self.global_epoch, izy_bound, izx_bound), end=' ')
        print('latent1:{:.2f} latent2:{:.2f}'
              .format(latent_loss1, latent_loss2), end=' ')
        #if self.min_zax:
        #    print('IAX:{:.2f}'
        #          .format(info_att_loss), end=' ')
        print('acc:{:.4f} avg_acc:{:.4f}'
              .format(accuracy, avg_accuracy), end=' ')
        print('acc_ori:{:.4f} avg_acc_ori:{:.4f}'
              .format(accuracy_ori, avg_accuracy_ori))
        #print('err:{:.4f} avg_err:{:.4f}'
        #      .format(1 - accuracy, 1 - avg_accuracy))
        # print()

        if self.history['avg_acc'] < avg_accuracy and self.history['avg_acc_ema'] < avg_accuracy:
            self.history['avg_acc'] = avg_accuracy
            self.history['class_loss'] = class_loss
            self.history['class_loss_ori'] = class_loss
            self.history['latent_loss1'] = latent_loss1
            self.history['latent_loss2'] = latent_loss2
            self.history['info_loss'] = info_loss
            #if self.min_zax:
            #    self.history['info_att_loss'] = info_att_loss
            self.history['total_loss'] = total_loss
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            if save_ckpt: self.save_checkpoint('best_acc.tar')

        if self.tensorboard:
            self.tf.add_scalars(main_tag='performance/accuracy',
                                tag_scalar_dict={
                                    'test_one-shot': accuracy,
                                    'test_multi-shot': avg_accuracy,
                                    'test_one-shot_ori': accuracy_ori,
                                    'test_multi-shot_ori': avg_accuracy_ori,
                                },
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/error',
                                tag_scalar_dict={
                                    'test_one-shot': 1 - accuracy,
                                    'test_multi-shot': 1 - avg_accuracy,
                                    'test_one-shot_ori': 1 - accuracy_ori,
                                    'test_multi-shot_ori': 1 - avg_accuracy_ori,
                                },
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/cost',
                                tag_scalar_dict={
                                    'test_one-shot_class': class_loss,
                                    'test_one-shot_class_ori': class_loss_ori,
                                    'test_one-shot_latent1': latent_loss1,
                                    'test_one-shot_latent2': latent_loss2,
                                    'test_one-shot_info': info_loss,
                                    'test_one-shot_total': total_loss},
                                global_step=self.global_iter)
            # self.tf.add_scalars(main_tag='parameters/train',
            #                    tag_scalar_dict={
            #                        'att_mu': att_mu[0].data.item(),
            #                        'att_std': att_std[0].data.item(),
            #                        'std': std[0].data.item(),
            #                        'mu': mu[0].data.item()},
            #                    global_step=self.global_iter)
            self.tf.add_scalars(main_tag='mutual_information/test',
                                tag_scalar_dict={
                                    'I(Z;Y)': izy_bound,
                                    'I(Z;X)': izx_bound},
                                global_step=self.global_iter)

        self.set_mode('train')

        if save_att:
            self.save_attention()

    def test_ema(self, save_ckpt=False, save_att=False):
        self.set_mode('eval')

        class_loss = 0
        class_loss_ori = 0
        latent_loss1 = 0
        latent_loss2 = 0
        info_loss = 0
        correct = 0
        avg_correct = 0
        correct_ori = 0
        avg_correct_ori = 0
        total_num = 0
        #if self.min_zax:
        #    info_att_loss = 0
        for idx, (images, labels) in enumerate(self.data_loader['test']):

            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            logit, logit_ori, l_loss1, l_loss2 = self.model_ema.model(x, self.num_sample)
            #pdb.set_trace()
            current_num = y.size(0)
            total_num += current_num

            # c_loss = F.cross_entropy(logit, y).div(math.log(2))
            c_loss = self.class_lossfn(logit, y).div(math.log(2))
            class_loss += current_num * 1.0 / total_num * (c_loss.item() - class_loss)
            c_loss_ori = self.class_lossfn(logit_ori, y).div(math.log(2))
            class_loss_ori += current_num * 1.0 / total_num * (c_loss_ori.item() - class_loss_ori)
            #i_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
            # i_loss = -0.5 * (1 + 2 * latent_loss2.log() - latent_loss1.pow(2) - latent_loss2.pow(2)).sum(1).mean().div(math.log(2))  #old2
            i_loss = 0.5 * (l_loss1.mean().item() + l_loss2.mean().item())/(math.log(2)) #div
            # i_loss = (latent_loss1 + latent_loss2).mean()
            # if self.min_zax:
            #     i_loss += -0.5 * (1 + 2 * att_std.log() - att_mu.pow(2) - att_std.pow(2)).sum(1).mean().div(math.log(2))
            info_loss += current_num * 1.0 / total_num * (i_loss - info_loss)

            latent_loss1 += current_num * 1.0 / total_num * (l_loss1.mean().item() - latent_loss1)
            latent_loss2 += current_num * 1.0 / total_num * (l_loss2.mean().item() - latent_loss2)

            #if self.min_zax:
            #    i_a_loss = -0.5 * (1 + 2 * att_std.log() - att_mu.pow(2) - att_std.pow(2)).sum(1).mean().div(math.log(2))
            #    info_att_loss += current_num * 1.0 / total_num * (i_a_loss.item() - info_att_loss)

            prediction = F.softmax(logit, dim=1).max(1)[1]
            correct += torch.eq(prediction, y).float().sum().item()
            prediction_ori = F.softmax(logit_ori, dim=1).max(1)[1]
            correct_ori += torch.eq(prediction_ori, y).float().sum().item()

            if self.num_avg != 0:
                avg_soft_logit, avg_soft_logit_ori, _, _ = self.model(x, self.num_avg)
                avg_prediction = avg_soft_logit.max(1)[1]
                avg_correct += torch.eq(avg_prediction, y).float().sum().item()
                avg_prediction_ori = avg_soft_logit_ori.max(1)[1]
                avg_correct_ori += torch.eq(avg_prediction_ori, y).float().sum().item()

            else:
                avg_correct = Variable(cuda(torch.zeros(correct.size()), self.cuda)).item()
                avg_correct_ori = Variable(cuda(torch.zeros(correct_ori.size()), self.cuda)).item()

        accuracy = correct / total_num
        avg_accuracy = avg_correct / total_num
        accuracy_ori = correct_ori / total_num
        avg_accuracy_ori = avg_correct_ori / total_num

        total_loss = class_loss + self.beta * info_loss # one
        # total_loss = class_loss + class_loss_ori + self.beta * info_loss  # two
        # total_loss = 0.5 * (class_loss + class_loss_ori) + self.beta * info_loss  # two_2

        #if self.min_zax:
        #    total_loss += gamma * info_att_loss
        izy_bound = math.log(10, 2) - class_loss
        izx_bound = info_loss

        print('[TEST EMA RESULT]')
        print('e:{} IZY:{:.2f} IZX:{:.2f}'
              .format(self.global_epoch, izy_bound, izx_bound), end=' ')
        print('latent1:{:.2f} latent2:{:.2f}'
              .format(latent_loss1, latent_loss2), end=' ')
        #if self.min_zax:
        #    print('IAX:{:.2f}'
        #          .format(info_att_loss), end=' ')
        print('acc:{:.4f} avg_acc:{:.4f}'
              .format(accuracy, avg_accuracy), end=' ')
        print('acc_ori:{:.4f} avg_acc_ori:{:.4f}'
              .format(accuracy_ori, avg_accuracy_ori))
        #print('err:{:.4f} avg_err:{:.4f}'
        #      .format(1 - accuracy, 1 - avg_accuracy))
        # print()

        if self.history['avg_acc_ema'] < avg_accuracy and self.history['avg_acc'] < avg_accuracy:
            self.history['avg_acc_ema'] = avg_accuracy
            self.history['class_loss'] = class_loss
            self.history['class_loss_ori'] = class_loss
            self.history['latent_loss1'] = latent_loss1
            self.history['latent_loss2'] = latent_loss2
            self.history['info_loss'] = info_loss
            #if self.min_zax:
            #    self.history['info_att_loss'] = info_att_loss
            self.history['total_loss'] = total_loss
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            if save_ckpt: self.save_checkpoint('best_acc.tar')

        if self.tensorboard:
            self.tf.add_scalars(main_tag='performance/accuracy',
                                tag_scalar_dict={
                                    'test_one-shot': accuracy,
                                    'test_multi-shot': avg_accuracy,
                                    'test_one-shot_ori': accuracy_ori,
                                    'test_multi-shot_ori': avg_accuracy_ori,
                                },
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/error',
                                tag_scalar_dict={
                                    'test_one-shot': 1 - accuracy,
                                    'test_multi-shot': 1 - avg_accuracy,
                                    'test_one-shot_ori': 1 - accuracy_ori,
                                    'test_multi-shot_ori': 1 - avg_accuracy_ori,
                                },
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/cost',
                                tag_scalar_dict={
                                    'test_one-shot_class': class_loss,
                                    'test_one-shot_class_ori': class_loss_ori,
                                    'test_one-shot_latent1': latent_loss1,
                                    'test_one-shot_latent2': latent_loss2,
                                    'test_one-shot_info': info_loss,
                                    'test_one-shot_total': total_loss},
                                global_step=self.global_iter)
            #self.tf.add_scalars(main_tag='parameters/train',
            #                    tag_scalar_dict={
            #                        'att_mu': att_mu[0].data.item(),
            #                        'att_std': att_std[0].data.item(),
            #                        'std': std[0].data.item(),
            #                        'mu': mu[0].data.item()},
            #                    global_step=self.global_iter)
            self.tf.add_scalars(main_tag='mutual_information/test',
                                tag_scalar_dict={
                                    'I(Z;Y)': izy_bound,
                                    'I(Z;X)': izx_bound},
                                global_step=self.global_iter)

        self.set_mode('train')

        if save_att:
            self.save_attention()

    def save_attention(self):
        #if self.history['avg_acc'] > self.history['avg_acc_ema']:
        #    print('Saving model attention ...')
        #else:
        #    print('Saving model_ema attention ...')

        self.set_mode('eval')
        if 'MNIST' in self.dataset:
            va_trans = transforms.Compose([
                transforms.Normalize(mean=(-0.5 / 0.5,), std=(1 / 0.5,)),
                transforms.ToPILImage()])
            AttSize = MNIST_RESIZE
            ImgSize = MNIST_RESIZE
        elif 'CIFAR' in self.dataset:
            va_trans = transforms.Compose([
                # transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
                transforms.ToPILImage()])
            AttSize = [16, 16]
            ImgSize = CIFAR_RESIZE
        elif 'STL' in self.dataset:
            va_trans = transforms.Compose([
                transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.ToPILImage()])
            if self.backbone == 'alex':
                AttSize = [11, 11] # alex
            elif self.backbone == 'vgg16':
                AttSize = [12, 12] # vgg
                # AttSize = [24, 24] # vgg
                # AttSize = [96, 96] # vgg f0
            ImgSize = STL_RESIZE
        elif self.dataset in ['CUB', 'ILSVRC', 'CUB_crop']:
            # mean_vals = [0.485, 0.456, 0.406]
            # std_vals = [0.229, 0.224, 0.225]
            va_trans = transforms.Compose([
                transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.ToPILImage()])
            if self.backbone == 'alex':
                AttSize = [11, 11] # alex
            elif self.backbone == 'vgg16' or 'wrn_50_2':
                AttSize = [14, 14] # vgg
                # AttSize = [28, 28] # vgg
                # AttSize = [12, 12] # vgg
                # AttSize = [24, 24] # vgg
                # AttSize = [96, 96] # vgg f0
            ImgSize = WOL_RESIZE
        # elif 'MS_COCO' in self.dataset:
        #     va_trans = transforms.Compose([
        #         transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        #                              std=[1/0.229, 1/0.224, 1/0.225]),
        #         transforms.ToPILImage()])
        #     AttSize = [28, 28]
        #     ImgSize = COCO_RESIZE

        cnt = 0
        if not self.att_dir.exists(): self.att_dir.mkdir(parents=True, exist_ok=True)
        for idx, (images, labels) in enumerate(self.data_loader['test']):

            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            pred_logits, pred_logits_ori, att_maps = self.model(x, self.num_sample, train=False)
            _, predicted = torch.max(pred_logits.data, 1)
            flag = (predicted == y)
            _, predicted_ori = torch.max(pred_logits_ori.data, 1)
            flag_ori = (predicted_ori == y)

            # att_maps = att_maps.view(-1, self.test_batch, att_maps.size(2)) # wrong!
            att_maps_list = []
            for d_i in range(torch.cuda.device_count()):
                att_maps_list.append(att_maps[d_i * self.num_sample:(d_i + 1) * self.num_sample, :, :])
            att_maps = torch.cat(att_maps_list, dim=1)

            for b_num in range(self.test_batch):
                cnt = cnt + 1
                if cnt > va_num:
                    break

                ori_img = va_trans(images[b_num])
                # scipy.misc.imsave(os.path.join(out_folder, 'test_{:04d}_{}.jpg'.format(cnt, flag[b_num])),
                #                  ori_img.detach().cpu().numpy())
                ori_img.save(os.path.join(self.att_dir, 'test_{:04d}_cls{:02d}_{}_{}.jpg'.format(cnt, labels[b_num], flag[b_num], flag_ori[b_num])))

                # if return att_mask.mean(0)
                # att_map = att_maps[b_num].view(MNIST_RESIZE).detach().cpu().numpy()
                # tmp_map = postprocess_prediction(att_map)
                # scipy.misc.imsave(os.path.join(self.att_dir, 'test_{:04d}_{}_att.png'.format(
                #     cnt, flag[b_num])), tmp_map)
                # pdb.set_trace()

                # if return att_mask
                for a_num in range(att_maps.size(0)):
                # for a_num in range(att_maps.size(0)//torch.cuda.device_count()):
                    att_map = att_maps[a_num, b_num].view(AttSize).detach().cpu().numpy()
                    tmp_map = postprocess_prediction(att_map, size=ImgSize)
                    scipy.misc.imsave(os.path.join(self.att_dir, 'test_{:04d}_cls{:02d}_{}_{}_att{:02d}.png'.format(
                        cnt, labels[b_num], flag[b_num], flag_ori[b_num], a_num)), tmp_map)

            if cnt > va_num:
                break

        self.set_mode('train')

    def save_checkpoint(self, filename='best_acc.tar'):
        if torch.cuda.device_count()>1:
            # --- 1st -----
            # model_states = {
            #         'net':self.model.module.state_dict(),
            #         'net_ema':self.model_ema.model.state_dict(),
            #         }
            # --- 2nd ----
            model_states = {
                    'net':self.model.module.state_dict(),
                    'net_ema':self.model_ema.model.module.state_dict(),
                    }

        else:
            model_states = {
                    'net':self.model.state_dict(),
                    'net_ema':self.model_ema.model.state_dict(),
                    }
        optim_states = {
                'optim':self.optim.state_dict(),
                }
        states = {
                'iter':self.global_iter,
                'epoch':self.global_epoch,
                'history':self.history,
                'args':self.args,
                'model_states':model_states,
                'optim_states':optim_states,
                }

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states,file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path,self.global_iter))

    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']
            # pdb.set_trace()
            if torch.cuda.device_count()>1:
                tmp_state_dict = self.model.state_dict()
                for k in tmp_state_dict.keys():
                    tmp_state_dict[k] = checkpoint['model_states']['net'][k[7:]]
                self.model.load_state_dict(tmp_state_dict)
                tmp_ema_state_dict = self.model_ema.model.state_dict()
                for k in tmp_state_dict.keys():
                    tmp_ema_state_dict[k] = checkpoint['model_states']['net_ema'][k[7:]]
                self.model_ema.model.load_state_dict(tmp_ema_state_dict)
            else:
                self.model.load_state_dict(checkpoint['model_states']['net'])
                self.model_ema.model.load_state_dict(checkpoint['model_states']['net_ema'])

            self.optim.load_state_dict(checkpoint['optim_states']['optim'])

            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))


def cal_iou(box1, box2):
    """
    support:
    1. box1 and box2 are the same shape: [N, 4] [x1,y1,x2,y2]
    2.
    :param box1:
    :param box2:
    :return:
    """
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]) + 1
    ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    iou_val = i_area / (box1_area + box2_area - i_area)

    return iou_val

