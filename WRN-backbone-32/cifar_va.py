'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import pdb
import math
import scipy.misc
import cv2
import numpy as np
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from torch.autograd import Variable
import torchnet as tnt

from tqdm import tqdm
from pathlib import Path

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig, utils_my
from utils.config import *

from apex import amp

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100/SVHN Training')
parser.add_argument('--phase', default='train', type=str, help='running phase')
parser.add_argument('--ckptname', default='checkpoint.pth.tar', type=str, help='filename of model')
parser.add_argument('--bestname', default='model_best.pth.tar', type=str, help='filename of model')

# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)') # but the two wrn are the same testing set ...
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0.3, type=float,
                    metavar='Dropout', help='Dropout ratio')
# parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
#                         help='Decrease learning rate at these epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 160],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.2, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='WRN-32', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='wrn_va',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# VA param
parser.add_argument('--va_beta', default=beta, type=float, help='beta')
parser.add_argument('--K', default=512, type=int, help='dimension of encoding Z')
parser.add_argument('--att_K', default=256, type=int, help='dimension of attention A')
parser.add_argument('--num_sample', default=num_sample, type=int, help='the number of samples')
# parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--seed', default=random.randint(2, 10000), type=int, help='random seed')
parser.add_argument('--num_avg', default=4, type=int, help='the number of samples when\
               perform multi-shot prediction')  #

# parser.add_argument('--att_dir', default='att_maps', type=str, help='att_mask directory path')
parser.add_argument('--att_dir', default='att_maps_tmp', type=str, help='att_mask directory path')
parser.add_argument('--return_att', default=False, type=utils_my.str2bool, help='enable return att_mask')
parser.add_argument('--qt_num', default=2, type=int, help='number of quantum')
parser.add_argument('--vq_coef', default=0.4, type=float, help='weight for vq_loss')
parser.add_argument('--comit_coef', default=0.1, type=float, help='weight for commit_loss')
parser.add_argument('--rd_init', default=False, type=utils_my.str2bool, help='randomly init quantum layer or not.')
parser.add_argument('--decay', default=None, type=float, help='decay quantum layer if not set as None')
parser.add_argument('--qt_trainable', default=True, type=utils_my.str2bool, help='quantum is trainable or not.')

parser.add_argument('--lr_ratio', default=1e-4, type=float, help='ratio of emb weight learning rate')
# parser.add_argument('--hfc_r', default=0, type=int, help='radius for frequency bypassing')
# parser.add_argument('--freq_results', default=feature_freq_folder, type=str, help='frequency results saving directory path')
# parser.add_argument('--freq_metric', default='sim', type=str, help='metric for comparing results of frequency inputs with original.')
# parser.add_argument('--freq_m_ths', default='[0.85,0.9,0.95]', type=str, help='json list with epochs to drop lr on')

# parser.add_argument('--aug_results', default=feature_aug_folder, type=str, help='info-augmented results saving directory path')
# parser.add_argument('--aug_metric', default='cos_sim', type=str, help='metric for comparing results of info-augmented inputs with original.')
# parser.add_argument('--aug_m_ths', default='[0.85,0.9,0.95]', type=str, help='json list with epochs to drop lr on')
# parser.add_argument('--aug_type', default='', type=str, help='augmentation type: color or svhn')
# parser.add_argument('--aug_wsize', type=int, nargs='+', default=[8, 8], help='window size of augmentation.')

# parser.add_argument('--vis_results', default=vis_folder, type=str, help='frequency results saving directory path')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
# assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    if args.dataset[:5] == 'cifar':
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'), # _rft
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
        else:
            dataloader = datasets.CIFAR100
            num_classes = 100

        trainset = dataloader(root='../dataset', train=True, download=True, transform=transform_train)
        # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

        testset = dataloader(root='../dataset', train=False, download=False, transform=transform_test)
        # testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    elif args.dataset == 'svhn':
        input_size = 32
        pad_size = 4
        transform_train = transforms.Compose([
            # transforms.Resize((input_size, input_size)),
            # transforms.RandomCrop(input_size, padding=pad_size),  # _p0
            transforms.RandomCrop(input_size, padding=pad_size, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),  # ========================
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ], )

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ], )

        dataloader = datasets.SVHN
        num_classes = 10

        trainset = dataloader(root='../dataset', split='train', download=True, transform=transform_train)
        # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

        testset = dataloader(root='../dataset', split='test', download=True, transform=transform_test)
        # testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch == 'wrn_va':
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                    K=args.K,
                )
    elif args.arch=='wrn_va_qt':
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                    K=args.K,
                    qt_num=args.qt_num,
                    vq_coef=args.vq_coef,
                    comit_coef=args.comit_coef,
                    rd_init=args.rd_init,
                    decay=args.decay,
                    qt_trainable=args.qt_trainable
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    if args.arch.endswith('va_qt'):
        print('creating optimizer with lr %.6f for base and %.6f for emb' % (args.lr, args.lr * args.lr_ratio))
        my_list = ['emb.weight', 'module.emb.weight']
        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
        optimizer = optim.SGD([{'params': base_params}, {'params': params, 'lr': args.lr * args.lr_ratio}],
                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model = model.cuda()
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    model = torch.nn.DataParallel(model).cuda()

    # Resume
    title = args.dataset + '-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(os.path.join(args.resume, args.ckptname)), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(os.path.join(args.resume, args.ckptname))
        checkpoint = torch.load(os.path.join(args.resume, args.ckptname))
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        # logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        # if args.arch.endswith('_va'):
        if args.arch.endswith('_va'):
            test_loss, test_acc = test_va(testloader, model, criterion, start_epoch, use_cuda)
        elif args.arch.endswith('_va_qt'):
            test_loss, test_acc = test_va_qt(testloader, model, criterion, start_epoch, use_cuda)

        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        if args.arch.endswith('_va'):
            # print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
            print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
            train_loss, train_acc = train_va(trainloader, model, criterion, optimizer, epoch, use_cuda)
            # train_loss, train_acc = train_va_anl(trainloader, model, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = test_va(testloader, model, criterion, epoch, use_cuda)
        elif args.arch.endswith('_va_qt'):
            # print('Epoch: [%d | %d] LR: %f LR_emb: %f' % (epoch + 1, args.epochs, state['lr'], optimizer.param_groups[1]['lr']))
            print('Epoch: [%d | %d] LR: %f LR_emb: %f' % (epoch + 1, args.epochs,
                                                          optimizer.param_groups[0]['lr'],
                                                          optimizer.param_groups[1]['lr']))
            train_loss, train_acc = train_va_qt(trainloader, model, criterion, optimizer, epoch, use_cuda)
            # train_loss, train_acc = train_va_qt_anl(trainloader, model, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = test_va_qt(testloader, model, criterion, epoch, use_cuda)


        # append logger file
        # logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
        logger.append([optimizer.param_groups[0]['lr'], train_loss, test_loss, train_acc, test_acc])
        # logger.append([epoch, optimizer.param_groups[0]['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'amp': amp.state_dict()
            }, is_best, checkpoint=args.checkpoint)

        # save attention maps for best model
        if is_best:
            if args.arch.endswith('va'):
                save_attention_va(model, testloader, False)
            elif args.arch.endswith('va_qt'):
                save_attention_va_qt(model, testloader, False)
            print('******** Best [%d] %f for %s ********' % (epoch+1, best_acc, args.checkpoint))

    logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train_va(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    classacc = tnt.meter.ClassErrorMeter(topk=[1, 3, 5], accuracy=True)
    classacc.reset()
    end = time.time()


    bar = tqdm(trainloader)
    for batch_idx, (inputs, targets) in enumerate(bar):
        # pdb.set_trace()
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            # inputs, targets = inputs.cuda(), targets.cuda(async=True)
            inputs= inputs.cuda()
            targets = targets.to(inputs.device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        logit, latent_loss = model(inputs, num_sample=args.num_sample) # train=True, return_all=False
        class_loss = criterion(logit, targets).div(math.log(2))
        info_loss = latent_loss.mean().div(math.log(2))
        loss = class_loss + args.va_beta * info_loss


        # att_outputs, per_outputs, _ = model(inputs)
        # # att_outputs.size() torch.Size([64, 800])
        # # per_outputs.size() torch.Size([256, 200])

        # att_loss = criterion(att_outputs, targets)
        # per_loss = criterion(per_outputs, targets)
        # loss = att_loss + per_loss

        # measure accuracy and record loss
        # pdb.set_trace()
        # prec1, prec5 = accuracy(per_outputs.data, targets.data, topk=(1, 5))
        # prec1, prec5 = accuracy(logit.data, targets.data, topk=(1, 5))
        # # losses.update(loss.data[0], inputs.size(0))
        # # top1.update(prec1[0], inputs.size(0))
        # # top5.update(prec5[0], inputs.size(0))
        # losses.update(loss.item(), inputs.size(0))
        # top1.update(prec1.item(), inputs.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))
        classacc.add(logit.data, targets.data)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        bar.set_description('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    # total=bar.elapsed_td,
                    # eta=bar.eta_td,
                    loss=losses.avg,
                    # top1=top1.avg,
                    # top5=top5.avg,
                    top1=classacc.value()[0],
                    top5=classacc.value()[2],
                    ))
        # plot progress

    return (losses.avg, classacc.value()[0])

def test_va(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    classacc = tnt.meter.ClassErrorMeter(topk=[1, 3, 5], accuracy=True)
    classacc.reset()
    # switch to evaluate mode
    model.eval()

    end = time.time()

    bar = tqdm(testloader)
    for batch_idx, (inputs, targets) in enumerate(bar):

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            # inputs, targets = inputs.cuda(), targets.cuda()
            inputs = inputs.cuda()
            targets = targets.to(inputs.device)
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        # _, outputs, attention = model(inputs)
        # loss = criterion(outputs, targets)
        logit, latent_loss, attention = model(inputs, num_sample=args.num_avg, return_all=True)  # train=True, return_all=True
        class_loss = criterion(logit, targets).div(math.log(2))
        info_loss = latent_loss.mean().div(math.log(2))
        loss = class_loss + args.va_beta * info_loss

        # print(attention.min(), attention.max())
        """
        np_inputs = inputs.numpy()
        np_att = attention.numpy()
        for item_in, item_att in zip(np_inputs, np_att):
            print(item_in.shape, item_att.shape)
        """

        # measure accuracy and record loss
        # # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # prec1, prec5 = accuracy(logit.data, targets.data, topk=(1, 5))
        # # losses.update(loss.data[0], inputs.size(0))
        # # top1.update(prec1[0], inputs.size(0))
        # # top5.update(prec5[0], inputs.size(0))
        # losses.update(loss.item(), inputs.size(0))
        # top1.update(prec1.item(), inputs.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))
        classacc.add(logit.data, targets.data)

        # classacc.add(logit.data, targets.data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # pdb.set_trace()
        bar.set_description(
            '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} '
            '| top1: {top1: .4f} | top5: {top5: .4f} '
            '| att_min: {att_min: .4f} | att_max: {att_max: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                # total=bar.elapsed_td,
                # eta=bar.eta_td,
                loss=losses.avg,
                # top1=top1.avg,
                # top5=top5.avg,
                top1=classacc.value()[0],
                top5=classacc.value()[2],
                att_min=attention.mean(0).min().item(),
                att_max=attention.mean(0).max().item()
            )
        )

        # plot progress

    return (losses.avg, classacc.value()[0])

def train_va_qt(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    classacc = tnt.meter.ClassErrorMeter(topk=[1, 3, 5], accuracy=True)
    classacc.reset()
    end = time.time()


    bar = tqdm(trainloader)
    for batch_idx, (inputs, targets) in enumerate(bar):
        # pdb.set_trace()
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            # inputs, targets = inputs.cuda(), targets.cuda(async=True)
            inputs= inputs.cuda()
            targets = targets.to(inputs.device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        logit, latent_loss, vq_loss, commit_loss = model(inputs, num_sample=args.num_sample) # train=True, return_all=False
        class_loss = criterion(logit, targets).div(math.log(2))
        info_loss = latent_loss.mean().div(math.log(2))
        qt_loss = args.vq_coef * vq_loss.mean() + args.comit_coef * commit_loss.mean()
        loss = class_loss + args.va_beta * info_loss + qt_loss


        # att_outputs, per_outputs, _ = model(inputs)
        # # att_outputs.size() torch.Size([64, 800])
        # # per_outputs.size() torch.Size([256, 200])

        # att_loss = criterion(att_outputs, targets)
        # per_loss = criterion(per_outputs, targets)
        # loss = att_loss + per_loss

        # measure accuracy and record loss
        # pdb.set_trace()
        # # prec1, prec5 = accuracy(per_outputs.data, targets.data, topk=(1, 5))
        # prec1, prec5 = accuracy(logit.data, targets.data, topk=(1, 5))
        # # losses.update(loss.data[0], inputs.size(0))
        # # top1.update(prec1[0], inputs.size(0))
        # # top5.update(prec5[0], inputs.size(0))
        # losses.update(loss.item(), inputs.size(0))
        # top1.update(prec1.item(), inputs.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))
        classacc.add(logit.data, targets.data)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        bar.set_description('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    # total=bar.elapsed_td,
                    # eta=bar.eta_td,
                    loss=losses.avg,
                    # top1=top1.avg,
                    # top5=top5.avg,
                    top1=classacc.value()[0],
                    top5=classacc.value()[2],
                    ))
        # plot progress

    return (losses.avg, classacc.value()[0])

def test_va_qt(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    classacc = tnt.meter.ClassErrorMeter(topk=[1, 3, 5], accuracy=True)
    classacc.reset()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    bar = tqdm(testloader)
    for batch_idx, (inputs, targets) in enumerate(bar):

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            # inputs, targets = inputs.cuda(), targets.cuda()
            inputs = inputs.cuda()
            targets = targets.to(inputs.device)
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        # _, outputs, attention = model(inputs)
        # loss = criterion(outputs, targets)
        logit, latent_loss, vq_loss, commit_loss, attention, attention_q = \
            model(inputs, num_sample=args.num_avg, return_all=True)  # train=True, return_all=False
        class_loss = criterion(logit, targets).div(math.log(2))
        info_loss = latent_loss.mean().div(math.log(2))
        qt_loss = args.vq_coef * vq_loss.mean() + args.comit_coef * commit_loss.mean()
        loss = class_loss + args.va_beta * info_loss + qt_loss

        # print(attention.min(), attention.max())
        """
        np_inputs = inputs.numpy()
        np_att = attention.numpy()
        for item_in, item_att in zip(np_inputs, np_att):
            print(item_in.shape, item_att.shape)
        """

        # measure accuracy and record loss
        # # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # prec1, prec5 = accuracy(logit.data, targets.data, topk=(1, 5))
        # # losses.update(loss.data[0], inputs.size(0))
        # # top1.update(prec1[0], inputs.size(0))
        # # top5.update(prec5[0], inputs.size(0))
        # losses.update(loss.item(), inputs.size(0))
        # top1.update(prec1.item(), inputs.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))
        classacc.add(logit.data, targets.data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # pdb.set_trace()
        bar.set_description(
            '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} '
            '| top1: {top1: .4f} | top5: {top5: .4f} '
            '| att_min: {att_min: .4f}/{att_q_min: .4f} | att_max: {att_max: .4f}/{att_q_max: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                # total=bar.elapsed_td,
                # eta=bar.eta_td,
                loss=losses.avg,
                # top1=top1.avg,
                # top5=top5.avg,
                top1=classacc.value()[0],
                top5=classacc.value()[2],
                att_min=attention.mean(0).min().item(),
                att_max=attention.mean(0).max().item(),
                att_q_min=attention_q.mean(0).min().item(),
                att_q_max=attention_q.mean(0).max().item()
            )
        )

        # plot progress

    return (losses.avg, classacc.value()[0])


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename=args.ckptname):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, args.bestname))

def save_attention_va(model=None, testloader=None, print_info=True):
    if model is None and args.resume:

        if '100' in args.dataset:
            num_classes = 100
        else:
            num_classes = 10
        # pdb.set_trace()
        print("==> creating model '{}'".format(args.arch))
        if args.arch == 'wrn_va':
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
                K=args.K,
            )
        elif args.arch == 'wrn_va_qt':
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
                K=args.K,
                qt_num=args.qt_num,
                vq_coef=args.vq_coef,
                comit_coef=args.comit_coef,
                rd_init=args.rd_init,
                decay=args.decay,
                qt_trainable=args.qt_trainable
            )
        else:
            model = models.__dict__[args.arch](num_classes=num_classes)
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(os.path.join(args.resume, args.bestname)), 'Error: no checkpoint directory found!'
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        args.checkpoint = os.path.dirname(os.path.join(args.resume, args.bestname))
        checkpoint = torch.load(os.path.join(args.resume, args.bestname))
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('******** Best [%d] %f ********' % (start_epoch+1, best_acc))

    if testloader is None:
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print('==> Preparing dataset %s' % args.dataset)
        if args.dataset[:5] == 'cifar':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            if args.dataset == 'cifar10':
                dataloader = datasets.CIFAR10
                num_classes = 10
            else:
                dataloader = datasets.CIFAR100
                num_classes = 100

            testset = dataloader(root='../dataset', train=False, download=False,
                                 transform=transform_test)

        elif args.dataset == 'svhn':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ], )

            dataloader = datasets.SVHN
            num_classes = 10

            testset = dataloader(root='../dataset', split='test', download=False,
                                 transform=transform_test)

        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    if args.dataset[:5] == 'cifar' or args.dataset[:2] == 'cf' or args.dataset[:5]=='CIFAR':
        va_trans = transforms.Compose([
            transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
                                 (1/0.2023, 1/0.1994, 1/0.2010)),
            transforms.ToPILImage()
        ])

    elif args.dataset == 'svhn':
        va_trans = transforms.Compose([
            transforms.Normalize((-1.0, -1.0, -1.0), (1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5)),
            transforms.ToPILImage()
        ], )


    att_dir = Path(args.checkpoint.replace(args.checkpoint.split('/')[0], args.att_dir))

    if not att_dir.exists(): att_dir.mkdir(parents=True, exist_ok=True)

    AttSize = [16, 16]
    ImgSize = CIFAR_RESIZE
    # switch to evaluate mode
    model.eval()


    cnt = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):

        if use_cuda:
            # inputs, targets = inputs.cuda(), targets.cuda()
            inputs = inputs.cuda()
            targets = targets.to(inputs.device)
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        x, y = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, att_maps = model(x, num_sample=args.num_sample, train=False)
        _, predicted = torch.max(outputs.data, 1)
        flag = (predicted == y)

        att_maps_list = []
        for d_i in range(torch.cuda.device_count()):
            att_maps_list.append(att_maps[d_i * args.num_sample:(d_i + 1) * args.num_sample, :, :])
        att_maps = torch.cat(att_maps_list, dim=1)

        for b_num in range(inputs.size(0)):
            cnt = cnt + 1
            if cnt > va_num:
                break

            ori_img = va_trans(inputs[b_num])
            # scipy.misc.imsave(os.path.join(out_folder, 'test_{:04d}_{}.jpg'.format(cnt, flag[b_num])),
            #                  ori_img.detach().cpu().numpy())
            ori_img.save(os.path.join(att_dir,
                                      'test_{:04d}_cls{:02d}_{:01d}.jpg'.format(cnt, targets.data[b_num], flag[b_num])))

            for a_num in range(att_maps.size(0)):
                # for a_num in range(att_maps.size(0)//torch.cuda.device_count()):
                att_map = att_maps[a_num, b_num].view(AttSize).detach().cpu().numpy()
                tmp_map = utils_my.postprocess_prediction(att_map, size=ImgSize, print_info=print_info)
                scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_cls{:02d}_{}_att{:02d}.png'.format(
                    cnt, targets.data[b_num], flag[b_num], a_num)), tmp_map)
            # # pdb.set_trace()
            # tmp_map = utils_my.postprocess_prediction(att_maps[b_num, 0].detach().cpu().numpy(), size=ImgSize, print_info=print_info)
            # scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_cls{:02d}_{}.png'.format(
            #     cnt, targets.data[b_num], flag[b_num])), tmp_map)

        if cnt > va_num:
            break

def save_attention_va_qt(model=None, testloader=None, print_info=True):
    if model is None and args.resume:
        if args.dataset == 'cifar10' or args.dataset == 'svhn':
            num_classes = 10
        else:
            num_classes = 100

        print("==> creating model '{}'".format(args.arch))
        if args.arch == 'wrn_va':
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
                K=args.K,
            )
        elif args.arch == 'wrn_va_qt':
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
                K=args.K,
                qt_num=args.qt_num,
                vq_coef=args.vq_coef,
                comit_coef=args.comit_coef,
                rd_init=args.rd_init,
                decay=args.decay,
                qt_trainable=args.qt_trainable
            )
        else:
            model = models.__dict__[args.arch](num_classes=num_classes)

        print('==> Resuming from checkpoint..')
        assert os.path.isfile(os.path.join(args.resume, args.bestname)), 'Error: no checkpoint directory found!'
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        args.checkpoint = os.path.dirname(os.path.join(args.resume, args.bestname))
        checkpoint = torch.load(os.path.join(args.resume, args.bestname))
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('******** Best [%d] %f ********' % (start_epoch + 1, best_acc))

    if testloader is None:
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print('==> Preparing dataset %s' % args.dataset)
        if args.dataset[:5] == 'cifar':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            if args.dataset == 'cifar10':
                dataloader = datasets.CIFAR10
                num_classes = 10
            else:
                dataloader = datasets.CIFAR100
                num_classes = 100

            testset = dataloader(root='../dataset', train=False, download=False,
                                 transform=transform_test)

        elif args.dataset == 'svhn':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ], )

            dataloader = datasets.SVHN
            num_classes = 10

            testset = dataloader(root='../dataset', split='test', download=False,
                                 transform=transform_test)

        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    if args.dataset[:5] == 'cifar' or args.dataset[:2] == 'cf' or args.dataset[:5]=='CIFAR':
        va_trans = transforms.Compose([
            transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
                                 (1/0.2023, 1/0.1994, 1/0.2010)),
            transforms.ToPILImage()
        ])

    elif args.dataset == 'svhn':
        va_trans = transforms.Compose([
            transforms.Normalize((-1.0, -1.0, -1.0), (1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5)),
            transforms.ToPILImage()
        ], )


    att_dir = Path(args.checkpoint.replace(args.checkpoint.split('/')[0], args.att_dir))

    if not att_dir.exists(): att_dir.mkdir(parents=True, exist_ok=True)

    AttSize = [16, 16]
    ImgSize = CIFAR_RESIZE
    # switch to evaluate mode
    model.eval()

    cnt = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):

        if use_cuda:
            # inputs, targets = inputs.cuda(), targets.cuda()
            inputs = inputs.cuda()
            targets = targets.to(inputs.device)
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        x, y = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, att_maps, att_maps_q = model(x, num_sample=args.num_sample, train=False)
        _, predicted = torch.max(outputs.data, 1)
        flag = (predicted == y)

        att_maps_list = []
        for d_i in range(torch.cuda.device_count()):
            att_maps_list.append(att_maps[d_i * args.num_sample:(d_i + 1) * args.num_sample, :, :])
        att_maps = torch.cat(att_maps_list, dim=1)

        att_maps_q_list = []
        for d_i in range(torch.cuda.device_count()):
            att_maps_q_list.append(att_maps_q[d_i * args.num_sample:(d_i + 1) * args.num_sample, :, :])
        att_maps_q = torch.cat(att_maps_q_list, dim=1)

        for b_num in range(inputs.size(0)):
            cnt = cnt + 1
            if cnt > va_num:
                break

            ori_img = va_trans(inputs[b_num])
            # scipy.misc.imsave(os.path.join(out_folder, 'test_{:04d}_{}.jpg'.format(cnt, flag[b_num])),
            #                  ori_img.detach().cpu().numpy())
            ori_img.save(os.path.join(att_dir,
                                      'test_{:04d}_cls{:02d}_{:01d}.jpg'.format(cnt, targets.data[b_num], flag[b_num])))

            for a_num in range(att_maps.size(0)):
                # for a_num in range(att_maps.size(0)//torch.cuda.device_count()):
                att_map = att_maps[a_num, b_num].view(AttSize).detach().cpu().numpy()
                tmp_map = utils_my.postprocess_prediction(att_map, size=ImgSize, print_info=print_info)
                scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_cls{:02d}_{}_att{:02d}.png'.format(
                    cnt, targets.data[b_num], flag[b_num], a_num)), tmp_map)

            # if return att_mask_q
            for a_num in range(att_maps_q.size(0)):
                att_map = att_maps_q[a_num, b_num].view(AttSize).detach().cpu().numpy()
                # tmp_map = postprocess_prediction(att_map, size=ImgSize)
                tmp_map = cv2.resize(att_map, (ImgSize[1], ImgSize[0]), interpolation=cv2.INTER_NEAREST)
                if print_info:
                    print('max %.4f min %.4f' % (np.max(tmp_map), np.min(tmp_map)))
                scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_cls{:02d}_{}_att{:02d}_q.png'.format(
                    cnt, targets.data[b_num], flag[b_num], a_num)), tmp_map.astype('float') * 255.)
            # # pdb.set_trace()
            # tmp_map = utils_my.postprocess_prediction(att_maps[b_num, 0].detach().cpu().numpy(), size=ImgSize, print_info=print_info)
            # scipy.misc.imsave(os.path.join(att_dir, 'test_{:04d}_cls{:02d}_{}.png'.format(
            #     cnt, targets.data[b_num], flag[b_num])), tmp_map)

        if cnt > va_num:
            break

def adjust_learning_rate(optimizer, epoch):
    global state

    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma


if __name__ == '__main__':
    if args.phase == 'train':
        print('schedule:', args.schedule)
        print('gamma:', args.gamma)
        main()
    elif args.phase == 'save_att_va':
        save_attention_va()
    elif args.phase == 'save_att_va_qt':
        save_attention_va_qt()


