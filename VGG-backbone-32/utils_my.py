from nested_dict import nested_dict
from functools import partial
import torch
from torch.nn.init import kaiming_normal_
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

import pdb

import numpy as np
import math
import cv2
from config import *

def str2bool(v):
    """
    codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor

def postprocess_prediction(prediction, size=None, print_info=True, ostu_th=False):
    """
    Postprocess saliency maps by resizing and applying gaussian blurringself.
    args:
        prediction: numpy array with saliency postprocess_prediction
        size: original (H,W) of the image
    returns:
        numpy array with saliency map normalized 0-255 (int8)
    """
    if print_info:
        print('max %.4f min %.4f'%(np.max(prediction), np.min(prediction))) # l1 norm is much larger than l2? but maps are similar

    prediction = prediction - np.min(prediction)

    # prediction = prediction - np.mean(prediction)
    # prediction[prediction<0] = 0

    # print('max %.4f min %.4f'%(np.max(prediction), np.min(prediction))) # l1 norm is much larger than l2? but maps are similar
    if np.max(prediction) != 0:
        saliency_map = (prediction/np.max(prediction) * 255).astype(np.uint8)
    else:
        saliency_map = prediction.astype(np.uint8)

    if size is None:
        size = MNIST_RESIZE

    # resize back to original size
    saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0)
    saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)

    # clip again
    # saliency_map = np.clip(saliency_map, 0, 255)
    saliency_map = saliency_map - np.min(saliency_map)
    if np.max(saliency_map)!=0:
        saliency_map = saliency_map.astype('float') / np.max(saliency_map) * 255.
    else:
        print('Zero saliency map.')

    if ostu_th:
        _, th2 = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ret2, th2 = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th2

    return saliency_map


def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    # l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_kl = F.kl_div(p, q, reduction='sum') * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)

def distillation_my(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    # l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_kl = F.kl_div(p, q, reduction='sum') * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels).div(math.log(2)) # divide log(2)
    return l_kl * alpha + l_ce * (1. - alpha)


def at_my(x):
    return F.normalize(x.pow(2).mean(1))

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

# TODO: add variants for this function for va
def at_loss(x, y):
    # pdb.set_trace()
    if y.size()[-2:] != x.size()[-2:]:
        y = F.interpolate(y, x.size()[-2:])
    # y = y.view(y.size(0), -1)
    return (at(x) - at(y)).pow(2).mean()


def at_loss_my_new(x, y):
    # pdb.set_trace()
    if y.size()[-2:] != x.size()[-2:]:
        y = F.interpolate(y, x.size()[-2:])

    return (x - y).pow(2).mean()

# def kl_divergence(self, latent_space1, latent_space2):
#     kl_div = kl.kl_divergence(latent_space1, latent_space2)
#     return kl_div
def at_loss_my_dist(s, t):
    return torch.mean(kl.kl_divergence(s, t))


def at_loss_my(x, y):
    # pdb.set_trace()
    if y.size()[-2:] != x.size()[-2:]:
        y = F.interpolate(y, x.size()[-2:])
    y = y.view(y.size(0), -1)

    # y = (y-y.min()) / (y.max()+1e-8)
    # tmp_x = at(x)
    # tmp_x = (tmp_x-tmp_x.min()) / (tmp_x.max()+1e-8) # _norm
    # return (tmp_x - y).pow(2).mean()
    return (x - y).pow(2).mean()


# def at_loss_my(x, y):
#     # pdb.set_trace()
#     if y.size()[-2:] != x.size()[-2:]:
#         y = F.interpolate(y, x.size()[-2:])
#     y = y.view(y.size(0), -1)
#     y = y * 0.25 # _d4
#     return (at(x) - y).pow(2).mean()




def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}


def data_parallel(f, input, params, mode, device_ids, output_device=None):
    device_ids = list(device_ids)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    # pdb.set_trace()
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True
