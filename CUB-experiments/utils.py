import torch
from torch import nn
from torch.autograd import Variable
import cv2
import numpy as np

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


class Weight_EMA_Update(object):

    def __init__(self, model, initial_state_dict, decay=0.999):
        self.model = model
        self.model.load_state_dict(initial_state_dict, strict=True)
        self.decay = decay

    def update(self, new_state_dict):
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = (self.decay)*state_dict[key] + (1-self.decay)*new_state_dict[key]
            #state_dict[key] = (1-self.decay)*state_dict[key] + (self.decay)*new_state_dict[key]

        self.model.load_state_dict(state_dict)


def postprocess_prediction(prediction, size=None):
    """
    Postprocess saliency maps by resizing and applying gaussian blurringself.
    args:
        prediction: numpy array with saliency postprocess_prediction
        size: original (H,W) of the image
    returns:
        numpy array with saliency map normalized 0-255 (int8)
    """
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

    # saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    # saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0)

    # clip again
    # saliency_map = np.clip(saliency_map, 0, 255)
    if np.max(saliency_map)!=0:
        saliency_map = saliency_map.astype('float') / np.max(saliency_map) * 255.
    else:
        print('Zero saliency map.')

    return saliency_map

