import os
import pdb
import sys
import cv2
import pickle
import scipy.misc
import scipy.io
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import numpy as np
import random
from PIL import Image

# import torch.nn.functional as F
# from torch_geometric.data import Data, Batch
import torch.nn.functional as F

# --- coco api --------------
import json
import time
from collections import defaultdict
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import *

# CUB
def get_name_id(name_path):
    name_id = name_path.strip().split('/')[-1]
    name_id = name_id.strip().split('.')[0]
    return name_id

class CUB(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root, mode='train', return_path=False, N=None,
                 transform=None, onehot_label=False, num_classes=cub_classes): #'train', 'test'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.path_dataset = root # PATH_CUB?
        self.mode = mode
        # self.path_images = os.path.join(self.path_dataset, mode + '2014')
        self.path_images = os.path.join(self.path_dataset, 'images')
        self.return_path = return_path
        self.datalist_file = os.path.join(self.path_dataset, 'lists/%s_list.txt'%self.mode)
        list_names, labels = self.read_labeled_image_list(self.path_images, self.datalist_file)
        self.transform = transform
        self.onehot_label = onehot_label
        self.num_classes = num_classes

        self.trainFlag = False

        list_names = np.array(list_names)
        labels = np.array(labels)
        self.list_names = list_names
        self.labels = labels

        if N is not None:
            self.list_names = list_names[:N]
            self.labels = labels[:N]

        print("Init CUB_200_2011 dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        img_name = self.list_names[index]
        assert os.path.exists(img_name), 'file {} not exits'.format(img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.onehot_label:
            gt_label = np.zeros(self.num_classes, dtype=np.float32)
            gt_label[self.labels[index].astype(int)] = 1
        else:
            gt_label = self.labels[index].astype(np.long)

        if self.return_path:
            return image, gt_label, img_name.split('/')[-1].split('.')[0]
        else:
            return image, gt_label

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(np.asarray(labels))
        return img_name_list, img_labels

class CUB_crop(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root, mode='train', return_path=False, N=None,
                 transform=None, onehot_label=False, num_classes=cub_classes): #'train', 'test'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.path_dataset = root # PATH_CUB?
        self.mode = mode
        # self.path_images = os.path.join(self.path_dataset, mode + '2014')
        self.path_images = os.path.join(self.path_dataset, 'images')
        self.return_path = return_path
        self.datalist_file = os.path.join(self.path_dataset, 'lists/%s_crop_list.txt'%self.mode)
        list_names, labels, bboxes = self.read_labeled_image_crop_list(self.path_images, self.datalist_file)
        self.transform = transform
        self.onehot_label = onehot_label
        self.num_classes = num_classes

        self.trainFlag = False

        list_names = np.array(list_names)
        labels = np.array(labels)
        bboxes = np.array(bboxes)
        self.list_names = list_names
        self.labels = labels
        self.bboxes = bboxes

        if N is not None:
            self.list_names = list_names[:N]
            self.labels = labels[:N]
            self.bboxes = bboxes[:N]

        print("Init CUB_200_2011 crop dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        img_name = self.list_names[index]
        assert os.path.exists(img_name), 'file {} not exits'.format(img_name)
        image = Image.open(img_name).convert('RGB')
        # print(image.size)
        # print(self.bboxes[index])
        # print(self.labels)
        # print(self.list_names)
        # pdb.set_trace()
        # crop image using gt bbox <x> <y> <width> <height>
        # (It will not change orginal image)
        left, top, width, height = self.bboxes[index]
        right = left + width
        bottom = top + height
        image = image.crop((left, top, right, bottom))
        # image = image.crop((int(left), int(top), int(right), int(bottom)))

        if self.transform is not None:
            image = self.transform(image)

        if self.onehot_label:
            gt_label = np.zeros(self.num_classes, dtype=np.float32)
            gt_label[self.labels[index].astype(int)] = 1
        else:
            gt_label = self.labels[index].astype(np.long)

        if self.return_path:
            return image, gt_label, img_name.split('/')[-1].split('.')[0]
        else:
            return image, gt_label

    def read_labeled_image_crop_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images, ground truth labels and bounding boxes.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and labels, respectively.
          And an array of bounding boxes.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        img_bboxes = []

        for line in f:

            line = line.strip().split()
            image = line[0]
            labels = int(line[1])
            bbox = json.loads('[%s]' % (','.join(line[2:])))
            # pdb.set_trace()
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(np.asarray(labels))
            img_bboxes.append(bbox)

        return img_name_list, img_labels, img_bboxes
        # return img_name_list, img_labels, np.array(img_bboxes)
