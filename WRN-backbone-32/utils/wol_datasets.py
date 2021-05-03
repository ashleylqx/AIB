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
import json
from PIL import Image

from torchvision import datasets
# from caltech_my import Caltech101, Caltech256
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
from kornia.enhance.zca import ZCAWhitening

from .config import *

# *** MS_COCO use one transform, CUG&ILSVRC use another transform

class MS_COCO(Dataset):
    def __init__(self, root, mode='train', return_path=False, N=None,
                 img_h = COCO_RESIZE[0], img_w = COCO_RESIZE[1], transform=None): #'train', 'test', 'val'

        self.path_dataset = root
        self.path_images = os.path.join(self.path_dataset, mode+'2014')
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        self.transform = transform

        # self.normalize_feature = normalize_feature

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        # if mode=='train':
        #     list_names = np.array(['COCO_train2014_000000001108',
        #                            'COCO_train2014_000000002148',
        #                            'COCO_train2014_000000003348',
        #                            'COCO_train2014_000000004575'])
        # elif mode=='val':
        #     list_names = np.array(['COCO_val2014_000000005586',
        #                            'COCO_val2014_000000011122',
        #                            'COCO_val2014_000000016733',
        #                            'COCO_val2014_000000022199'])
        #
        # self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]


        # self.coco = COCO(os.path.join(PATH_COCO, 'annotations', 'instances_%s2014.json'%mode))
        self.imgNsToCat = pickle.load(open(os.path.join(PATH_COCO, 'imgNsToCat_{}.p'.format(mode)), "rb"))

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init MS_COCO full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
        image = scipy.misc.imread(rgb_ima, mode='RGB')
        image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA).astype(np.float32)

        if self.transform is not None:
            img_processed = self.transform(image/255.)
        else:
            img_processed = transforms.ToTensor()(image)

        # get coco label
        label_indices = self.imgNsToCat[self.list_names[index]]
        # label_indices = self.coco.imgNsToCat[self.list_names[index]]
        label = torch.zeros(coco_num_classes)
        if len(label_indices)>0:
            label[label_indices] = 1
        else:
            label[0] = 1

        if self.return_path:
            return img_processed, label, self.list_names[index]
        else:
            return img_processed, label


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


# ILSVRC
class ILSVRC(Dataset):
    def __init__(self, root, mode='train', return_path=False, N=None,
                 transform=None, onehot_label=False, num_classes=ilsvrc_classes): #'train', 'test', 'val', num_tgt_cls=ilsvrc_classes

        # self.num_tgt_cls = num_tgt_cls
        self.mode = mode
        self.path_dataset = root # PATH_ILSVRC
        self.path_images = os.path.join(self.path_dataset, 'images', self.mode) # rearrange folder
        self.return_path = return_path
        self.transform = transform
        self.onehot_label = onehot_label
        self.num_classes = num_classes


        # get list images
        with open(os.path.join(self.path_dataset, 'lists/%s_list.txt'%self.mode), 'r') as f:
            tmp = f.readlines()
        list_names = [l.split(' ')[0] for l in tmp] # .JPEG
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        labels = [int(l.split(' ')[1]) for l in tmp]
        labels = np.array(labels)
        self.labels = labels
        if N is not None:
            self.list_names = list_names[:N]
            self.labels = labels[:N]

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init ILSVRC dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        # img_name = self.list_names[index]
        img_name = os.path.join(self.path_images, self.list_names[index]+'.JPEG')
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
            return image, gt_label, self.list_names[index]
        else:
            return image, gt_label

        # # Image and saliency map paths
        # rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.JPEG')
        # image = scipy.misc.imread(rgb_ima, mode='RGB')
        #
        #
        # image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        # img_processed = self.transform(image / 255.)
        #
        # label = self.labels[index]
        #
        # if self.return_path:
        #     return img_processed, label, self.list_names[index]
        # else:
        #     return img_processed, label


# ==== collate_fn for handling grayscale images in batch of RGB images ====
def collate_fn_caltech(batch): # This does not work when normalization transform contains 3-dim mean and std.
    images = list()
    labels = list()
    # pdb.set_trace()
    for i, X in enumerate(batch):
        print('X[0]', X[0].size(0))
        # if X[0].size(0) < 3:
        #     tmp_img = X[0].repeat(3, 1, 1)
        #     print('tmp_img', tmp_img.size())
        #     images.append(tmp_img.unsqueeze(0))
        if X[0].size(0) == 3:
            images.append(X[0].unsqueeze(0))
        else:
            images.append(X[0].unsqueeze(0).repeat(1, 3, 1, 1))

        labels.append(X[1])
        # labels.append(X[1].unsqueeze(0))

    images_batch = torch.cat(images, dim=0)
    # images_batch = torch.cat(labels, dim=0)
    labels_batch = torch.tensor(labels)

    return images_batch, labels_batch


# ==== weakly object segmentation ====
# Object Discovery; generate list from folder
class ObjectDiscovery(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root, return_path=False, N=None,
                 transform=None, onehot_label=False, num_classes=cub_classes): #'train', 'test'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.path_dataset = root # PATH_EVENT
        # self.path_images = self.path_dataset
        self.path_images = os.path.join(self.path_dataset, 'Data')
        self.return_path = return_path
        self.datalist_file = os.path.join(self.path_dataset, 'image_list.txt')
        list_names, labels = self.read_labeled_image_list(self.datalist_file)
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

        print("Init ObjectDiscovery dataset ...")
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        img_name = os.path.join(self.path_images, self.list_names[index])
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
            # return image, gt_label, img_name.split('/')[-1].split('.')[0]
            return image, gt_label, self.list_names[index]
        else:
            return image, gt_label

    def read_labeled_image_list(self, data_list):
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
            # img_name_list.append(os.path.join(data_dir, image))
            img_name_list.append(image)
            img_labels.append(np.asarray(labels))
        return img_name_list, img_labels

# ==== other datasets used in cross-dataset classification ====
# STL_train (pytorch) shuffle=False, index follow the order; generate list from dataloader

# STL_test (pytorch) shuffle=False, index follow the order; generate list from dataloader

# caltech-101 (pytorch) shuffle=False, index follow the order; generate list from dataloader

# caltech-256 (pytorch) shuffle=False, index follow the order; generate list from dataloader

# Event-8; generate list from folder
class Event8(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root, return_path=False, N=None,
                 transform=None, onehot_label=False, num_classes=cub_classes): #'train', 'test'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.path_dataset = root # PATH_EVENT
        self.path_images = self.path_dataset
        # self.path_images = os.path.join(self.path_dataset, 'images')
        self.return_path = return_path
        self.datalist_file = os.path.join(self.path_dataset, 'image_list.txt')
        list_names, labels = self.read_labeled_image_list(self.datalist_file)
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

        print("Init Event-8 dataset ...")
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        img_name = os.path.join(self.path_images, self.list_names[index])
        assert os.path.exists(img_name), 'file {} not exits'.format(img_name)
        image = Image.open(img_name).convert('RGB') # try not use PIL Image, not use ToTensor before ZCA tomorrow
        # image = scipy.misc.imread(img_name, mode='RGB')
        # image = cv2.resize(image, tuple(CIFAR_RESIZE), interpolation=cv2.INTER_LINEAR)
        # # # image = image.astype('float32')
        # # # image = cv2.resize(image, (input_h, input_w), interpolation=cv2.INTER_LINEAR)
        # image = torch.tensor(image, dtype=torch.float32)
        # zca = ZCAWhitening().fit(image)
        # image = zca(image)
        # image = image.numpy()

        if self.transform is not None:
            image = self.transform(image)

        if self.onehot_label:
            gt_label = np.zeros(self.num_classes, dtype=np.float32)
            gt_label[self.labels[index].astype(int)] = 1
        else:
            gt_label = self.labels[index].astype(np.long)

        if self.return_path:
            # return image, gt_label, img_name.split('/')[-1].split('.')[0]
            return image, gt_label, self.list_names[index]
        else:
            return image, gt_label

    def read_labeled_image_list(self, data_list):
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
            # img_name_list.append(os.path.join(data_dir, image))
            img_name_list.append(image)
            img_labels.append(np.asarray(labels))
        return img_name_list, img_labels

# Action-40; generate list from folder
class Action40(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root, return_path=False, N=None,
                 transform=None, onehot_label=False, num_classes=cub_classes): #'train', 'test'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.path_dataset = root # PATH_EVENT
        # self.path_images = self.path_dataset
        self.path_images = os.path.join(self.path_dataset, 'JPEGImages')
        self.return_path = return_path
        self.datalist_file = os.path.join(self.path_dataset, 'image_list.txt')
        list_names, labels = self.read_labeled_image_list(self.datalist_file)
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

        print("Init Action-40 dataset ...")
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        img_name = os.path.join(self.path_images, self.list_names[index])
        assert os.path.exists(img_name), 'file {} not exits'.format(img_name)
        image = Image.open(img_name).convert('RGB')
        # image = scipy.misc.imread(img_name, mode='RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.onehot_label:
            gt_label = np.zeros(self.num_classes, dtype=np.float32)
            gt_label[self.labels[index].astype(int)] = 1
        else:
            gt_label = self.labels[index].astype(np.long)

        if self.return_path:
            # return image, gt_label, img_name.split('/')[-1].split('.')[0]
            return image, gt_label, self.list_names[index]
        else:
            return image, gt_label

    def read_labeled_image_list(self, data_list):
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
            # img_name_list.append(os.path.join(data_dir, image))
            img_name_list.append(image)
            img_labels.append(np.asarray(labels))
        return img_name_list, img_labels


# Scene-67; generate list from folder
class Scene67(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root, return_path=False, N=None,
                 transform=None, onehot_label=False, num_classes=cub_classes): #'train', 'test'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.path_dataset = root # PATH_EVENT
        # self.path_images = self.path_dataset
        self.path_images = os.path.join(self.path_dataset, 'Images')
        self.return_path = return_path
        self.datalist_file = os.path.join(self.path_dataset, 'image_list.txt')
        list_names, labels = self.read_labeled_image_list(self.datalist_file)
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

        print("Init Scene-67 dataset ...")
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        img_name = os.path.join(self.path_images, self.list_names[index])
        assert os.path.exists(img_name), 'file {} not exits'.format(img_name)
        image = Image.open(img_name).convert('RGB')
        # image = scipy.misc.imread(img_name, mode='RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.onehot_label:
            gt_label = np.zeros(self.num_classes, dtype=np.float32)
            gt_label[self.labels[index].astype(int)] = 1
        else:
            gt_label = self.labels[index].astype(np.long)

        if self.return_path:
            # return image, gt_label, img_name.split('/')[-1].split('.')[0]
            return image, gt_label, self.list_names[index]
        else:
            return image, gt_label

    def read_labeled_image_list(self, data_list):
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
                if len(line.strip().split(' ')) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else: # if image name contains space
                    line = line.strip().split(' ')
                    image = ' '.join(line[:-1])
                    # labels = map(int, line[1:])
                    labels = int(line[-1])
            # img_name_list.append(os.path.join(data_dir, image))
            img_name_list.append(image)
            img_labels.append(np.asarray(labels))
        return img_name_list, img_labels




# Tiny Imagenet
class TinyImagenet(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root, mode='train', return_path=False, N=None,
                 transform=None, onehot_label=False, num_classes=cub_classes): #'train', 'test'
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.path_dataset = root # PATH_CUB?
        self.mode = mode
        self.path_images = os.path.join(self.path_dataset, self.mode)
        self.return_path = return_path

        self.transform = transform
        self.onehot_label = onehot_label
        self.num_classes = num_classes

        label_indices = range(self.num_classes)
        # this might be faster than reading folders from disk?
        wnids_txt = os.path.join(self.path_dataset, 'wnids.txt')
        with open(wnids_txt) as f:
            # label_wnids = f.readlines()
            label_wnids = f.read().splitlines()
        self.label_dict = dict(zip(label_wnids, label_indices))

        if self.mode == 'train':
            list_names = []
            labels = []
            for label in label_wnids:
                bbox_txt = os.path.join(self.path_images, label, label+'_boxes.txt')
                # pdb.set_trace()
                with open(bbox_txt) as f:
                    # lines = f.readlines()
                    # names = [os.path.join(self.path_images, label, 'images', l.split(' ')[0]) for l in lines]
                    lines = f.read().splitlines()
                    names = [os.path.join(self.path_images, label, 'images', l.split('\t')[0]) for l in lines]
                    list_names.extend(names)
                    labels.extend([self.label_dict[label]]*len(names))

        elif self.mode == 'val':
            anno_txt = os.path.join(self.path_images, 'val_annotations.txt')
            # pdb.set_trace()
            with open(anno_txt) as f:
                # lines = f.readlines()
                # names = [os.path.join(self.path_images, 'images', l.split(' ')[0]) for l in lines]
                # lbs = [self.label_dict[l.split(' ')[1]] for l in lines]
                lines = f.read().splitlines()
                list_names = [os.path.join(self.path_images, 'images', l.split('\t')[0]) for l in lines]
                labels = [self.label_dict[l.split('\t')[1]] for l in lines]

        assert max(labels) == (self.num_classes-1)
        list_names = np.array(list_names)
        self.list_names = list_names
        labels = np.array(labels)
        self.labels = labels

        if N is not None:
            self.list_names = list_names[:N]
            self.labels = labels[:N]

        print("Init Tiny ImageNet dataset in mode {}".format(mode))
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




if __name__ == "__main__":

    # transformation for training set
    tencrop = True
    print(tencrop == True)
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    # input_size = (256, 256)
    # crop_size = (224, 224) # ILSVRC, CUB

    input_size = (80, 80)
    crop_size = (80, 80) # CUB_crop

    tsfm_train = transforms.Compose([transforms.Resize(input_size),  # 256
                                     transforms.RandomCrop(crop_size),  # 224
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)])

    if tencrop:
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in
                                    crops])),
                           ]
    else:
        func_transforms = [transforms.Resize(input_size),
                           # transforms.Resize(crop_size),
                           transforms.CenterCrop(crop_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean_vals, std_vals), ]
    tsfm_clstest = transforms.Compose(func_transforms)

    # transformation for test loc set
    tsfm_loctest = transforms.Compose([transforms.Resize(crop_size),  # 224
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean_vals, std_vals)])

    # test ILSVRC datasets
    # # ds_train = ILSVRC(root=PATH_ILSVRC, N=4, return_path=True, mode='train', transform=tsfm_train)  # OK image, label, (, image_name)
    # ds_train = ILSVRC(root=PATH_ILSVRC, N=4, return_path=True, mode='val', transform=tsfm_clstest)  # OK image, label, (, image_name)

    # test CUB datasets
    # ds_train = CUB(root=PATH_CUB, N=4, return_path=True, mode='train', transform=tsfm_train)  # OK image, label, (, image_name)
    # ds_train = CUB(root=PATH_CUB, N=4, return_path=True, mode='test', transform=tsfm_clstest)  # OK image, label, (, image_name)

    # test CUB_crop datasets
    # ds_train = CUB_crop(root=PATH_CUB, N=4, return_path=True, mode='train', transform=tsfm_train)  # OK image, label, (, image_name)
    # ds_train = CUB_crop(root=PATH_CUB, N=4, return_path=True, mode='test', transform=tsfm_clstest)  # OK image, label, (, image_name)

    # test Caltech datasets
    # caltech_transforms = transforms.Compose([transforms.Resize(crop_size),  # 224
    #                                    transforms.ToTensor(),])
    #
    # # ds_train = datasets.Caltech101(PATH_CALTECH101, download=False, transform=caltech_transforms)
    # # train_dataloader = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn_caltech)
    # ds_train = Caltech101(PATH_CALTECH101, download=False, transform=tsfm_train)

    # test Event-8 dataset
    # ds_train = Event8(root=PATH_EVENT, N=4, return_path=True, transform=tsfm_train)  # OK image, label, (, image_name)

    # test Action-40 dataset
    # ds_train = Action40(root=PATH_ACTION, N=4, return_path=True, transform=tsfm_train)  # OK image, label, (, image_name)

    # test Scene-67 dataset
    # ds_train = Scene67(root=PATH_SCENE, N=4, return_path=True, transform=tsfm_train)  # OK image, label, (, image_name)

    # test ObjectDiscovery dataset
    # ds_train = ObjectDiscovery(root=PATH_OD, N=4, return_path=True, transform=tsfm_train)  # OK image, label, (, image_name)

    # test TinyImagenet dataset
    ds_train = TinyImagenet(root=PATH_TINYIM, N=4, return_path=True, mode='train', transform=tsfm_train)  # OK image, label, (, image_name)
    # ds_train = TinyImagenet(root=PATH_TINYIM, N=4, return_path=True, mode='val', transform=tsfm_train)  # OK image, label, (, image_name)


    train_dataloader = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=2)



    for i, X in enumerate(train_dataloader):
        print(i)
        print('images', X[0].size())
        print('labels', X[1].size())
        print('image_names', X[-1])
        # if i>10:
        #     break
