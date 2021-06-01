import torch, os
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import random
import pdb

from config import *
from my_datasets import CUB, CUB_crop

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    train_batch = args.train_batch * torch.cuda.device_count()
    test_batch = args.test_batch * torch.cuda.device_count()
    dset_ratio = args.dset_ratio

    if 'CUB' == name:
         # transformation for training set
         mean_vals = [0.485, 0.456, 0.406]
         std_vals = [0.229, 0.224, 0.225]
         input_size = (256, 256)
         crop_size = (224, 224)

         tsfm_train = transforms.Compose([transforms.Resize(input_size),  # 256
                                          transforms.RandomCrop(crop_size),  # 224
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean_vals, std_vals)])

         if args.tencrop:
             func_transforms = [transforms.Resize(input_size),
                                transforms.TenCrop(crop_size),
                                transforms.Lambda(
                                    lambda crops: torch.stack(
                                        [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop
                                         in crops])),
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
         root = PATH_CUB
         train_kwargs = {'root':root, 'mode':'train', 'transform':tsfm_train}
         test_kwargs = {'root':root,'mode':'test','transform':tsfm_clstest}
         loctest_kwargs = {'root':root,'mode':'test','transform':tsfm_loctest,'return_path':True} #
         dset = CUB

    elif 'CUB_crop' == name:
         # transformation for training set
         mean_vals = [0.485, 0.456, 0.406]
         std_vals = [0.229, 0.224, 0.225]
         input_size = (256, 256)
         crop_size = (224, 224)

         tsfm_train = transforms.Compose([transforms.Resize(input_size),  # 256
                                          transforms.RandomCrop(crop_size),  # 224
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean_vals, std_vals)])

         if args.tencrop:
             func_transforms = [transforms.Resize(input_size),
                                transforms.TenCrop(crop_size),
                                transforms.Lambda(
                                    lambda crops: torch.stack(
                                        [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop
                                         in crops])),
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
         root = PATH_CUB
         train_kwargs = {'root':root, 'mode':'train', 'transform':tsfm_train}
         test_kwargs = {'root':root,'mode':'test','transform':tsfm_clstest}
         loctest_kwargs = {'root':root,'mode':'test','transform':tsfm_loctest,'return_path':True} #
         dset = CUB_crop

    else: raise UnknownDatasetError()

    train_data = dset(**train_kwargs)

    train_loader = DataLoader(train_data,
                                batch_size=train_batch,
                                shuffle=True,
                                num_workers=4,
                                drop_last=True)

    test_data = dset(**test_kwargs)
    test_loader = DataLoader(test_data,
                                batch_size=test_batch,
                                shuffle=False,
                                num_workers=4,
                                drop_last=False)

    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader

    return data_loader

