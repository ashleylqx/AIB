'''
    Dataloader for frequency images
    Folder: ./cifar-10-hfc
    Modified from https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
'''

from PIL import Image
import os
import os.path
import numpy as np
import pickle
import pdb
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        return_high (bool, optional): If true, return high-frequency images, else, return
            low-frequence images
        r (int): radius of frequence filter
    """
    base_folder = 'cifar-10-hfc'
    # url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    # filename = "cifar-10-python.tar.gz"
    # tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    # train_list = [
    #     ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    #     ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    #     ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    #     ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    #     ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    # ]
    #
    # test_list = [
    #     ['test_batch', '40351d587109b95175f43aff81a1287e'],
    # ]
    # meta = {
    #     'filename': 'batches.meta',
    #     'key': 'label_names',
    #     'md5': '5ff9c542aee3614f3951f8cda6e48888',
    # }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            # download: bool = False,
            return_high: bool=True,
            r: int = 4,
            return_img: bool=False
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.return_high = return_high
        self.r = r
        self.return_img = return_img
        # if download:
        #     self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        # if self.train:
        #     downloaded_list = self.train_list
        # else:
        #     downloaded_list = self.test_list

        # self.data: Any = []
        # self.data = []
        # self.targets = []

        # now load the picked numpy arrays
        # for file_name, checksum in downloaded_list:
        #     file_path = os.path.join(self.root, self.base_folder, file_name)
        #     with open(file_path, 'rb') as f:
        #         entry = pickle.load(f, encoding='latin1')
        #         self.data.append(entry['data'])
        #         if 'labels' in entry:
        #             self.targets.extend(entry['labels'])
        #         else:
        #             self.targets.extend(entry['fine_labels'])

        if train:
            pass
        else:
            self.targets = np.load(os.path.join(root, self.base_folder, 'test_label.npy'))
            if self.return_high:
                data = np.load(os.path.join(root, self.base_folder, 'test_data_high_%d.npy' % self.r)).transpose((0, 3, 1, 2))
            else:
                data = np.load(os.path.join(root, self.base_folder, 'test_data_low_%d.npy' % self.r)).transpose((0, 3, 1, 2))

        # map data to int [0, 255] or float [0, 1]
        #  if the array has a shape of (height, width, 3) it automatically assumes it's an RGB image and
        #  expects it to have a dtype of uint8! In your case,
        #  however, you have an RBG image with float values from 0 to 1.
        num_data, num_channel = data.shape[0], data.shape[1]
        self.data = (data - data.reshape(num_data, num_channel, -1).min(axis=-1, keepdims=True)[...,np.newaxis])
        self.data = self.data/(self.data.reshape(num_data, num_channel, -1).max(axis=-1, keepdims=True)[...,np.newaxis]+np.finfo(float).eps) * 255.0
        self.data = self.data.astype(np.uint8)

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        # pdb.set_trace()

        # self._load_meta()

    # def _load_meta(self) -> None:
    #     path = os.path.join(self.root, self.base_folder, self.meta['filename'])
    #     if not check_integrity(path, self.meta['md5']):
    #         raise RuntimeError('Dataset metadata file not found or corrupted.' +
    #                            ' You can use download=True to download it')
    #     with open(path, 'rb') as infile:
    #         data = pickle.load(infile, encoding='latin1')
    #         self.classes = data[self.meta['key']]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.return_img:
            ori_img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_img:
            return img, target, ori_img
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    # def _check_integrity(self) -> bool:
    #     root = self.root
    #     for fentry in (self.train_list + self.test_list):
    #         filename, md5 = fentry[0], fentry[1]
    #         fpath = os.path.join(root, self.base_folder, filename)
    #         if not check_integrity(fpath, md5):
    #             return False
    #     return True
    #
    # def download(self) -> None:
    #     if self._check_integrity():
    #         print('Files already downloaded and verified')
    #         return
    #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-hfc'
