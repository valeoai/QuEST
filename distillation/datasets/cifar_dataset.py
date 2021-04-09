from __future__ import print_function

import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image


_CIFAR_DATASET_DIR = './datasets/CIFAR'
_CIFAR_CATEGORY_SPLITS_DIR = './data/cifar-fs_splits'
_CIFAR_MEAN_PIXEL  = [0.5071, 0.4867, 0.4408]#[x/255.0 for x in [125.3, 123.0, 113.9]]
_CIFAR_STD_PIXEL = [0.2675, 0.2565, 0.2761]  #[x/255.0 for x in [63.0, 62.1, 66.7]]


class CIFARbase(data.Dataset):
    def __init__(
        self,
        data_dir=_CIFAR_DATASET_DIR,
        split='train',
        transform_train=None,
        transform_test=None,
        version='CIFAR100'):

        assert split in ('train', 'val')
        self.split = split
        self.name = version + '_' + split

        self.transform_test = transform_test
        self.transform_train = transform_train
        if self.split is not 'train':
            self.transform = self.transform_test
        else:
            self.transform = self.transform_train

        print(self.transform)

        self.data = datasets.__dict__[version](
            data_dir,
            train=(self.split=='train'),
            download=True,
            transform=self.transform)
        self.labels = self.data.targets
        self.images = self.data.data

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)


class CIFAR100(CIFARbase):
    def __init__(
        self,
        data_dir=_CIFAR_DATASET_DIR,
        split='train',
        do_not_use_random_transf=False):
        normalize = transforms.Normalize(mean=_CIFAR_MEAN_PIXEL, std=_CIFAR_STD_PIXEL)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        if do_not_use_random_transf:
            transform_train = transform_test
        CIFARbase.__init__(
            self,
            data_dir=data_dir,
            split=split,
            transform_train=transform_train,
            transform_test=transform_test,
            version='CIFAR100')
