from __future__ import print_function

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

import distillation.utils as utils
from distillation.datasets.imagenet_dataset import ImageJitter


_CIFAR_DATASET_DIR = './datasets/CIFAR'
_CIFAR_CATEGORY_SPLITS_DIR = './data/cifar-fs_splits'
_CIFAR_MEAN_PIXEL  = [0.5071, 0.4867, 0.4408]#[x/255.0 for x in [125.3, 123.0, 113.9]]
_CIFAR_STD_PIXEL = [0.2675, 0.2565, 0.2761]  #[x/255.0 for x in [63.0, 62.1, 66.7]]


class CIFARbase(data.Dataset):
    def __init__(
        self,
        split='train',
        transform_train=None,
        transform_test=None,
        version='CIFAR100',
        do_not_use_random_transf=False):

        assert split in ('train', 'val')
        self.split = split
        self.name = version + '_' + split

        self.transform_test = transform_test
        self.transform_train = transform_train
        if (self.split is not 'train' or do_not_use_random_transf==True):
            self.transform = self.transform_test
        else:
            self.transform = self.transform_train

        print(self.transform)

        self.data = datasets.__dict__[version](
            _CIFAR_DATASET_DIR,
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

'''
class CIFAR(CIFARbase):
    def __init__(
        self,
        split='train',
        version='CIFAR100',
        do_not_use_random_transf=False,
        cutout_length=0,
        cutout_n_holes=1):

        normalize = transforms.Normalize(
            mean=_CIFAR_MEAN_PIXEL, std=_CIFAR_STD_PIXEL)
        transform_test = transforms.Compose([
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        if cutout_length > 0:
            transform_train.transforms.append(
                utils.Cutout(n_holes=cutout_n_holes, length=cutout_length))

        CIFARbase.__init__(
            self,
            split=split,
            transform_train=transform_train,
            transform_test=transform_test,
            version=version,
            do_not_use_random_transf=do_not_use_random_transf)


class CIFARv2(CIFARbase):
    def __init__(
        self,
        split='train',
        version='CIFAR100',
        do_not_use_random_transf=False,
        gray_probability=0.33,
        cutout_length=0,
        cutout_n_holes=1):

        normalize = transforms.Normalize(
            mean=_CIFAR_MEAN_PIXEL,
            std=_CIFAR_STD_PIXEL)

        transform_test = transforms.Compose([
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        def random_grayscale(image):
            if random.uniform(0, 1) <= gray_probability:
                return transforms.functional.to_grayscale(
                    image, num_output_channels=3)
            else:
                return image

        jitter_params = {'Brightness': 0.4, 'Contrast': 0.4, 'Color': 0.4}

        transform_train = transforms.Compose([
            ImageJitter(jitter_params),
            random_grayscale,
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,])

        if cutout_length > 0:
            transform_train.transforms.append(
                utils.Cutout(n_holes=cutout_n_holes, length=cutout_length))

        CIFARbase.__init__(
            self,
            split=split,
            transform_train=transform_train,
            transform_test=transform_test,
            version=version,
            do_not_use_random_transf=do_not_use_random_transf)
'''

class CIFARv3(CIFARbase):
    def __init__(
        self,
        split='train',
        version='CIFAR100',
        do_not_use_random_transf=False,
        cutout_length=0,
        cutout_n_holes=1):

        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761])
        transform_test = transforms.Compose([
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        if cutout_length > 0:
            transform_train.transforms.append(
                utils.Cutout(n_holes=cutout_n_holes, length=cutout_length))

        CIFARbase.__init__(
            self,
            split=split,
            transform_train=transform_train,
            transform_test=transform_test,
            version=version,
            do_not_use_random_transf=do_not_use_random_transf)
