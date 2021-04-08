from __future__ import print_function

import os
import os.path
import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from valeodata import download

import distillation.utils as utils

# Set the appropriate paths of the datasets here.
_IMAGENET_DATASET_DIR = './datasets/IMAGENET/ILSVRC2012'
_MEAN_PIXEL = [0.485, 0.456, 0.406]
_STD_PIXEL = [0.229, 0.224, 0.225]


class ImageNetBase(data.Dataset):
    def __init__(self, split='train', transform=None):
        assert (split in ('train', 'val')) or (split.find('train_subset') != -1)
        self.split = split
        self.name = f'ImageNet_Split_' + self.split

        data_dir = download('ImageNet')
        print(f'==> Loading {dataset_name} dataset - split {self.split}')
        print(f'==> {dataset_name} directory: {data_dir}')

        self.transform = transform
        print(f'==> transform: {self.transform}')
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        split_dir = train_dir if (self.split.find('train') != -1) else val_dir
        self.data = datasets.ImageFolder(split_dir, self.transform)
        self.labels = [item[1] for item in self.data.imgs]

        if self.split.find('train_subset') != -1:
            subsetK = int(self.split[len('train_subset'):])
            assert subsetK > 0
            self.split = 'train'

            label2ind = utils.buildLabelIndex(self.data.targets)
            all_indices = []
            for label, img_indices in label2ind.items():
                assert len(img_indices) >= subsetK
                all_indices += img_indices[:subsetK]

            self.data.imgs = [self.data.imgs[idx] for idx in  all_indices]
            self.data.samples = [self.data.samples[idx] for idx in  all_indices]
            self.data.targets = [self.data.targets[idx] for idx in  all_indices]
            self.labels = [self.labels[idx] for idx in  all_indices]

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


class ImageNet(ImageNetBase):
    def __init__(self, split='train'):

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
        ])
        if split == 'val':
            transform = transform_test
        else:
            transform = transform_train
        ImageNetBase.__init__(self, split=split, transform=transform)
