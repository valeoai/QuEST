from __future__ import print_function

import json
import os
import os.path

import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

import distillation.utils as utils

from distillation.datasets.imagenet_dataset import _MEAN_PIXEL
from distillation.datasets.imagenet_dataset import _STD_PIXEL

# Set the appropriate paths of the datasets here.
_MIT67_DATASET_DIR = '/datasets_local/MITScenes67'


class MaskedRandomSampler(data.Sampler):
    """
    Samples elements from selected indices,
    sampling is repeated s.t. size of one epoch is equal to
    vanilla sampling.
    """

    def __init__(self, mask, rep_num):
        self.mask = mask
        self.rep_num = rep_num

    def __iter__(self):
        randperms = []
        for _ in range(self.rep_num):
            random.shuffle(self.mask)
            randperms.extend(self.mask)

        return iter(randperms)

    def __len__(self):
        return len(self.mask)*self.rep_num


def get_dataset_subsplit(data, num_examples, split_ratio, repeat=False):
    label2ind = utils.buildLabelIndex(data.targets)
    all_indices = []
    for label, img_indices in label2ind.items():
        indices = np.where(np.array(data.targets)==label)[0].tolist()
        assert img_indices == indices

        if num_examples is None:
            start_idx = int(len(img_indices)*split_ratio[0])
            end_idx = int(len(img_indices)*split_ratio[1])
        else:
            assert num_examples <= len(img_indices)
            start_idx = int(num_examples*split_ratio[0])
            end_idx = int(num_examples*split_ratio[1])

        img_indices = img_indices[start_idx:end_idx]
        all_indices += img_indices

    rep_num = max(1, int(len(data)/len(all_indices))) if repeat else 1
    sampler = MaskedRandomSampler(list(range(len(all_indices))), rep_num)

    if repeat and (rep_num > 1):
        all_indices = all_indices * rep_num

    data.imgs = [data.imgs[idx] for idx in  all_indices]
    data.samples = [data.samples[idx] for idx in  all_indices]
    data.targets = [data.targets[idx] for idx in  all_indices]

    return data, sampler


class AbstractDataset(data.Dataset):
    def __init__(
        self,
        data_dir=_MIT67_DATASET_DIR,
        split="train",
        dataset_name="MITScenes67",
        num_examples=None,
        transform=None):
        assert dataset_name in ("MITScenes67",)
        assert (split in ("train", "val", "trainval", "test"))
        self.split = split
        self.name = f"{dataset_name}_Split_{self.split}"
        if (num_examples is not None) and (split in ("train", "val", "trainval")):
            self.name += f"_NumExamples_{num_examples}"

        print(f"==> Loading {dataset_name} dataset - split {self.split}")
        print(f"==> {dataset_name} directory: {data_dir}")

        self.transform = transform
        print(f"==> transform: {self.transform}")
        split_dir = "test" if (split == "test") else "train"
        split_dir = os.path.join(data_dir, split_dir)
        print(f"==> split images in {split_dir}")
        self.data = datasets.ImageFolder(split_dir, self.transform)

        self.split_ratios = {
            "train": [0.0, 0.8],
            "val": [0.8, 1.0],
            "trainval": [0.0, 1.0],
            "test": [0.0, 1.0],
        }

        if self.split in ("train", "val", "trainval"):
            data, sampler = get_dataset_subsplit(
                data=self.data,
                num_examples=num_examples,
                split_ratio=self.split_ratios[self.split],
                repeat=(self.split == "train"))
            self.data = data
            self.sampler = sampler

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


class MITScenes(AbstractDataset):
    def __init__(
        self,
        data_dir=_MIT67_DATASET_DIR,
        split="train",
        num_examples=None,
        do_not_use_random_transf=False):

        normalize = transforms.Normalize(
            mean=_MEAN_PIXEL,
            std=_STD_PIXEL,
        )
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        if do_not_use_random_transf or (split == "test"):
            transform = transform_test
        else:
            transform = transform_train

        AbstractDataset.__init__(
            self, data_dir=data_dir, split=split, dataset_name="MITScenes67",
            num_examples=num_examples, transform=transform)
