from __future__ import print_function

import json
import os
import os.path

import h5py
import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from valeodata import download

import distillation.utils as utils

# Set the appropriate paths of the datasets here.
_IMAGENET_DATASET_DIR = './datasets/IMAGENET/ILSVRC2012'
_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH = './data/IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS.json'
_MEAN_PIXEL = [0.485, 0.456, 0.406]
_STD_PIXEL = [0.229, 0.224, 0.225]

INTERP = 3


def load_ImageNet_fewshot_split(class_names, version=1):
    with open(_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH, 'r') as f:
        label_idx = json.load(f)

    assert len(label_idx['label_names']) == len(class_names)

    def get_class_indices(class_indices1):
        class_indices2 = []
        for index in class_indices1:
            class_name_this = label_idx['label_names'][index]
            assert class_name_this in class_names
            class_indices2.append(class_names.index(class_name_this))

        class_names_tmp1 = [
            label_idx['label_names'][index] for index in class_indices1]
        class_names_tmp2 = [class_names[index] for index in class_indices2]

        assert class_names_tmp1 == class_names_tmp2

        return class_indices2

    if version == 1:
        base_classes = label_idx['base_classes']
        base_classes_val = label_idx['base_classes_1']
        base_classes_test = label_idx['base_classes_2']
        novel_classes_val = label_idx['novel_classes_1']
        novel_classes_test = label_idx['novel_classes_2']
    elif version == 2:
        base_classes = get_class_indices(label_idx['base_classes'])
        base_classes_val = get_class_indices(label_idx['base_classes_1'])
        base_classes_test = get_class_indices(label_idx['base_classes_2'])
        novel_classes_val = get_class_indices(label_idx['novel_classes_1'])
        novel_classes_test = get_class_indices(label_idx['novel_classes_2'])

    return (base_classes,
            base_classes_val, base_classes_test,
            novel_classes_val, novel_classes_test)


class ImageJitter:
    def __init__(self, transformdict):
        transformtypedict=dict(
            Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
            Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color
        )
        self.transforms = [
            (transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


class ImageNetBase(data.Dataset):
    def __init__(self, split='train', size256=False, transform=None):

        dataset_name = 'ImageNet256' if size256 else 'ImageNet'
        assert (split in ('train', 'val')) or (split.find('train_subset') != -1)
        self.split = split
        self.name = f'{dataset_name}_Split_' + self.split

        data_dir = download(dataset_name)
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
    def __init__(
        self,
        split='train',
        use_geometric_aug=True,
        use_simple_geometric_aug=False,
        use_color_aug=True,
        cutout_length=0,
        do_not_use_random_transf=False,
        size256=False):

        # use_geometric_aug: If True geometric augmentations are used for the
        # images of the training split.
        # use_color_aug: if True color augmentations are used for the images
        # of the test/val split.

        transform_train = []
        assert not (use_simple_geometric_aug and use_geometric_aug)
        if use_geometric_aug:
            transform_train.append(transforms.RandomResizedCrop(224))
            transform_train.append(transforms.RandomHorizontalFlip())
        elif use_simple_geometric_aug:
            transform_train.append(transforms.Resize(256))
            transform_train.append(transforms.RandomCrop(224))
            transform_train.append(transforms.RandomHorizontalFlip())
        else:
            transform_train.append(transforms.Resize(256))
            transform_train.append(transforms.CenterCrop(224))

        if use_color_aug:
            jitter_params = {'Brightness': 0.4, 'Contrast': 0.4, 'Color': 0.4}
            transform_train.append(ImageJitter(jitter_params))

        transform_train.append(lambda x: np.asarray(x))
        transform_train.append(transforms.ToTensor())
        transform_train.append(
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL))

        if cutout_length > 0:
            print('==> cutout_length: {0}'.format(cutout_length))
            transform_train.append(
                utils.Cutout(n_holes=1, length=cutout_length))

        transform_train = transforms.Compose(transform_train)
        self.transform_train = transform_train

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
        ])

        if do_not_use_random_transf or split == 'val':
            transform = transform_test
        else:
            transform = transform_train

        ImageNetBase.__init__(
            self, split=split, size256=size256, transform=transform)


class ImageNetWithPrtbImages(ImageNetBase):
    def __init__(
        self,
        split='train',
        do_not_use_random_transf=False,
        cutout_length=0,
        cutout_n_holes=0,
        cutout_inverse=False,
        gray_probability=0.33,
        size256=False,
        type=1):

        ImageNetBase.__init__(
            self, split=split, size256=size256, transform=None)

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
        ])

        print(f'==> type: {type}')
        if type == 1:
            self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
            ])
        elif type == 2:
            self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
            ])

        def random_grayscale(image):
            if random.uniform(0, 1) <= gray_probability:
                return transforms.functional.to_grayscale(
                    image, num_output_channels=3)
            else:
                return image

        self.transform_perturbed = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ImageJitter({'Brightness': 0.4, 'Contrast': 0.4, 'Color': 0.4}),
            random_grayscale,
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
        ])

        if cutout_length > 0:
            self.transform_perturbed.transforms.append(
                utils.CutoutConstantSize(
                    n_holes=cutout_n_holes,
                    length=cutout_length,
                    inverse=cutout_inverse))

        if do_not_use_random_transf or split == 'val':
            self.transform = self.transform_test
        else:
            self.transform = self.transform_train

        print(f'==> transform: {self.transform}')
        print(f'==> transform_test: {self.transform_test}')
        print(f'==> transform_train: {self.transform_train}')
        print(f'==> transform_perturbed: {self.transform_perturbed}')

    def __getitem__(self, index):
        img, label = self.data[index]
        return self.transform(img), self.transform_perturbed(img), label


class ImageNetWithPrtbImagesV2(ImageNetBase):
    def __init__(
        self,
        split='train',
        do_not_use_random_transf=False,
        cutout_length=0,
        cutout_n_holes=0,
        size256=False,
        same_hflip=False,
        type=1):

        ImageNetBase.__init__(
            self, split=split, size256=size256, transform=None)

        self.same_hflip = same_hflip
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)

        post_transform = transforms.Compose([
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)])

        rand_crop = transforms.RandomResizedCrop(
            224, scale=(0.3, 1.0), ratio=(0.7, 1.4), interpolation=INTERP)

        color_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)

        rnd_gray = transforms.RandomGrayscale(p=0.25)

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            post_transform])

        self.transform_perturbed = transforms.Compose([
            rand_crop,
            color_jitter,
            rnd_gray,
            post_transform])

        print(f'==> type: {type}')
        if type == 1:
            self.transform_train = self.transform_test
        elif type == 2:
            self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                post_transform])
        elif type == 3:
            self.transform_train = self.transform_perturbed

        if cutout_length > 0:
            self.transform_perturbed.transforms.append(
                utils.CutoutConstantSize(
                    n_holes=cutout_n_holes,
                    length=cutout_length,
                    inverse=False))

        if do_not_use_random_transf or split == 'val':
            self.transform = self.transform_test
        else:
            self.transform = self.transform_train

        print(f'==> transform: {self.transform}')
        print(f'==> transform_test: {self.transform_test}')
        print(f'==> transform_train: {self.transform_train}')
        print(f'==> transform_perturbed: {self.transform_perturbed}')

    def __getitem__(self, index):
        img, label = self.data[index]
        if self.same_hflip:
            img = self.hflip(img)
            return self.transform(img), self.transform_perturbed(img), label
        else:
            img1 = self.hflip(img)
            img2 = self.hflip(img)
            return self.transform(img1), self.transform_perturbed(img2), label


class ImageNet3x3Patches(data.Dataset):
    def __init__(
        self,
        split='train',
        resize=296,
        crop_size=255,
        patch_jitter=21):
        # use_geometric_aug: If True geometric augmentations are used for the
        # images of the training split.
        # use_color_aug: if True color augmentations are used for the images
        # of the test/val split.

        self.split = split
        assert split in ('train', 'val')
        self.name = 'ImageNet3x3Patches_Split_' + split

        data_dir = download('ImageNet')
        print('==> Loading ImageNet dataset - split {0}'.format(split))
        print('==> ImageNet directory: {0}'.format(data_dir))

        probability = 0.66

        def random_grayscale(image):
            if random.uniform(0, 1) <= probability:
                return transforms.functional.to_grayscale(
                    image, num_output_channels=3)
            else:
                return image

        def crop_3x3_patches(image):
            return utils.image_to_patches(
                image,
                is_training=False,
                split_per_side=3,
                patch_jitter=patch_jitter)

        def crop_3x3_patches_random_jitter(image):
            return utils.image_to_patches(
                image,
                is_training=True,
                split_per_side=3,
                patch_jitter=patch_jitter)

        normalize = transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)

        transform_train = transforms.Compose([
            random_grayscale,
            transforms.RandomHorizontalFlip(),
            transforms.Resize((resize,resize)),
            transforms.RandomCrop(crop_size),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
            crop_3x3_patches_random_jitter,])

        transform_test = transforms.Compose([
            transforms.Resize((resize,resize)),
            transforms.CenterCrop(crop_size),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
            crop_3x3_patches,])

        self.trainsform_train = transform_train

        self.transform = transform_train if split=='train' else transform_test
        print('==> transform: {0}'.format(self.transform))
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        split_dir = train_dir if split=='train' else val_dir
        self.data = datasets.ImageFolder(split_dir, self.transform)
        self.labels = [item[1] for item in self.data.imgs]

    #@profile
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


class ImageNetLowShot(ImageNet):
    def __init__(
        self,
        phase='train',
        split='train',
        do_not_use_random_transf=False,
        version=1):

        assert phase in ('train', 'test', 'val')
        assert split in ('train', 'val')
        assert version == 1 or version == 2

        use_aug = (phase=='train') and (do_not_use_random_transf==False)

        ImageNet.__init__(
            self, split=split, use_geometric_aug=use_aug, use_color_aug=use_aug)

        self.phase = phase
        self.split = split
        self.name = 'ImageNetLowShot_Phase_' + phase + '_Split_' + split
        print('==> Loading ImageNet dataset (for few-shot benchmark) - phase {0}'.
            format(phase))
        if version == 2:
            self.name += '_version2'
        #***********************************************************************
        (base_classes, _, _, novel_classes_val, novel_classes_test) = (
            load_ImageNet_fewshot_split(self.data.classes, version=version))
        #***********************************************************************

        self.label2ind = utils.buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert self.num_cats == 1000

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)
        if self.phase=='val' or self.phase=='test':
            self.labelIds_novel = (
                novel_classes_val if (self.phase=='val') else
                novel_classes_test)
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0


class ImageNetLowShot3x3Patches(ImageNet3x3Patches):
    def __init__(
        self,
        phase='train',
        split='train',
        resize=296,
        crop_size=255,
        patch_jitter=21,
        do_not_use_random_transf=False,
        version=1):

        assert phase in ('train', 'test', 'val')
        assert split in ('train', 'val')
        assert version == 1 or version == 2

        ImageNet3x3Patches.__init__(
            self,
            split=split,
            resize=resize,
            crop_size=crop_size,
            patch_jitter=patch_jitter)

        self.phase = phase
        self.split = split
        self.name = 'ImageNetLowShot3x3Patches_Phase_' + phase + '_Split_' + split
        if version == 2:
            self.name += '_version2'
        print('==> Loading ImageNet dataset (for few-shot benchmark) - phase {0}'.
            format(phase))

        #***********************************************************************
        (base_classes, _, _, novel_classes_val, novel_classes_test) = (
            load_ImageNet_fewshot_split(self.data.classes, version=version))
        #***********************************************************************

        self.label2ind = utils.buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert self.num_cats==1000

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)
        if self.phase=='val' or self.phase=='test':
            self.labelIds_novel = (
                novel_classes_val if (self.phase=='val') else
                novel_classes_test)
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0


class ImageNetLowShotFeatures:
    def __init__(
        self,
        data_dir, # path to the directory with the saved ImageNet features.
        image_split='train', # the image split of the ImageNet that will be loaded.
        phase='train', # whether the dataset will be used for training, validating, or testing a model.
        version=1):
        assert image_split in ('train', 'val')
        assert phase in ('train', 'val', 'test')
        assert version == 1 or version == 2

        self.phase = phase
        self.image_split = image_split
        self.name = (f'ImageNetLowShotFeatures_ImageSplit_{self.image_split}'
                     f'_Phase_{self.phase}')
        if version == 2:
            self.name += '_version2'

        dataset_file = os.path.join(
            data_dir, 'ImageNet_' + self.image_split + '.h5')
        self.data_file = h5py.File(dataset_file, 'r')
        self.count = self.data_file['count'][0]
        self.features = self.data_file['all_features'][...]
        self.labels = self.data_file['all_labels'][:self.count].tolist()

        #***********************************************************************
        # with open(_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH, 'r') as f:
        #    label_idx = json.load(f)
        # base_classes = label_idx['base_classes']
        # base_classes_val = label_idx['base_classes_1']
        # base_classes_test = label_idx['base_classes_2']
        # novel_classes_val = label_idx['novel_classes_1']
        # novel_classes_test = label_idx['novel_classes_2']
        #***********************************************************************
        data_tmp = datasets.ImageFolder(
            os.path.join(download('ImageNet'), 'train'), None)
        (base_classes, base_classes_val, base_classes_test,
         novel_classes_val, novel_classes_test) = (
            load_ImageNet_fewshot_split(data_tmp.classes, version=version))
        #***********************************************************************

        self.label2ind = utils.buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert self.num_cats==1000

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)

        if self.phase=='val' or self.phase=='test':
            self.labelIds_novel = (
                novel_classes_val if (self.phase=='val') else
                novel_classes_test)
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)
            self.base_classes_eval_split = (
                base_classes_val if (self.phase=='val') else
                base_classes_test)
            self.base_classes_subset = self.base_classes_eval_split


    def __getitem__(self, index):
        features_this = torch.Tensor(self.features[index]).view(-1,1,1)
        label_this = self.labels[index]
        return features_this, label_this

    def __len__(self):
        return int(self.count)


class ImageNetFeatures:
    def __init__(
        self,
        data_dir, # path to the directory with the saved ImageNet features.
        split='train', # the image split of the ImageNet that will be loaded.
        ):
        assert split in ('train', 'val')

        self.split = split
        self.name = (f'ImageNetFeatures_ImageSplit_{self.split}')

        dataset_file = os.path.join(
            data_dir, 'ImageNet_' + self.split + '.h5')
        self.data_file = h5py.File(dataset_file, 'r')
        self.count = self.data_file['count'][0]
        self.features = self.data_file['all_features'][...]
        self.labels = self.data_file['all_labels'][:self.count].tolist()

        self.label2ind = utils.buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert self.num_cats == 1000

    def __getitem__(self, index):
        features_this = torch.Tensor(self.features[index]).view(-1,1,1)
        label_this = self.labels[index]
        return features_this, label_this

    def __len__(self):
        return int(self.count)


if __name__ == '__main__':
    import torchvision
    from distillation.dataloaders.basic_dataloaders import OnlyImageDataloader
    #dataset = ImageNet(split='train_subset260')
    #breakpoint()
    dataset = ImageNetWithPrtbImagesV2(
        split='train_subset260',
        do_not_use_random_transf=False,
        cutout_length=0,
        cutout_n_holes=0,
        type=1,
        same_hflip=False,
        size256=True)

    dloader = OnlyImageDataloader(dataset, batch_size=32, train=True)

    for i, batch in enumerate(dloader()):
        x, x_prtb = batch
        print(f'x: {x.size()}')
        print(f'x_prtb: {x_prtb.size()}')

        x_all = torch.cat([x, x_prtb], dim=0)
        x_grid = torchvision.utils.make_grid(
            x_all, nrow=8, padding=8, normalize=True)
        torchvision.utils.save_image(
            x_grid, './temp_x_dloader_{0}.jpg'.format(i))

        if i >= 10:
            break
    #dataset = ImageNetLowShot(split='train', phase='train')

    #for i in range(10):
    #    x, y = dataset[i]
    #    image = utils.convertImgFromNormalizedTensorToUint8Numpy(
    #        x, _MEAN_PIXEL, _STD_PIXEL)
    #
    #    image = Image.fromarray(image).convert('RGB')
    #    image.save('./imagenet_train_{0}_image_nc.png'.format(i))

    #from matplotlib import pyplot as plt
    #plt.figure(1)
    #plt.imshow(dataset.inv_transform(dataset[0][0]))
    #plt.show()
