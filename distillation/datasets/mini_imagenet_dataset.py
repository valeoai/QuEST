from __future__ import print_function

import os
import os.path
import random

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from valeodata import download

import distillation.utils as utils

from distillation.datasets.imagenet_dataset import ImageJitter

_MEAN_PIXEL = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
_STD_PIXEL = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]


class MiniImageNetBase(data.Dataset):
    def __init__(
        self,
        transform_test,
        transform_train,
        phase='train',
        load_single_file_split=False,
        file_split=None,
        do_not_use_random_transf=False):

        data_dir = download('MiniImagenet')
        print('==> Download MiniImageNet dataset at {0}'.format(data_dir))
        file_train_categories_train_phase = os.path.join(
            data_dir, 'miniImageNet_category_split_train_phase_train.pickle')
        file_train_categories_val_phase = os.path.join(
            data_dir, 'miniImageNet_category_split_train_phase_val.pickle')
        file_train_categories_test_phase = os.path.join(
            data_dir, 'miniImageNet_category_split_train_phase_test.pickle')
        file_val_categories_val_phase = os.path.join(
            data_dir, 'miniImageNet_category_split_val.pickle')
        file_test_categories_test_phase = os.path.join(
            data_dir, 'miniImageNet_category_split_test.pickle')

        self.phase = phase
        if load_single_file_split:
            assert file_split in ('category_split_train_phase_train',
                                  'category_split_train_phase_val',
                                  'category_split_train_phase_test',
                                  'category_split_val',
                                  'category_split_test')
            self.file_split = file_split
            self.name = 'MiniImageNet_' + file_split

            print(
                '==> Loading mini ImageNet dataset - phase {0}'
                .format(file_split))

            file_to_load = os.path.join(
                data_dir,
                'miniImageNet_{0}.pickle'.format(file_split))

            data = utils.load_pickle_data(file_to_load)
            self.data = data['data']
            self.labels = data['labels']
            self.label2ind = utils.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
        else:
            assert(phase in ('train', 'val', 'test', 'trainval')
                   or 'train_subset' in phase)
            self.name = 'MiniImageNet_' + phase


            print('Loading mini ImageNet dataset - phase {0}'.format(phase))
            if self.phase=='train':
                # Loads the training classes (and their data) as base classes
                data_train = utils.load_pickle_data(
                    file_train_categories_train_phase)
                self.data = data_train['data']
                self.labels = data_train['labels']

                self.label2ind = utils.buildLabelIndex(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)

            elif self.phase=='trainval':
                # Loads the training + validation classes (and their data) as
                # base classes
                data_train = utils.load_pickle_data(
                    file_train_categories_train_phase)
                data_val = utils.load_pickle_data(
                    file_val_categories_val_phase)
                self.data = np.concatenate(
                    [data_train['data'], data_val['data']], axis=0)
                self.labels = data_train['labels'] + data_val['labels']

                self.label2ind = utils.buildLabelIndex(
                    self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)

            elif self.phase.find('train_subset') != -1:
                subsetK = int(self.phase[len('train_subset'):])
                assert subsetK > 0
                # Loads the training classes as base classes. For each class it
                # loads only the `subsetK` first images.

                data_train = utils.load_pickle_data(
                    file_train_categories_train_phase)
                label2ind = utils.buildLabelIndex(data_train['labels'])

                all_indices = []
                for label, img_indices in label2ind.items():
                    assert len(img_indices) >= subsetK
                    all_indices += img_indices[:subsetK]

                labels_semi = [
                    data_train['labels'][idx]
                    for idx in all_indices]
                data_semi = data_train['data'][all_indices]

                self.data = data_semi
                self.labels = labels_semi

                self.label2ind = utils.buildLabelIndex(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)

                self.phase = 'train'

            elif self.phase=='val' or self.phase=='test':
                # Uses the validation / test classes (and their data) as novel
                # as novel class data and the vaditation / test image split of
                # the training classes for the base classes.

                if self.phase=='test':
                    # load data that will be used for evaluating the recognition
                    # accuracy of the base classes.
                    data_base = utils.load_pickle_data(
                        file_train_categories_test_phase)
                    # load data that will be use for evaluating the few-shot
                    # recogniton accuracy on the novel classes.
                    data_novel = utils.load_pickle_data(
                        file_test_categories_test_phase)
                else: # phase=='val'
                    # load data that will be used for evaluating the recognition
                    # accuracy of the base classes.
                    data_base = utils.load_pickle_data(
                        file_train_categories_val_phase)
                    # load data that will be use for evaluating the few-shot
                    # recogniton accuracy on the novel classes.
                    data_novel = utils.load_pickle_data(
                        file_val_categories_val_phase)

                self.data = np.concatenate(
                    [data_base['data'], data_novel['data']], axis=0)
                self.labels = data_base['labels'] + data_novel['labels']

                self.label2ind = utils.buildLabelIndex(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)

                self.labelIds_base = utils.buildLabelIndex(data_base['labels']).keys()
                self.labelIds_novel = utils.buildLabelIndex(data_novel['labels']).keys()
                self.num_cats_base = len(self.labelIds_base)
                self.num_cats_novel = len(self.labelIds_novel)
                intersection = set(self.labelIds_base) & set(self.labelIds_novel)
                assert len(intersection) == 0
            else:
                raise ValueError('Not valid phase {0}'.format(self.phase))

        self.transform_test = transform_test
        self.transform_train = transform_train
        if ((self.phase=='test' or self.phase=='val') or
            (do_not_use_random_transf==True)):
            self.transform = self.transform_test
        else:
            self.transform = self.transform_train

    def __getitem__(self, index, transform_mode=None):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if transform_mode is None:
            img = self.transform(img)
        else:
            if transform_mode == 'train':
                img = self.transform_train(img)
            else:
                img = self.transform_test(img)


        return img, label

    def __len__(self):
        return len(self.data)


class MiniImageNet(MiniImageNetBase):
    def __init__(
        self,
        phase='train',
        image_size=84,
        load_single_file_split=False,
        file_split=None,
        do_not_use_random_transf=False,
        cutout_length=0):

        normalize = transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)

        if image_size==84:

            transform_test = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])

            transform_train = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        else:
            assert image_size > 0

            transform_test = transforms.Compose([
                transforms.Resize(image_size),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,])

            transform_train = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,])


        if cutout_length > 0:
            transform_train.transforms.append(
                utils.Cutout(n_holes=1, length=cutout_length))

        MiniImageNetBase.__init__(
            self,
            transform_test=transform_test,
            transform_train=transform_train,
            phase=phase,
            load_single_file_split=load_single_file_split,
            file_split=file_split,
            do_not_use_random_transf=do_not_use_random_transf)


class MiniImageNet80x80(MiniImageNet):
    def __init__(
        self,
        phase='train',
        load_single_file_split=False,
        file_split=None,
        do_not_use_random_transf=False,
        cutout_length=0):

        MiniImageNet.__init__(
            self,
            phase=phase,
            image_size=80,
            load_single_file_split=load_single_file_split,
            file_split=file_split,
            do_not_use_random_transf=do_not_use_random_transf,
            cutout_length=cutout_length)


class MiniImageNet3x3Patches(MiniImageNetBase):
    def __init__(
        self,
        phase='train',
        load_single_file_split=False,
        file_split=None,
        do_not_use_random_transf=False,
        image_size=90,
        patch_jitter=6,
        ):

        normalize = transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)
        probability = 0.66

        def random_grayscale(image):
            if random.uniform(0, 1) <= probability:
                return transforms.functional.to_grayscale(
                    image, num_output_channels=3)
            else:
                return image

        def crop_3x3_patches(image):
            return utils.image_to_patches(
                image, is_training=False, split_per_side=3,
                patch_jitter=patch_jitter)

        def crop_3x3_patches_random_jitter(image):
            return utils.image_to_patches(
                image, is_training=True, split_per_side=3,
                patch_jitter=patch_jitter)

        normalize = transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)

        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
            crop_3x3_patches,])

        transform_train = transforms.Compose([
            random_grayscale,
            transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
            crop_3x3_patches_random_jitter,])

        MiniImageNetBase.__init__(
            self,
            transform_test=transform_test,
            transform_train=transform_train,
            phase=phase,
            load_single_file_split=load_single_file_split,
            file_split=file_split,
            do_not_use_random_transf=do_not_use_random_transf)


class MiniImageNetImagesAnd3x3Patches(MiniImageNetBase):
    def __init__(
        self,
        phase='train',
        load_single_file_split=False,
        file_split=None,
        do_not_use_random_transf=False,
        image_size=84,
        cutout_length=0,
        image_size_for_patch=96,
        patch_jitter=8):

        normalize = transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)

        if image_size==84:

            transform_test = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])

            transform_train = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        else:
            assert image_size > 0

            transform_test = transforms.Compose([
                transforms.Resize(image_size),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,])

            transform_train = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,])


        if cutout_length > 0:
            transform_train.transforms.append(
                utils.Cutout(n_holes=1, length=cutout_length))

        MiniImageNetBase.__init__(
            self,
            transform_test=transform_test,
            transform_train=transform_train,
            phase=phase,
            load_single_file_split=load_single_file_split,
            file_split=file_split,
            do_not_use_random_transf=do_not_use_random_transf)


        gray_probability = 0.66
        def random_grayscale(image):
            if random.uniform(0, 1) <= gray_probability:
                return transforms.functional.to_grayscale(
                    image, num_output_channels=3)
            else:
                return image

        def crop_3x3_patches(image):
            return utils.image_to_patches(
                image, is_training=False, split_per_side=3,
                patch_jitter=patch_jitter)

        def crop_3x3_patches_random_jitter(image):
            return utils.image_to_patches(
                image, is_training=True, split_per_side=3,
                patch_jitter=patch_jitter)

        transform_patch_test = transforms.Compose([
            transforms.Resize(image_size_for_patch),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
            crop_3x3_patches,])

        transform_patch_train = transforms.Compose([
            random_grayscale,
            transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size_for_patch),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
            crop_3x3_patches_random_jitter,])

        self.transform_patch_test = transform_patch_test
        self.transform_patch_train = transform_patch_train

        if ((self.phase=='test' or self.phase=='val') or
            (do_not_use_random_transf==True)):
            self.transform_patch = self.transform_patch_test
        else:
            self.transform_patch = self.transform_patch_train

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img_out = self.transform(img)
        patches = self.transform_patch(img)

        return img_out, patches, label


class MiniImageNetImagesAndPrtbImages(MiniImageNetBase):
    def __init__(
        self,
        phase='train',
        load_single_file_split=False,
        file_split=None,
        do_not_use_random_transf=False,
        image_size=84,
        cutout_length=0,
        cutout_n_holes=1,
        cutout_inverse=False,
        gray_probability=0.33,
        padding=8,
        only_cutout=False):

        normalize = transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)

        if image_size==84:
            transform_test = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])

            transform_train = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        else:
            assert image_size > 0

            transform_test = transforms.Compose([
                transforms.Resize(image_size),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,])

            transform_train = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,])

        MiniImageNetBase.__init__(
            self,
            transform_test=transform_test,
            transform_train=transform_train,
            phase=phase,
            load_single_file_split=load_single_file_split,
            file_split=file_split,
            do_not_use_random_transf=do_not_use_random_transf)
        print('transform_test:', transform_test)
        print('transform_train:', transform_train)

        def random_grayscale(image):
            if random.uniform(0, 1) <= gray_probability:
                return transforms.functional.to_grayscale(
                    image, num_output_channels=3)
            else:
                return image

        self.only_cutout = only_cutout
        if only_cutout:
            self.transform_cutout = utils.CutoutConstantSize(
                n_holes=cutout_n_holes,
                length=cutout_length,
                inverse=cutout_inverse)
            print('self.transform_cutout:', self.transform_cutout)
        else:
            jitter_params = {'Brightness': 0.4, 'Contrast': 0.4, 'Color': 0.4}
            transform_perturbed = transforms.Compose([
                ImageJitter(jitter_params),
                random_grayscale,
                transforms.RandomCrop(84, padding=padding),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,])

            if cutout_length > 0:
                transform_perturbed.transforms.append(
                    utils.CutoutConstantSize(
                        n_holes=cutout_n_holes, length=cutout_length))

            self.transform_perturbed = transform_perturbed
            print('self.transform_perturbed:', self.transform_perturbed)


    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_cutout:
            img = self.transform(img)
            return img, self.transform_cutout(img), label
        else:
            return self.transform(img), self.transform_perturbed(img), label


class MiniImageNetImagesAndPrtbImagesv2(MiniImageNetBase):
    def __init__(
        self,
        phase='train',
        load_single_file_split=False,
        file_split=None,
        do_not_use_random_transf=False,
        image_size=84,
        cutout_length=0,
        cutout_n_holes=1,
        gray_probability=0.33,
        padding=8):

        normalize = transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)

        assert image_size > 0

        transform_test = transforms.Compose([
            transforms.Resize(image_size)])

        transform_train = transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),])

        MiniImageNetBase.__init__(
            self,
            transform_test=transform_test,
            transform_train=transform_train,
            phase=phase,
            load_single_file_split=load_single_file_split,
            file_split=file_split,
            do_not_use_random_transf=do_not_use_random_transf)

        padding = 8
        gray_probability = 0.33

        def random_grayscale(image):
            if random.uniform(0, 1) <= gray_probability:
                return transforms.functional.to_grayscale(
                    image, num_output_channels=3)
            else:
                return image

        self.transform_post = transforms.Compose([
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,])

        jitter_params = {'Brightness': 0.4, 'Contrast': 0.4, 'Color': 0.4}
        self.transform_post_perturbed = transforms.Compose([
            ImageJitter(jitter_params),
            random_grayscale,
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
            utils.CutoutConstantSize(n_holes=cutout_n_holes, length=cutout_length),])

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.transform(img)

        return self.transform_post(img), self.transform_post_perturbed(img), label


def load_features_labels(dataset_file):
    data_file = h5py.File(dataset_file, 'r')
    count = data_file['count'][0]
    features = data_file['all_features'][...]
    labels = data_file['all_labels'][:count].tolist()
    features = features[:count,:]

    return data_file, count, features, labels


class MiniImageNetFeatures(data.Dataset):
    def __init__(self, data_directory, phase='train'):
        file_train_categories_train_phase = os.path.join(
            data_directory,
            'MiniImageNet_category_split_train_phase_train.json')
        file_train_categories_val_phase = os.path.join(
            data_directory,
            'MiniImageNet_category_split_train_phase_val.json')
        file_train_categories_test_phase = os.path.join(
            data_directory,
            'MiniImageNet_category_split_train_phase_test.json')
        file_val_categories_val_phase = os.path.join(
            data_directory,
            'MiniImageNet_category_split_val.json')
        file_test_categories_test_phase = os.path.join(
            data_directory,
            'MiniImageNet_category_split_test.json')

        self.phase = phase
        assert phase in ('train', 'val', 'test', 'trainval', 'trainval_test')
        self.name = 'MiniImageNetFeatures_Phase_' + phase

        print('Loading mini ImageNet dataset - phase {0}'.format(phase))
        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            _, _, features, labels = load_features_labels(
                file_train_categories_train_phase)

            self.features = features
            self.labels = labels

            self.label2ind = utils.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase=='trainval':
            _, _, features_train, labels_train = load_features_labels(
                file_train_categories_train_phase)
            _, _, features_val, labels_val = load_features_labels(
                file_val_categories_val_phase)

            self.features = np.concatenate(
                [features_train, features_val], axis=0)
            self.labels = labels_train + labels_val

            self.label2ind = utils.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)
        elif self.phase=='val' or self.phase=='test':

            if self.phase=='test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                _, _, base_features, base_labels = load_features_labels(
                    file_train_categories_test_phase)

                # load data that will be use for evaluating the few-shot
                # recogniton accuracy on the novel categories.
                _, _, novel_features, novel_labels = load_features_labels(
                    file_test_categories_test_phase)
            else: # phase=='val'
                # load data that will be used for evaluating the
                # recognition accuracy of the base categories.
                _, _, base_features, base_labels = load_features_labels(
                    file_train_categories_val_phase)

                # load data that will be use for evaluating the few-shot
                # recogniton accuracy on the novel categories.
                _, _, novel_features, novel_labels = load_features_labels(
                    file_val_categories_val_phase)

            self.features = np.concatenate(
                [base_features, novel_features], axis=0)
            self.labels = base_labels + novel_labels

            self.label2ind = utils.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = utils.buildLabelIndex(base_labels).keys()
            self.labelIds_novel = utils.buildLabelIndex(novel_labels).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0
        elif self.phase=='trainval_test':
            # load data that will be use for evaluating the few-shot recogniton
            # accuracy on the novel categories.
            _, _, novel_features, novel_labels = load_features_labels(
                file_test_categories_test_phase)

            self.features = novel_features
            self.labels = novel_labels

            self.label2ind = utils.buildLabelIndex(self.labels)
            num_trainval_classes = 80
            assert num_trainval_classes <= min(novel_labels)
            for label in range(num_trainval_classes):
                self.label2ind[label] = []

            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = {
                label for label in range(num_trainval_classes)}
            self.labelIds_novel = utils.buildLabelIndex(novel_labels).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0
            print('TRAINVAL_TEST phase')
            self.phase = 'test'
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

    def __getitem__(self, index):
        features_this = torch.Tensor(self.features[index]).view(-1,1,1)
        label_this = self.labels[index]
        return features_this, label_this

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import torchvision
    from distillation.dataloaders.basic_dataloaders import OnlyImageDataloader
    #dataset = MiniImageNet(phase='train')

    dataset = MiniImageNetImagesAndPrtbImages(
        phase='train',
        image_size=80,
        cutout_length=0,
        cutout_n_holes=0,
        cutout_inverse=False,
        gray_probability=0.33,
        padding=8,
        only_cutout=False)
    breakpoint()
    dloader = OnlyImageDataloader(dataset, batch_size=32, train=True)


    """
    for i in range(5):
        x, y = dataset[i]
        print(x.size())
        #x = utils.standardize_image(x)
        #x_grid = torchvision.utils.make_grid(
        #    x, nrow=3, padding=5, normalize=True)
        #torchvision.utils.save_image(
        #    x_grid, './temp_{0}.jpg'.format(i))
    """

    #apply_mixcut(images, labels=None, beta_val=1.0)

    for i, batch in enumerate(dloader()):
        x, xp = batch

        xp, lambda_val, bbox, _ = utils.apply_mixcut(
            xp, labels=None, beta_val=1.0)
        print('x:', x.size())
        print('lambda_val:', lambda_val)
        print('bbox:', bbox)

        x_all = torch.cat([x, xp], dim=0)
        x_grid = torchvision.utils.make_grid(
            x_all, nrow=8, padding=5, normalize=True)
        torchvision.utils.save_image(
            x_grid, './temp_x_dloader_{0}.jpg'.format(i))

        if i >= 10:
            break
