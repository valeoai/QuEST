from __future__ import print_function

import csv
import os
import os.path

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from valeodata import download

import distillation.utils as utils

# Set the appropriate paths of the datasets here.
_TIERED_MINI_IMAGENET_DATASET_DIR = './datasets/tieredMiniImageNet'

_TIERED_MIN_IMAGENET_METADATA_DIR = './data/tiered_imagenet_split'
_MIN_IMAGENET_METADATA_DIR = './data/mini_imagenet_split'

def get_labels_mini_imagenet_valtest():
    data_dir_mini= download('MiniImagenet')
    print('==> Download MiniImageNet dataset at {0}'.format(data_dir_mini))
    file_train_categories_train_phase = os.path.join(
        data_dir_mini, 'miniImageNet_category_split_train_phase_train.pickle')
    file_val_categories_val_phase = os.path.join(
        data_dir_mini, 'miniImageNet_category_split_val.pickle')
    file_test_categories_test_phase = os.path.join(
        data_dir_mini, 'miniImageNet_category_split_test.pickle')

    data_train_mini = utils.load_pickle_data(file_train_categories_train_phase)
    data_val_mini = utils.load_pickle_data(file_val_categories_val_phase)
    data_test_mini = utils.load_pickle_data(file_test_categories_test_phase)

    label2classname_train = [None] * 64
    for cname, label in data_train_mini['catname2label'].items():
        label2classname_train[label] = cname.decode()

    label2classname_val = [
        cname.decode() for cname in data_val_mini['label2catname']]
    label2classname_test = [
        cname.decode() for cname in data_test_mini['label2catname']]


    train_val_test_classes = (
        label2classname_train + label2classname_val + label2classname_test)
    train_val_test_classes = [[x,] for x in train_val_test_classes]
    write_csv_class_file(
        './data/mini_imagenet_split/train_val_test_classnames.csv',
        train_val_test_classes)

    train_val_test_classes2 = read_csv_class_file(
        './data/mini_imagenet_split/train_val_test_classnames.csv')
    return train_val_test_classes


def read_csv_class_file(filename):
    data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            #print('==> Line {0}: content: {1}'.format(line_count, row))
            line_count += 1
            data.append(row)

    return data


def write_csv_class_file(filename, data):
    with open(filename, mode='w') as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            csv_writer.writerow(row)


def get_train_class_names_of_tiered_mini_imagenet():
    file_train_class_meta_data = os.path.join(
        _TIERED_MIN_IMAGENET_METADATA_DIR, 'train.csv')
    meta_data_train = read_csv_class_file(file_train_class_meta_data)
    train_class_names = [class_name for (class_name, _) in meta_data_train]
    return train_class_names


def get_all_class_names_of_mini_imagenet():
    file_train_class_meta_data = os.path.join(
        _MIN_IMAGENET_METADATA_DIR, 'train_val_test_classnames.csv')
    meta_data_all = read_csv_class_file(file_train_class_meta_data)
    all_class_names = [class_name for (class_name,) in meta_data_all]
    return all_class_names


def tiered_train_classes_not_miniimagenet():
    tiered_classes = get_train_class_names_of_tiered_mini_imagenet()
    mini_classes = get_all_class_names_of_mini_imagenet()
    print('==> tiered-MiniImageNet train classes: {0}'.format(tiered_classes))
    print('==> MiniImageNet all classes: {0}'.format(mini_classes))

    tiered_classes_not_mini_imagenet = [True] * len(tiered_classes)
    counter = 0
    for i, t, in enumerate(tiered_classes):
        if t in mini_classes:
            tiered_classes_not_mini_imagenet[i] = False
            counter += 1

    print(
        '==> {0} classes removed from tiered-MiniImageNet train classes.'
        .format(counter))

    return tiered_classes_not_mini_imagenet

class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class tieredMiniImageNet(data.Dataset):
    def __init__(
        self,
        phase='train',
        load_single_file_split=False,
        do_not_use_random_transf=False):

        data_dir = download('tieredMiniImageNet')
        print(data_dir)

        self.base_folder = 'tieredMiniImageNet'
        class_splits = ['train', 'val', 'test']
        files_images = {}
        files_labels = {}
        for class_split in class_splits:
            files_images[class_split] = os.path.join(
                data_dir, '{0}_images_png.pkl'.format(class_split))
            # TODO: maybe decompress it and save it in np format.
            files_labels[class_split] = os.path.join(
                data_dir, '{0}_labels.pkl'.format(class_split))

        self.phase = phase
        if load_single_file_split:
            assert self.phase in ('train', 'val', 'test')

            self.name = 'tieredMiniImagenet_' + self.phase
            print('Loading tiered Mini ImageNet {0}'.format(self.phase))

            images = utils.load_pickle_data(files_images[self.phase])
            labels = utils.load_pickle_data(files_labels[self.phase])
            labels = labels['label_specific'].tolist()

            self.data = images
            self.labels = labels
            self.label2ind = utils.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

        else:
            assert phase in ('train', 'trainval', 'train_not_miniimagnet',
                             'val', 'test')

            self.name = 'tieredMiniImagenet_' + phase
            print('Loading tiered Mini ImageNet {0}'.format(phase))

            labels_train = utils.load_pickle_data(files_labels['train'])
            num_train_classes = labels_train['label_specific'].max() + 1

            labels_val = utils.load_pickle_data(files_labels['val'])
            num_val_classes = labels_val['label_specific'].max() + 1

            labels_test = utils.load_pickle_data(files_labels['test'])
            num_test_classes = labels_test['label_specific'].max() + 1
            print('==> tiered MiniImageNet:')
            print('====> number of train classes : {0}'.format(num_train_classes))
            print('====> number of validation classes : {0}'.format(num_val_classes))
            print('====> number of test classes : {0}'.format(num_test_classes))
            if self.phase=='train':
                # During training phase we only load the training phase images
                # of the training categories (aka base categories).
                images_train = utils.load_pickle_data(files_images['train'])
                labels_train_list = labels_train['label_specific'].tolist()
                self.data = images_train
                self.labels = labels_train_list

                self.label2ind = utils.buildLabelIndex(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)
            elif self.phase=='train_not_miniimagnet':
                images_train = utils.load_pickle_data(files_images['train'])
                labels_train_list = labels_train['label_specific'].tolist()
                # filter out classes that are in the test set or validation set
                # of mini imagenet.
                label2ind = utils.buildLabelIndex(labels_train_list)
                valid_tiered_classes = tiered_train_classes_not_miniimagenet()

                keep_indices = []
                for label, img_indices in label2ind.items():
                    if valid_tiered_classes[label]:
                        keep_indices += img_indices

                keep_labels_train_list = [
                    labels_train_list[i] for i in keep_indices]
                keep_images_train = [images_train[i] for i in keep_indices]

                self.data = keep_images_train
                self.labels = keep_labels_train_list
                self.label2ind = utils.buildLabelIndex(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)
            elif self.phase=='trainval':
                # During training phase we only load the training phase images
                # of the training categories (aka base categories).
                images_train = utils.load_pickle_data(files_images['train'])
                labels_train_list = labels_train['label_specific'].tolist()

                images_val = utils.load_pickle_data(files_images['val'])
                labels_val_list = (
                    labels_val['label_specific'] + num_train_classes).tolist()

                self.data = images_train + images_val
                self.labels = labels_train_list + labels_val_list

                self.label2ind = utils.buildLabelIndex(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)

            elif self.phase=='val' or self.phase=='test':

                class_split = 'val' if self.phase == 'val' else 'test'
                images_eval = utils.load_pickle_data(files_images[class_split])
                labels_eval = utils.load_pickle_data(files_labels[class_split])
                labels_eval_list = (
                    labels_eval['label_specific'] + num_train_classes).tolist()

                self.data = images_eval
                self.labels = labels_eval_list

                self.label2ind = utils.buildLabelIndex(self.labels)
                for label in range(num_train_classes):
                    self.label2ind[label] = []

                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = {label for label in range(num_train_classes)}
                self.labelIds_novel = utils.buildLabelIndex(labels_eval_list).keys()
                self.num_cats_base = len(self.labelIds_base)
                self.num_cats_novel = len(self.labelIds_novel)
                intersection = set(self.labelIds_base) & set(self.labelIds_novel)
                assert len(intersection) == 0
            else:
                raise ValueError('Not valid phase {0}'.format(self.phase))

        print('==> {0} images were loaded.'.format(len(self.labels)))

        mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase=='test' or self.phase=='val') or (do_not_use_random_transf==True):
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])

        self.inv_transform = transforms.Compose([
            lambda x: x.clone(),
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])


    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #breakpoint()
        img = cv2.cvtColor(cv2.imdecode(img, 1), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class tieredMiniImageNet80x80(tieredMiniImageNet):
    def __init__(
        self,
        phase='train',
        load_single_file_split=False,
        do_not_use_random_transf=False,
        ):
        tieredMiniImageNet.__init__(
            self,
            phase=phase,
            load_single_file_split=load_single_file_split,
            do_not_use_random_transf=do_not_use_random_transf)

        mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if ((self.phase=='test' or self.phase=='val') or
            (do_not_use_random_transf==True)):
            self.transform = transforms.Compose([
                transforms.Resize(80),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.Resize(80),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])


def load_features_labels(dataset_file):
    data_file = h5py.File(dataset_file, 'r')
    count = data_file['count'][0]
    features = data_file['all_features'][...]
    labels = data_file['all_labels'][:count].tolist()
    features = features[:count,:]
    return data_file, count, features, labels


class tieredMiniImageNetFeatures(data.Dataset):
    def __init__(self, data_directory, phase='train'):

        class_splits = ['train', 'val', 'test']
        files_features = {}
        files_labels = {}
        for class_split in class_splits:
            files_features[class_split] = os.path.join(
                data_directory,
                'feature_dataset_tieredMiniImageNet_' + class_split + '.json')
            files_labels[class_split] = os.path.join(
                _TIERED_MINI_IMAGENET_DATASET_DIR,
                '{0}_labels.pkl'.format(class_split))
        print(files_features)
        self.phase = phase
        assert phase in ('train', 'val', 'test')
        self.name = 'tieredMiniImageNetFeatures_Split_' + phase

        num_train_classes = 351

        print('Loading teried MiniImageNet dataset - split {0}'.format(phase))
        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            _, _, features, labels = load_features_labels(
                files_features['train'])

            labels_train = utils.load_pickle_data(files_labels['train'])
            assert labels == labels_train['label_specific'].tolist()
            labels_super = labels_train['label_general'].tolist()
            self.labels_super = labels_super

            self.features = features
            self.labels = labels
            self.labels_super = labels_super

            self.label2ind = utils.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase=='val' or self.phase=='test':

            _, _, features, labels = load_features_labels(
                files_features[self.phase])

            labels = [num_train_classes + label for label in labels]
            self.features = features
            self.labels = labels

            self.label2ind = utils.buildLabelIndex(self.labels)
            for label in range(num_train_classes):
                self.label2ind[label] = []

            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = {label for label in range(num_train_classes)}
            self.labelIds_novel = utils.buildLabelIndex(labels).keys()

            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))


    def __getitem__(self, index):
        features_this = torch.Tensor(self.features[index]).view(-1,1,1)
        label_this = self.labels[index]
        return features_this, label_this

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = tieredMiniImageNet(phase='train_not_miniimagnet')
    #dataset = tieredMiniImageNet(phase='test')
    #from matplotlib import pyplot as plt
    #plt.figure(1)
    #plt.imshow(dataset.inv_transform(dataset[0][0]))
    #plt.show()
