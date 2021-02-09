from distillation.datasets.imagenet_dataset import ImageNet
from distillation.datasets.imagenet_dataset import ImageNetLowShot
from distillation.datasets.imagenet_dataset import ImageNet3x3Patches
from distillation.datasets.imagenet_dataset import ImageNetLowShot3x3Patches
from distillation.datasets.imagenet_dataset import ImageNetFeatures
from distillation.datasets.imagenet_dataset import ImageNetLowShotFeatures
from distillation.datasets.imagenet_dataset import ImageNetWithPrtbImages
from distillation.datasets.imagenet_dataset import ImageNetWithPrtbImagesV2
from distillation.datasets.mini_imagenet_dataset import MiniImageNet
from distillation.datasets.mini_imagenet_dataset import MiniImageNet3x3Patches
from distillation.datasets.mini_imagenet_dataset import MiniImageNetImagesAnd3x3Patches
from distillation.datasets.mini_imagenet_dataset import MiniImageNetImagesAndPrtbImages
from distillation.datasets.mini_imagenet_dataset import MiniImageNetImagesAndPrtbImagesv2
from distillation.datasets.mini_imagenet_dataset import MiniImageNet80x80
from distillation.datasets.mini_imagenet_dataset import MiniImageNetFeatures
from distillation.datasets.tiered_mini_imagenet_dataset import tieredMiniImageNet
from distillation.datasets.tiered_mini_imagenet_dataset import tieredMiniImageNet80x80
from distillation.datasets.tiered_mini_imagenet_dataset import tieredMiniImageNetFeatures
#from distillation.datasets.cifar_dataset import CIFAR
#from distillation.datasets.cifar_dataset import CIFARv2
from distillation.datasets.cifar_dataset import CIFARv3
from distillation.datasets.mit67_cub200_datasets import MITScenes
from distillation.datasets.mit67_cub200_datasets import CUB200

def dataset_factory(dataset_name, *args, **kwargs):
    datasets_collection = {}
    datasets_collection['MiniImageNet'] = MiniImageNet
    datasets_collection['MiniImageNet80x80'] = MiniImageNet80x80
    datasets_collection['MiniImageNetFeatures'] = MiniImageNetFeatures
    datasets_collection['MiniImageNet3x3Patches'] = MiniImageNet3x3Patches
    datasets_collection['MiniImageNetImagesAnd3x3Patches'] = MiniImageNetImagesAnd3x3Patches
    datasets_collection['MiniImageNetImagesAndPrtbImages'] = MiniImageNetImagesAndPrtbImages
    datasets_collection['MiniImageNetImagesAndPrtbImagesv2'] = MiniImageNetImagesAndPrtbImagesv2
    datasets_collection['tieredMiniImageNet'] = tieredMiniImageNet
    datasets_collection['tieredMiniImageNet80x80'] = tieredMiniImageNet80x80
    datasets_collection['tieredMiniImageNetFeatures'] = tieredMiniImageNetFeatures
    datasets_collection['ImageNet'] = ImageNet
    datasets_collection['ImageNetLowShot'] = ImageNetLowShot
    datasets_collection['ImageNet3x3Patches'] = ImageNet3x3Patches
    datasets_collection['ImageNetLowShot3x3Patches'] = ImageNetLowShot3x3Patches
    datasets_collection['ImageNetFeatures'] = ImageNetFeatures
    datasets_collection['ImageNetLowShotFeatures'] = ImageNetLowShotFeatures
    datasets_collection['ImageNetWithPrtbImages'] = ImageNetWithPrtbImages
    datasets_collection['ImageNetWithPrtbImagesV2'] = ImageNetWithPrtbImagesV2

#    datasets_collection['CIFAR'] = CIFAR
#    datasets_collection['CIFARv2'] = CIFARv2
    datasets_collection['CIFARv3'] = CIFARv3

    datasets_collection['MITScenes'] = MITScenes
    datasets_collection['CUB200'] = CUB200

    return datasets_collection[dataset_name](*args, **kwargs)
