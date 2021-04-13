import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections.abc import Iterable
import distillation.architectures.feature_extractors.utils as futils
import distillation.architectures.feature_extractors.VGG_ImageNet as VGG_ImageNet
import distillation.architectures.feature_extractors.mobilenetv2 as mobilenet


class ResNetClassifier(nn.Module):
    def __init__(self, model):
        super(ResNetClassifier, self).__init__()
        self.avgpool = model.avgpool
        self.reshape = model.reshape
        self.fc = model.fc

    def forward(self, x):
        return self.fc(self.reshape(self.avgpool(x)))


class ResNetFeatureExtractor(futils.SequentialFeatureExtractorAbstractClass):
    def __init__(self, model):
        feature_blocks = [
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]
        all_feat_names = [
            'conv1',
            'bn1',
            'relu',
            'maxpool',
            'layer1',
            'layer2',
            'layer3',
            'layer4',
        ]
        super(ResNetFeatureExtractor, self).__init__(
            all_feat_names, feature_blocks)


class VGG9Classifier(nn.Module):
    def __init__(self, model):
        super(VGG9Classifier, self).__init__()
        self.avgpool = model._modules['classifier'][0]
        self.reshape = model._modules['classifier'][1]
        self.fc = model._modules['classifier'][2]

    def forward(self, x):
        return self.fc(self.reshape(self.avgpool(x)))


class VGG9FeatureExtractor(futils.SequentialFeatureExtractorAbstractClass):
    def __init__(self, model, extract_from):
        ext_frm = sorted([int(i) for i in extract_from]) # to make sure the order is right
        n_splits = len(ext_frm)
        all_feat_names = ['layer' + str(4 - i) for i in range(len(ext_frm))]
        all_feat_names.reverse()
        if list(model._modules['features']._modules.keys())[-1] not in extract_from:
            n_splits+=1
            all_feat_names.append('layer_x')
        feature_blocks = [[] for i in range(n_splits)]
        j=0
        for i, m in enumerate(model._modules['features']):
            feature_blocks[j].append(m)
            if j<len(ext_frm) and i==ext_frm[j]:
                j+=1
        feature_blocks_for_RN = [nn.Sequential(*b) for b in feature_blocks]
        super(VGG9FeatureExtractor, self).__init__(
            all_feat_names, feature_blocks_for_RN)


def split_network( model, extract_from_layer, after_relu):
    """
    This function splits the network in Feature extractor and classifier networks.
    extract_from_layer: list of layer names from which features are to be extracted.
    after_relu: list of booleans, True means use relu after the extraction layer specified in extract_from_layer list.
    """
    if len(extract_from_layer) == 1:
        feat_ext = []
        clf = []
        key_names = list(model._modules.keys())
        name = extract_from_layer[0]
        extract_from = key_names.index(name)
        for i in range(extract_from+1): # Feature extractor
            feat_ext.append(model._modules[key_names[i]])
        i = extract_from+1
        if after_relu[0]:  # Include till next ReLU
                if key_names[i].startswith('bn'):
                    feat_ext.append(model._modules[key_names[i]])  # BatchNorm
                    i +=1
                feat_ext.append(model._modules[key_names[i]])  # ReLU
                i +=1
        for key in key_names[i:]: # Classifier
            clf.append(model._modules[key])
        if isinstance(model, mobilenet.MobileNetV2):  # MobileNet2 for CIFAR
            fext = nn.Sequential(*feat_ext)
            feat_ext = []
            for k in fext._modules.keys():
                if isinstance(fext._modules[k], Iterable):
                    feat_ext.extend(list(fext._modules[k]))
                else:
                    feat_ext.append(fext._modules[k])
        return nn.Sequential(*feat_ext), nn.Sequential(*clf)
    else:  # For multiple extraction layers, implemented split only for VGG9 and ResNet34.
        if isinstance(model, VGG_ImageNet.VGG):  # VGG9. As in VGG9 network the layers are named by digits.
            feature_ext = VGG9FeatureExtractor(model, extract_from_layer)
            classifier = VGG9Classifier(model)
            return feature_ext, classifier
        else:  # ResNet34
            assert all([(x == False) for x in after_relu])
            feature_ext = ResNetFeatureExtractor(model)
            classifier = ResNetClassifier(model)
            return feature_ext, classifier


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def global_pooling(x, pool_type):
    assert(x.dim() == 4)
    if pool_type == 'max':
        return F.max_pool2d(x, (x.size(2), x.size(3)))
    elif pool_type == 'avg':
        return F.avg_pool2d(x, (x.size(2), x.size(3)))
    else:
        raise ValueError('Unknown pooling type.')


class GlobalPooling(nn.Module):
    def __init__(self, pool_type):
        super(GlobalPooling, self).__init__()
        assert(pool_type == 'avg' or pool_type == 'max')
        self.pool_type = pool_type

    def forward(self, x):
        return global_pooling(x, pool_type=self.pool_type)


class Conv2dCos(nn.Module):
    def __init__(self, in_planes, out_planes, bias=False, scale=None, learn_scale=True):
        super(Conv2dCos, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes

        weight = torch.FloatTensor(out_planes, in_planes, 1, 1).normal_(
            0.0, np.sqrt(2.0/in_planes))
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(out_planes).fill_(0.0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None
        if scale:
            scale = torch.FloatTensor(1).fill_(scale)
            self.scale = nn.Parameter(scale, requires_grad=learn_scale)
        else:
            self.scale = None

    def forward(self, x):
        weight = self.weight
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        weight = F.normalize(weight, p=2, dim=1, eps=1e-12)
        if self.scale is not None:
            weight = weight * self.scale.view(-1, 1, 1, 1)
        return F.conv2d(x, weight, bias=self.bias, stride=1, padding=0)
