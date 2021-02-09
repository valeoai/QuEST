import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import distillation.architectures.tools as tools


class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()

        self.classifier_type = opt['classifier_type']
        self.num_channels = opt['num_channels']
        self.num_classes = opt['num_classes']
        self.global_pooling = opt.get('global_pooling', False)

        if self.classifier_type == 'linear':
            bias = opt.get('bias', True)
            self.layers = nn.Linear(
                self.num_channels, self.num_classes, bias=bias)
            if bias:
                self.layers.bias.data.zero_()

            fout = self.layers.out_features
            self.layers.weight.data.normal_(0.0,  np.sqrt(2.0/fout))

        elif self.classifier_type == 'conv_cosine':
            assert self.global_pooling is False
            normalize_x = opt.get('normalize_x', True)
            normalize_w = opt.get('normalize_w', True)
            bias = opt.get('bias', False)
            per_plane = opt.get('per_plane', False)
            kernel = opt.get('kernel', 1)
            padding = (kernel - 1) // 2
            self.layers = tools.Conv2dCos(
                self.num_channels, self.num_classes,
                kernel_size=kernel, stride=1, padding=padding,
                bias=bias,
                scale=opt['scale_cls'],
                learn_scale=opt['learn_scale'],
                per_plane=per_plane,
                normalize_x=normalize_x,
                normalize_w=normalize_w)
            add_pre_bnrelu = opt.get('add_pre_bnrelu', False)
            if add_pre_bnrelu:
                conv_cos = self.layers
                self.layers = nn.Sequential(
                    nn.BatchNorm2d(self.num_channels),
                    nn.ReLU(inplace=True),
                    conv_cos
                )
        else:
            raise ValueError(
                'Not implemented / recognized classifier type {0}'.format(
                    self.classifier_type))

    def flatten(self):
        return self.classifier_type == 'linear'

    def forward(self, features):
        if self.global_pooling:
            features = tools.global_pooling(features, pool_type='avg')

        if features.dim() > 2 and self.flatten():
            features = features.view(features.size(0), -1)
        scores = self.layers(features)

        return scores


def create_model(opt):
    return Classifier(opt)
