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
            bias = opt.get('bias', False)
            self.layers = tools.Conv2dCos(
                self.num_channels, self.num_classes, bias=bias,
                scale=opt['scale_cls'], learn_scale=opt['learn_scale'])
        else:
            raise ValueError(f'Not supported classifier type {self.classifier_type}')

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
