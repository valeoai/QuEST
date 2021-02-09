config = {}
# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['dataset_name'] = 'ImageNet'
data_train_opt['dataset_args'] = {
    'split': 'train',
    'use_geometric_aug':True,
    'use_color_aug': False,
    'cutout_length': 0}
data_train_opt['epoch_size'] = None
data_train_opt['batch_size'] = 256

data_test_opt = {}
data_test_opt['dataset_name'] = 'ImageNet'
data_test_opt['dataset_args'] = {
    'split': 'val',
    'use_geometric_aug':False,
    'use_color_aug': False,
    'cutout_length': 0}
data_test_opt['batch_size'] = 200

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

config['max_num_epochs'] = 100
LUT_lr = [(30, 0.1), (60, 0.01), (90, 0.001), (100, 0.0001)]

networks = {}
depth = 34
num_filters = None
num_classes = 1000
pretrained = True # to load weights from torchvision.models

net_optionsF = {'depth': depth, 'add_reshape':True, 'num_filters':[512],
     'extract_from':['layer4'], 'extract_after_relu':[False], 'pretrained':pretrained} ##  no relu after layer3 so, extract_after_relu should always be False

networks['student_net'] = {
    'def_file': 'feature_extractors.ResNet_ImageNet',
    'pretrained': pretrained, 'opt': net_optionsF,
    'optim_params': None, 'force':True}

config['networks'] = networks
config['keep_border'] = True
config['criterions'] = {}
config['algorithm_type'] = 'classification.classification'