config = {}
# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['dataset_name'] = 'CIFAR100'
data_train_opt['dataset_args'] = {'split': 'train'}
data_train_opt['epoch_size'] = None
data_train_opt['batch_size'] = 64

data_test_opt = {}
data_test_opt['dataset_name'] = 'CIFAR100'
data_test_opt['dataset_args'] = {'split': 'val'}
data_test_opt['batch_size'] = 100

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

num_classes = 100
width = 2
depth = 40
num_embeddings = 4096

networks = {}
pretrained = f'./saved_models/cifar100v3_WRNd{depth}w{width}LinearClassifier/ckpt_epoch_240.pth'

net_optionsF = {
    'depth': depth, 'widen_Factor': width, 'dropRate': 0.0,
    'num_classes': num_classes, 'extract_from': ['avgpool_2x2'], 'extract_after_relu': [False], 'pool': 'avg',
    'downscale': True}

networks['student_net'] = {
    'def_file': 'feature_extractors.wide_resnet',
    'pretrained': pretrained, 'opt': net_optionsF,
    'optim_params': None, 'force':True}

config['networks'] = networks
config['algorithm_type'] = 'classification.classification'
