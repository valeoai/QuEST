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
num_filters = [16, 16, 32, 64]
depth = 56
num_embeddings = 4096

networks = {}
pretrained_teacher = './saved_models/resnet56_vanilla/ckpt_epoch_240.pth'

net_optionsF = {
    'depth': depth, 'num_filters': num_filters,
    'num_classes':num_classes, 'extract_from':['layer3'], 'extract_after_relu':[False]}

networks['student_net'] = {
    'def_file': 'feature_extractors.resnet',
    'pretrained': pretrained_teacher, 'opt': net_optionsF,
    'optim_params': None, 'force':True}

config['networks'] = networks
config['algorithm_type'] = 'classification.classification'
