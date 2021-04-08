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
networks = {}
num_filters = [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]]

pretrained = "./saved_models/vgg13_vanilla/ckpt_epoch_240.pth"

net_optionsF = {
    'num_filters': num_filters,
    'num_classes':num_classes, 'extract_from':['block4'], 'extract_after_relu':[True], 'downscale': True}

networks['student_net'] = {
    'def_file': 'feature_extractors.vgg',
    'pretrained': pretrained, 'opt': net_optionsF,
    'optim_params': None, 'force':True}

config['networks'] = networks
config['keep_border'] = True
config['criterions'] = {}
config['algorithm_type'] = 'classification.classification'
