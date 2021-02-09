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
data_test_opt['batch_size'] = 100

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

config['max_num_epochs'] = 100
LUT_lr = [(30, 0.1), (60, 0.01), (90, 0.001), (100, 0.0001)]
max_num_epochs = LUT_lr[-1][0]
config['max_num_epochs'] = max_num_epochs
assert max_num_epochs == 100

run = 1
num_classes = 1000
num_filters_teacher = [512]
depth_teacher = 34
num_filters_student = [512]
depth_student = 18
num_embeddings = 4096

networks = {}
pretrained = True # to load weights from torchvision.models

net_optionsF = {'depth': depth_teacher, 'add_reshape':True, 'num_filters':num_filters_teacher,
     'extract_from':['layer4'], 'extract_after_relu':[False], 'pretrained':pretrained}

networks['teacher_net'] = {
    'def_file': 'feature_extractors.ResNet_ImageNet', 'opt': net_optionsF,
    'optim_params': None, 'force':True}

pretrained_VQ = f'./experiments/VQ/Config_ImageNet/RN34/vector_quantizer_kmeansK4096_200k'

net_optionsVQ = {
    'num_embeddings': num_embeddings,
    'embedding_dim': num_filters_teacher[-1],
    'commitment_cost': 0.25,
    'decay': 0.99,
    'epsilon': 1e-5,
    'temperature': 5}
networks['vector_quantizer_target'] = {
    'def_file': 'miscellaneous.vector_quantization',
    'pretrained': pretrained_VQ,
    'opt': net_optionsVQ,
    'optim_params': None}

net_optim_paramsC = {
    'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 1e-4,
    'nesterov': True, 'LUT_lr':LUT_lr}
net_optionsC = {
    'classifier_type': 'linear', 'num_classes': num_classes,
    'num_channels': num_filters_teacher[-1], 'global_pooling': True}

net_optim_paramsF = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 1e-4,
    'nesterov': True, 'LUT_lr':LUT_lr, 'classifier_optim':net_optim_paramsC}

net_optionsF = {'depth': depth_student, 'add_reshape':True, 'num_filters': num_filters_student,
                'num_classes':num_classes, 'extract_from':['layer4'], 'extract_after_relu':[False], 'pretrained':False,
                'classifier_options':net_optionsC}

networks['student_net'] = {'def_file': 'feature_extractors.ResNet_ImageNet',
    'pretrained': None, 'opt': net_optionsF, 'optim_params': net_optim_paramsF}

net_optim_paramsCP = {
    'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 1e-4,
    'nesterov': True, 'LUT_lr':LUT_lr}
net_optionsCP = {
    'classifier_type': 'conv_cosine',
    'num_classes': num_embeddings,
    'num_channels': num_filters_student[-1],
    'scale_cls': 3.0,
    'learn_scale': True}
networks['BoW_predictor'] = {
    'def_file': 'classifiers.classifier',
    'pretrained': None, 'opt': net_optionsCP, 'optim_params': net_optim_paramsCP}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions

config['algorithm_type'] = 'classification.classification_bow_transfer'
config['BoW_loss_coef'] = 1.0
