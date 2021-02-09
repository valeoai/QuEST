config = {}
# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['dataset_name'] = 'MITScenes'
data_train_opt['dataset_args'] = {'split': 'train'}
data_train_opt['epoch_size'] = None
data_train_opt['batch_size'] = 32

data_test_opt = {}
data_test_opt['dataset_name'] = 'MITScenes'
data_test_opt['dataset_args'] = {'split': 'test'}
data_test_opt['batch_size'] = 67

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

LUT_lr = [(150, 0.01), (200, 0.001),]
max_num_epochs = LUT_lr[-1][0]
config['max_num_epochs'] = max_num_epochs

run = 1
num_classes = 67
# Networks
num_filters_teacher = [512]
depth_teacher = 34
num_filters_student = [512]
num_embeddings = 4096
num_embeddings_layer3 = num_embeddings

networks = {}
feature_name = ['layer3', 'layer4']

net_optionsF = {
    'depth': depth_teacher,
    'add_reshape':True,
    'num_filters':num_filters_teacher,
    'extract_from':feature_name,
    'extract_after_relu':[False, False],
    'pretrained':True}

networks['teacher_net'] = {
    'def_file': 'feature_extractors.ResNet_ImageNet',
    'opt': net_optionsF,
    'optim_params': None,
    'force':True}

pretrained_VQ_layer4 = './experiments/VQ/Config_ImageNet/RN34/vector_quantizer_kmeansK4096_200k'

net_optionsVQ = {
    'num_embeddings': num_embeddings,
    'embedding_dim': num_filters_teacher[-1],
    'commitment_cost': 0.25,
    'decay': 0.99,
    'epsilon': 1e-5,
    'temperature': 5}
networks['vector_quantizer_target_layer4'] = {
    'def_file': 'miscellaneous.vector_quantization',
    'pretrained': pretrained_VQ_layer4,
    'opt': net_optionsVQ,
    'optim_params': None}

pretrained_VQ_layer3 = './experiments/VQ/Config_ImageNet/RN34/vector_quantizer_kmeansK4096_layer3_200k'
net_optionsVQ = {
    'num_embeddings': num_embeddings_layer3,
    'embedding_dim': num_filters_teacher[-1] // 2,
    'commitment_cost': 0.25,
    'decay': 0.99,
    'epsilon': 1e-5,
    'temperature': 500}
networks['vector_quantizer_target_layer3'] = {
    'def_file': 'miscellaneous.vector_quantization',
    'pretrained': pretrained_VQ_layer3,
    'opt': net_optionsVQ,
    'optim_params': None}

## Student
net_optim_paramsC = {
    'optim_type': 'sgd',
    'lr': 0.1,
    'momentum':0.9,
    'weight_decay': 5e-4,
    'nesterov': False,
    'LUT_lr':LUT_lr}
net_optionsC = {
    'classifier_type': 'linear',
    'num_classes': num_classes,
    'num_channels': num_filters_student[-1],
    'global_pooling': True}
net_optim_paramsF = {
    'optim_type': 'sgd',
    'lr': 0.1,
    'momentum':0.9,
    'weight_decay': 5e-4,
    'nesterov': False,
    'LUT_lr':LUT_lr,
    'classifier_optim': net_optim_paramsC}
net_optionsF = {
    'num_classes':num_classes,
    'extract_from':['21', '28'],
    'extract_after_relu':[False],
    'classifier_options':net_optionsC}
networks['student_net'] = {
    'def_file': 'feature_extractors.VGG_ImageNet',
    'pretrained': None,
    'opt': net_optionsF,
    'optim_params': net_optim_paramsF}


## BoW_predictor

net_optim_paramsCP = {
    'optim_type': 'sgd',
    'lr': 0.1,
    'momentum':0.9,
    'weight_decay': 5e-4,
    'nesterov': False,
    'LUT_lr':LUT_lr}

net_optionsCP_layer4 = {
    'classifier_type': 'conv_cosine',
    'num_classes': num_embeddings,
    'num_channels': num_filters_student[-1],
    'scale_cls': 3.0,
    'learn_scale': True}

net_optionsCP_layer3 = {
    'classifier_type': 'conv_cosine',
    'num_classes': num_embeddings_layer3,
    'num_channels': num_filters_student[-1],
    'scale_cls': 3.0,
    'learn_scale': True}

net_optionsCP = {
    'nets': [
        {'architecture': 'classifiers.classifier', 'opt': net_optionsCP_layer3},
        {'architecture': 'classifiers.classifier', 'opt': net_optionsCP_layer4},
    ],
    'concatenate_dim': -1,
}

networks['BoW_predictor'] = {
    'def_file': 'miscellaneous.parallel_head',
    'pretrained': None,
    'opt': net_optionsCP,
    'optim_params': net_optim_paramsCP,
}


config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions
config['feature_name'] = feature_name
config['algorithm_type'] = 'classification.classification_bow_transfer'
config['BoW_loss_coef'] = 20.0