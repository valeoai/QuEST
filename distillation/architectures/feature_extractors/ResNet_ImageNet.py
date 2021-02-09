import torchvision.models as models
import distillation.architectures.tools as tools
from collections import OrderedDict
import torch.nn as nn

def create_model(opt):
    depth = opt['depth']
    num_filters = opt['num_filters']
    pretrained = opt['pretrained']
    add_reshape = opt.get('add_reshape', None)

    if depth==34:
        print(f'depth={depth} pretrained={pretrained}')
        model = models.resnet34(pretrained=pretrained)
    elif depth==50:
        print(f'depth={depth} pretrained={pretrained}')
        model = models.resnet50(pretrained=pretrained)
    elif depth==18:
        model = models.resnet18()
    else:
        print("Not implemented.")
        return None

    if add_reshape: # adding reshape before the last layer
        key_names = list(model._modules.keys())
        od = OrderedDict()
        for k in key_names[:-1]:
            od[k] = model._modules[k]
        od['reshape'] = tools.Reshape(-1, num_filters[-1])
        od[key_names[-1]] = model._modules[key_names[-1]]
        new_model = nn.Sequential(od)
        return new_model
    else:
        return model
