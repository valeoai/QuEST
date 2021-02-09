import torch
import torch.nn as nn
import distillation.architectures as architectures

class ParallelHead(nn.Module):
    def __init__(self, opt):
        super(ParallelHead, self).__init__()

        nets_opt = opt['nets']
        assert isinstance(nets_opt, (list, tuple))
        self.num_parallel_head = len(nets_opt)

        parallel_net = []
        for i in range(self.num_parallel_head):
            parallel_net.append(architectures.factory(
                architecture_name=nets_opt[i]['architecture'],
                opt=nets_opt[i]['opt']))

        self.parallel_net = nn.ModuleList(parallel_net)
        self.concatenate_dim = opt['concatenate_dim']
        self.out_shape = opt.get('out_shape', None)

    def forward(self, features):
        outputs = []
        if isinstance(features, (list, tuple)):
            assert len(features) == self.num_parallel_head
            for i in range(self.num_parallel_head):
                outputs.append(self.parallel_net[i](features[i]))
        else:
            for i in range(self.num_parallel_head):
                outputs.append(self.parallel_net[i](features))

        if self.concatenate_dim >= 0:
            outputs = torch.cat(outputs, dim=self.concatenate_dim)
            if self.out_shape is not None:
                outputs = outputs.view(self.out_shape)

        return outputs


def create_model(opt):
    return ParallelHead(opt)
