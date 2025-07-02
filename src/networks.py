"""
STONet: A Neural Operator for Modeling Solute Transport in Micro-Cracked Reservoirs

This code is part of the STONet repository: https://github.com/ehsanhaghighat/STONet

Citation:
@article{haghighat2024stonet,
  title={STONet: A neural operator for modeling solute transport in micro-cracked reservoirs},
  author={Haghighat, Ehsan and Adeli, Mohammad Hesan and Mousavi, S Mohammad and Juanes, Ruben},
  journal={arXiv preprint arXiv:2412.05576},
  year={2024}
}

Paper: https://arxiv.org/abs/2412.05576
"""


import torch
from typing import Dict

def to_list(x):
    if isinstance(x, list):
        return x
    return [x]

class BaseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'BaseNet'

    def forward(self, inputs):
        raise NotImplementedError

    def __repr__(self):
        return self.name

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()



class Fourier(BaseNet):
    def __init__(self, input_dim, n=10, p=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.n = n
        self.p = p
        self.frequencies = torch.arange(0., float(n)) * torch.pi / p
        self.output_dim = 2 * n * sum(input_dim.values())
        self.name = f'Fourier:{input_dim}->{self.output_dim}'

    def forward(self, inputs: torch.Tensor):
        outputs = []
        for freq in self.frequencies:
            outputs += [torch.sin(freq * inputs), torch.cos(freq * inputs)]
        return torch.cat(outputs, dim=-1)


class Dense(BaseNet):
    def __init__(self, input_dim, output_dim, actf='Tanh', fourier=None):
        super().__init__()
        assert fourier is None or isinstance(fourier, Fourier), 'fourier must be a Fourier object'
        self.fourier = fourier
        if fourier is not None:
            self.linear = torch.nn.Linear(fourier.output_dim, output_dim)
        else:
            self.linear = torch.nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.uniform_(self.linear.bias)
        if actf.lower() == 'linear':
            self.actf = None
        elif hasattr(torch.nn, actf):
            self.actf = getattr(torch.nn, actf)()
        elif hasattr(torch, actf):
            self.actf = getattr(torch, actf)
        else:
            raise ValueError(f'Not recognized activation function: {actf}')
        self.name = f'Dense:{input_dim}->{output_dim}:{actf}'

    def forward(self, inputs: torch.Tensor):
        if self.fourier is not None:
            inputs = self.fourier(inputs)
        outputs = self.linear(inputs)
        if self.actf is not None:
            outputs = self.actf(outputs)
        return outputs


class DenseLayer(BaseNet):
    def __init__(self,
                 input_dim: Dict=dict(x=1),
                 output_dim: Dict=dict(f=1),
                 actf='Tanh'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert len(output_dim) == 1, 'Only support one output dimension'
        self.dense = Dense(sum(input_dim.values()), sum(output_dim.values()), actf)

    def forward(self, inputs):
        xs = torch.cat([inputs[x] for x in self.input_dim], dim=-1)
        ys = self.dense(xs)
        return {k: ys for k in self.output_dim}


class MLP(BaseNet):
    def __init__(self,
                 input_dim=dict(x=1),
                 output_dim=dict(f=1),
                 hidden_layers=4*[50],
                 actf='Tanh',
                 output_actf='Linear',
                 fourier=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # find input/output dimensions
        in_layer_dim = [sum(input_dim.values())] + hidden_layers[:-1]
        out_layer_dim = hidden_layers
        # set hidden layers
        self.hidden_layers = torch.nn.ModuleDict()
        for i, (in_dim, out_dim) in enumerate(zip(in_layer_dim, out_layer_dim)):
            self.hidden_layers[f'D-{i}'] = Dense(in_dim, out_dim, actf, fourier if i==0 else None)
        # set output layer
        in_dim_to_output_layer = out_layer_dim[-1]
        self.output_layers = torch.nn.ModuleDict()
        for out_name, out_dim in output_dim.items():
            name = f'{out_name}:{in_dim_to_output_layer}->{out_dim}'
            self.output_layers[name] = Dense(in_dim_to_output_layer, out_dim, output_actf)

    def forward(self, inputs):
        xs = torch.cat([inputs[x] for x in self.input_dim], dim=-1)
        for _, layer in self.hidden_layers.items():
            xs = layer(xs)
        ys = {}
        for layer_name, layer in self.output_layers.items():
            ys[layer_name.split(':')[0]] = layer(xs)
        return ys


class EnrichedDeepONet(BaseNet):
    def __init__(self,
                 trunk_net=MLP(dict(x=1), dict(e=20), 2*[50], 'Tanh'),
                 branch_net=MLP(dict(s=1), dict(e=20), 2*[50], 'Tanh'),
                 root_net=MLP(dict(e=60), dict(f=1), 2*[50], 'Tanh')):
        super().__init__()
        self.trunk_net = trunk_net
        self.branch_nets = torch.nn.ModuleList()
        for b in to_list(branch_net):
            self.branch_nets.append(b)
        self.root_net = root_net

    def forward(self, inputs):
        out_branches = [net(inputs) for net in self.branch_nets]
        out_trunk = self.trunk_net(inputs)
        in_root = {}
        for key in out_trunk:
            in_tensors = []
            for out_branch in out_branches:
                in_tensors += [out_trunk[key] * out_branch[key],
                               out_trunk[key] + out_branch[key],
                               out_trunk[key] - out_branch[key]]
            in_root[key] = torch.cat(in_tensors, dim=-1)
        return self.root_net(in_root)


class STONet_Attention(BaseNet):
    def __init__(self,
                 embedding_dim: Dict=dict(e=50),
                 actf: str='Tanh',
                 ):
        super().__init__()
        self.multi = DenseLayer(embedding_dim, embedding_dim, actf)
        self.add = DenseLayer(embedding_dim, embedding_dim, actf)
        self.sub = DenseLayer(embedding_dim, embedding_dim, actf)
        self.output = DenseLayer({k: d*3 for k, d in embedding_dim.items()}, embedding_dim, actf)

    def forward(self, xs_trunk, xs_branch):
        # eval multi-head attention
        ys_multi = self.multi({k: xs_trunk[k] * xs_branch[k] for k in xs_trunk})
        ys_add = self.add({k: xs_trunk[k] + xs_branch[k] for k in xs_trunk})
        ys_sub = self.sub({k: xs_trunk[k] - xs_branch[k] for k in xs_trunk})
        ys = {k: torch.cat([ys_multi[k], ys_add[k], ys_sub[k]], dim=-1) for k in ys_multi}
        return self.output(ys)


class STONet(BaseNet):
    def __init__(self,
                 trunk_net=MLP(dict(x=1), dict(e=20), 2*[50], 'Tanh'),
                 branch_net=MLP(dict(s=1), dict(e=20), 2*[50], 'Tanh'),
                 attention_net=4*[STONet_Attention(dict(e=20), 'Tanh')],
                 root_net=MLP(dict(e=20), dict(f=1), 2*[50], 'Tanh')):
        super().__init__()
        self.trunk_net = trunk_net
        self.branch_net = branch_net
        self.attention_nets = torch.nn.ModuleList()
        for a in to_list(attention_net):
            self.attention_nets.append(a)
        self.root_net = root_net

    def forward(self, inputs):
        out_branch = self.branch_net(inputs)
        out_trunk = self.trunk_net(inputs)
        for net in self.attention_nets:
            out_trunk = net(out_trunk, out_branch)
        return self.root_net(out_trunk)

