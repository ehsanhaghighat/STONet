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


from abc import abstractmethod
from typing import Dict, Any
import torch
import logging

class Loss(torch.nn.Module):
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                sample_weights: Dict[str, torch.Tensor] = {}) -> Any:
        loss = self.eval(inputs, outputs, targets, sample_weights)
        return loss * self.weight

    @abstractmethod
    def eval(self,
             inputs: Dict[str, torch.Tensor],
             outputs: Dict[str, torch.Tensor],
             targets: Dict[str, torch.Tensor],
             sample_weights: Dict[str, torch.Tensor] = {}) -> Any:
        assert False, 'Not implemented!'

    @staticmethod
    def grad(y, x):
        return torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=True, create_graph=True)[0]
