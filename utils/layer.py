import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import *


class BinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, mode='Stochastic'):
        super(BinaryLinear, self).__init__(in_features, out_features)
        self.mode = mode
        self.bias = nn.Parameter(torch.randn(in_features, out_features))
        self.bin_weight = self.binarization(self.weight, self.mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.clipping_weight(self.weight)
        self.bin_weight = self.binarization(self.weight, self.mode)
        return F.linear(x, self.bin_weight, self.bias)

    def binarization(self, w: torch.tensor, mode: str) -> torch.tensor:
        with torch.no_grad():
            if mode == 'deterministic':
                bin_weight = deterministic(w)

            elif mode == "stochastic":
                bin_weight = stochastic(w)

            else:
                raise RuntimeError("mode name should be 'deterministic' or 'stochastic'")
        bin_weight.requires_grad = True
        bin_weight.register_hook(self.grad_deep_copy)
        return bin_weight

    def grad_deep_copy(self, grad):
        self.weight.grad = torch.detach(grad).clone()


def deterministic(w: torch.tensor) -> torch.tensor:
    with torch.no_grad():
        w[w >= 0] = 1
        w[w < 0] = -1
        return w


def stochastic(w: torch.tensor) -> torch.tensor:
    with torch.no_grad():
        p = hard_sigmoid(w)
        matrix = torch.empty(p.shape).uniform_(-1, 1)
        bin_weight = (p >= matrix).type(torch.float32)
        bin_weight[bin_weight == 0] = -1
        return bin_weight


def hard_sigmoid(x: torch.tensor) -> torch.tensor:
    clamp = (torch.clamp((x + 1.0) / 2.0, 0, 1))
    return clamp


if __name__ == "__main__":
    binary_linear = BinaryLinear(3, 3, 'stochastic')
    print(binary_linear)