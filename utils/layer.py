import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.stats import binom


def hard_sigmoid(x):
    clamp = (torch.clamp((x + 1.0) / 2.0, -1, 1))
    return clamp


def binarization(w, deterministic=True):
    if deterministic:
        w[w >= 0] = 1
        w[w < 0] = -1
        # sign = torch.sign(x)
        # sign[sign == 0] = 1
        return w
    else:
        p = hard_sigmoid(w)
        matrix = torch.empty(p.shape).uniform_(-1, 1)
        bin_weight = (p >= matrix).type(torch.float32)
        bin_weight[bin_weight == 0] = -1
        return bin_weight

        # return torch.rand(hard_sigmoid(w).size()).round() * 2 - 1


class BinaryLinear(nn.Linear):
    def __init__(self, weight, weight_org, bias):
        super(BinaryLinear, self).__init__()
        self.weight = weight
        self.weight_org = weight_org
        self.bias = bias

    def forward(self, x):
        if x.size(1) != 784:
            x.data = binarization(x.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = binarization(self.weight.org)
        out = F.linear(x, self.weight)
        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def glorot_init(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = np.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


if __name__ == "__main__":

    example = torch.rand(5, 5)
    print(binarization(example, deterministic=False))

