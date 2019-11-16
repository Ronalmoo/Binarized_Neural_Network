import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def hard_sigmoid(x):
    return torch.clamp((x + 1.0) / 2.0, 0, 1)


def binarization(x, deterministic=True):
    if deterministic:
        x[x >= 0] = 1
        x[x < 0] = -1
        # sign = torch.sign(x)
        # sign[sign == 0] = 1
        return x
    else:
        return hard_sigmoid(x)


class BinaryLinear(nn.Module):
    def __init__(self, weight, weight_org, bias):
        super(BinaryLinear, self).__init__()
        torch.flatten()
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
    example = torch.tensor([[0.7, -1.2, 0., 2.3],
                      [0.1, 0, -0.1, -3.0]]
                     )
    model = BinaryLinear()
    model.apply(BinaryLinear.glorot_init())