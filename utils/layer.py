import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


if __name__ == "__main__":

    example = torch.rand(5, 5)
    print(binarization(example, deterministic=False))
