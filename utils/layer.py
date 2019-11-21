import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import *


def hard_sigmoid(x: torch.tensor) -> torch.tensor:
    clamp = (torch.clamp((x + 1.0) / 2.0, 0, 1))
    return clamp


def binarization(w: torch.tensor, mode="deterministic") -> torch.tensor:
    if mode == 'deterministic':
        return deterministic(w)

    elif mode == "stochastic":
        return stochastic(w)

    else:
        raise RuntimeError


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


if __name__ == "__main__":
    # Test code
    # Use parser
    example = torch.randn(5, 5)
    print("example: {}".format(example))
    prob = hard_sigmoid(example)
    print("prob: {}".format(prob))
    binarization_det = binarization(example, mode='deterministic')

    print("Binarization: {}".format(binarization_det))

    binarization_sum = binarization(example, mode='stochastic')
    print("Before sum Binarization: {}".format(binarization_sum))

    for i in range(99):
        binarization_sum += binarization(example, mode="stochastic")
    print("After sum Binarizaion: {}".format(binarization_sum))
