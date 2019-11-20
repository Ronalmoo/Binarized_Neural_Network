import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse


def hard_sigmoid(x:torch.tensor):
    clamp = (torch.clamp((x + 1.0) / 2.0, 0, 1))
    return clamp


def binarization(w, **kwargs: str):
    if "deterministic" in kwargs.values():
        torch.tensor(w)
        w[w >= 0] = 1
        w[w < 0] = -1
        # sign = torch.sign(x)
        # sign[sign == 0] = 1
        return w
    elif "stochastic" in kwargs.values():
        p = hard_sigmoid(w)
        matrix = torch.empty(p.shape).uniform_(-1, 1)
        bin_weight = (p >= matrix).type(torch.float32)
        bin_weight[bin_weight == 0] = -1
        return bin_weight
        # return torch.rand(hard_sigmoid(w).size()).round() * 2 - 1
    else:
        raise RuntimeError("{} not supported".format(kwargs.values()))


if __name__ == "__main__":
    # Test code
    # Use parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="deterministic")
    args = parser.parse_args()
    example = torch.randn(5, 5)
    print("example: {}".format(example))
    prob = hard_sigmoid(example)
    print("prob: {}".format(prob))
    if args.mode == "deterministic":
        binarization = binarization(example, mode=args.mode)

        for i in range(99):
            print("Binarization: {}".format(binarization))

    elif args.mode == "stochastic":
        binarization_sum = binarization(example, mode=args.mode)
        print("Before sum Binarization: {}".format(binarization_sum))

        for i in range(99):
            binarization_sum += binarization(example, mode="stochastic")
        print("After sum Binarization: {}".format(binarization_sum))

    else:
        raise RuntimeError("{} not supported".format(args.mode()))
