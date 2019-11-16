import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def binarization(x, deterministic=True):
    if deterministic:
        sign = torch.sign(x)
        sign[sign == 0] = 1
        return sign
    else:
        return hard_sigmoid(x)


class BinaryLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(BinaryLinear, self).__init__()
        self.weight = torch.Tensor(output_size, input_size)
        self.weightB = torch.Tensor(output_size, input_size)
        self.weightOrg = torch.Tensor(output_size, input_size)
        self.maskStc = torch.Tensor(output_size, input_size)
        self.randmat = torch.Tensor(output_size, input_size)
        self.bias = torch.Tensor(output_size)
        self.gradWeight = torch.Tensor(output_size, input_size)
        self.gradBias = torch.Tensor(output_size)
        # self.stcWeights =


def hard_sigmoid(x):
    return torch.clamp((x + 1.0) / 2.0, 0, 1)


if __name__ == "__main__":
    example = torch.tensor([[0.7, -1.2, 0., 2.3],
                      [0.1, 0, -0.1, -3.0]]
                     )
    print(binarization(example, deterministic=True))
