import torch
import torch.nn as nn
from torch import Tensor


def mask_loss(input_: Tensor, target: Tensor):
    return torch.sub(input_, target).abs().mean()


if __name__ == '__main__':
    inputs = torch.rand([3, 4, 16, 1024])
    targets = torch.zeros([1, 1024])
    # x = torch.flatten(inputs).sub(torch.flatten(targets)).abs()
    print(nn.BCELoss(inputs, targets))
