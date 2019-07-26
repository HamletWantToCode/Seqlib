import torch
import torch.nn.functional as F


def nll_loss(output, target):
    _output = torch.transpose(output, dim0=1, dim1=2)
    return F.nll_loss(_output, target)

