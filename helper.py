import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from discriminator import _netD, _netG
import numpy as np 


# def log_sum_exp(x, axis=1):
#     soft = nn.LogSoftmax()
#     m, _ = torch.max(x, dim = axis, keepdim = True)
#     # m = m.data
#     return m+soft(x-m).data

def log_sum_exp(x, axis=1):
    m, _ = torch.max(x, axis, keepdim=True)
    # print(m.size(), "m_size")
    # print(x.size())
    sub = x-m.expand_as(x)
    # print(x, "x")
    # print(m, "m")
    # print(sub, "sub")
    right = torch.log(torch.sum(torch.exp(sub),1, keepdim=True))
    # print(torch.exp(sub), "exponent")
    # print(right, "right")
    # print(m.size())
    # print(right.size())
    # print(m.expand_as(right), "expanded right")
    sums = m.expand_as(right) + right
    return sums