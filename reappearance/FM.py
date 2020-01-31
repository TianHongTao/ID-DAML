#-*-coding:utf-8 -*-
import torch

from torch import nn

class FactorizationMachine(nn.Module):

    def __init__(self, factor_size: int, fm_k: int):
        super(FactorizationMachine, self).__init__()
        self.linear = nn.Linear(factor_size, 1)
        self.v = torch.nn.Parameter(torch.randn((factor_size, fm_k)))
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        # linear regression
        w = self.linear(x).squeeze()

        # cross feature
        inter1 = torch.matmul(x, self.v)
        inter2 = torch.matmul(x**2, self.v**2)
        inter = (inter1**2 - inter2) * 0.5
        inter = self.drop(inter)
        inter = torch.sum(inter, dim=1)

        return w + inter