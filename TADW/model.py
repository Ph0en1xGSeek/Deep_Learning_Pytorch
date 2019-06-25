import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class TADW(nn.Module):

    def __init__(self, M=None, T=None, k=64, f_t=None,
                 dim1=256, dim2=128):
        super(TADW, self).__init__()
        self.k = k
        self.f_t = f_t
        N = M.shape[0]
        W = np.zeros((N, k))
        # H = np.zeros((k, f_t))
        self.W = nn.Parameter(torch.tensor(W, dtype=torch.float), requires_grad=True)
        # self.H = nn.Parameter(torch.tensor(H, dtype=torch.float), requires_grad=True)
        self.M = torch.tensor(M, requires_grad=False, dtype=torch.float)
        self.T = torch.tensor(T, requires_grad=False, dtype=torch.float)

        self.H = nn.Linear(f_t, k, bias=False)
        # self.H = nn.Sequential(
        #     nn.Linear(f_t, dim1, bias=False),
        #     nn.ELU(),
        #     nn.Linear(dim1, dim2, bias=False),
        #     nn.ELU(),
        #     nn.Linear(dim2, k, bias=False),
        #     nn.ELU()
        # )

        init.xavier_uniform_(self.W)
        # init.xavier_uniform_(self.H)

    def forward(self):
        H_T = F.normalize(self.H(self.T), dim=1)
        # H_T = F.normalize(torch.mm(self.T, self.H.t()), dim=1)
        W = F.normalize(self.W, dim=1)
        output = F.softmax(torch.mm(W, H_T.t()), dim=1)
        return output, W, H_T

    def loss(self):
        output, _, _ = self.forward()
        loss = torch.sum(torch.pow(self.M - output, exponent=2))
        return loss


