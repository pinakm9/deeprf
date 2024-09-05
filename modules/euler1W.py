# load necessary modules
import numpy as np 
import os, sys, time
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import utility as ut

# from scipy.linalg import eigh
import pandas as pd
import json
import oneshot as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter 
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch.nn.functional as tfn
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
import chain


DTYPE = 'float64'
torch.set_default_dtype(torch.float64)

class Euler1W(nn.Module):
    def __init__(self, D, D_r, B):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.inner = nn.ModuleList([nn.Linear(self.D if i==0 else self.D_r, self.D_r, bias=True) for i in range(B)])
        self.outer = nn.Linear(self.D_r, self.D, bias=False)
        
    def forward(self, x):
        y = x + 0.
        for i in range(self.B):
            y = torch.tanh(self.inner[i](y))
        return self.outer(y)
    

class DeepRF(chain.DeepRF):
    def __init__(self, D_r, B, L0, L1, Uo, beta, name='nn', save_folder='.'):
        """
        Args:
            D_r: dimension of the feature 
            B: number of RF blocks
            name: name of the DeepRF
            L0: left limit of tanh input for defining good rows
            L1: right limit tanh input for defining good rows
            Uo: training data
            beta: regularization parameter
        """        
        super().__init__(D_r, B, L0, L1, Uo, beta, name, save_folder)
        self.net = Euler1W(self.sampler.dim, D_r, B)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})

    def compute_W(self, X, Y):
        """
        Description: computes W with Ridge regression

        Args:
            Wb_in: internal weights
            X: input
            Y: output 
        """
        for i in range(self.net.B):
            W_in = self.net.inner[i].weight.detach().numpy()
            b_in = self.net.inner[i].bias.detach().numpy()
            X = np.tanh(W_in @ X + b_in[:, np.newaxis])
        return (Y@X.T) @ np.linalg.solve(X@X.T + self.beta*self.I_r, self.I_r)  

    def init(self):
        UoOriginal = self.sampler.Uo
        Uo = UoOriginal + 0.
        for i in range(self.net.B):
            sampler = sm.GoodRowSampler(self.sampler.L0, self.sampler.L1, Uo)
            Wb = sampler.sample(self.net.D_r) 
            W_in = Wb[:, :-1]
            b_in = Wb[:, -1]
            self.net.inner[i].weight = nn.Parameter(torch.Tensor(W_in))
            self.net.inner[i].bias = nn.Parameter(torch.Tensor(b_in))
            Uo = np.tanh(W_in @ Uo + b_in[:, np.newaxis])

        W = self.compute_W(UoOriginal[:, :-1], UoOriginal[:, 1:])
        self.net.outer.weight = nn.Parameter(torch.Tensor(W))
