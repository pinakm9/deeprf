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

class EulerAut2(nn.Module):
    def __init__(self, D, D_r, B, alpha=0.5):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.alpha = alpha
        self.inner_0 = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=True) for _ in range(B)])
        self.inner_1 = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=True) for _ in range(B)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.D, bias=False) for _ in range(B)])
        
    def forward(self, x):
        y = x + 0.
        z = x + 0.
        for i in range(self.B):
            w = y + 0.
            y += self.outer[i](torch.tanh(self.alpha * self.inner_1[i](y) + (1. - self.alpha) * self.inner_0[i](z)))
            z = w + 0.
        return y
    

class DeepRF(chain.DeepRF):
    def __init__(self, D_r, B, alpha, L0, L1, Uo, beta, name='nn', save_folder='.', normalize=False):
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
        super().__init__(D_r, B, L0, L1, Uo, beta, name, save_folder, normalize)
        self.net = EulerAut2(self.sampler.dim, D_r, B, alpha)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})

    def compute_W(self, Wb1, Wb0, X1, X0, Y):
        """
        Description: computes W with Ridge regression

        Args:
            Wb0, Wb1: internal weights
            X0, X1: input
            Y: output 
        """
        R = np.tanh(self.net.alpha * self.augmul(Wb1, X1) + (1. - self.net.alpha) * self.augmul(Wb0, X0))
        return (Y@R.T) @ np.linalg.solve(R@R.T + self.beta*self.I_r, self.I_r)  

    
    def learn(self, train):
        self.set_stats(train)
        X0, X1, Y = self.correct(train[:, :-2], True), self.correct(train[:, 1:-1], True), self.correct(train[:, 2:], True)
        Y = Y - X1
        with torch.no_grad():
            for i in range(self.net.B):
                Wb0 = self.sampler.sample(self.net.D_r)
                Wb1 = self.sampler.sample(self.net.D_r)
                Wb0 *=  (np.sign(Wb1[:, -1] * np.sign(Wb0[:, -1]))[:, np.newaxis])
                W = self.compute_W(Wb1, Wb0, X1, X0, Y) if i > 0 else self.compute_W(Wb1, Wb0, X1, X1, Y)
                W0 = Wb0[:, :-1]
                b0 = Wb0[:, -1]
                self.net.inner_0[i].weight = nn.Parameter(torch.Tensor(W0))
                self.net.inner_0[i].bias = nn.Parameter(torch.Tensor(b0))
                W1 = Wb1[:, :-1]
                b1 = Wb1[:, -1]
                self.net.inner_1[i].weight = nn.Parameter(torch.Tensor(W1))
                self.net.inner_1[i].bias = nn.Parameter(torch.Tensor(b1))
                self.net.outer[i].weight = nn.Parameter(torch.Tensor(W))

   
            
class BatchDeepRF(chain.BatchDeepRF):
    def __init__(self, train: np.array, test: np.array, *rf_args):
        super().__init__(DeepRF, train, test, *rf_args)        
        