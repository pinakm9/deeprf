# load necessary modules
import numpy as np 
import os, sys, time
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import utility as ut
import sample as sm
# from scipy.linalg import eigh
import pandas as pd
import json
import sample as sm
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

class EulerSnake(nn.Module):
    def __init__(self, D, D_r, B):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.inner_W = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=True) for _ in range(B)])
        self.inner_U = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=True) for _ in range(B)])
        self.outer = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=False) for _ in range(B)])
        
    def forward(self, x):
        y = x + 0.
        for i in range(self.B):
            y += self.outer[i](nn.Tanh()(0.5*(self.inner_W[i](y) + self.inner_U[i](x))))
        return y



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
        self.net = EulerSnake(self.sampler.dim, D_r, B)
    

    def compute_W(self, Wb_in, Ub_in, X1, X, Y):
        """
        Description: computes W with Ridge regression

        Args:
            Wb_in: internal W weights
            Ub_in: internal U weights
            X: U input
            X1: W input
            Y: output 
        """
        X_ = np.vstack([X, np.ones(X.shape[-1])])
        X1_ = np.vstack([X1, np.ones(X1.shape[-1])])
        R = np.tanh(0.5 * (Wb_in @ X1_ + Ub_in @ X_))
        return (Y@R.T) @ np.linalg.solve(R@R.T + self.beta*self.I_r, self.I_r) 

    def init(self):
        N = self.sampler.Uo.shape[-1]
        X = self.sampler.Uo[:, :-1]
        X1 = self.sampler.Uo[:, :-1]
        Y = self.sampler.Uo[:, 1:]
        Z = np.vstack([X, np.ones(X.shape[-1])])
        Z1 = np.vstack([X1, np.ones(X1.shape[-1])])
        for i in range(self.net.B):
            X1 = self.sampler.Uo[:, :-1]
            X = self.sampler.Uo[:, :-1]
            Y = self.sampler.Uo[:, i+1:N-self.net.B+i+1]
            Z = np.vstack([X, np.ones(X.shape[-1])])
            Z1 = np.vstack([X1, np.ones(X1.shape[-1])])
            Wb = self.sampler.sample(self.net.D_r)
            Ub = self.sampler.sample(self.net.D_r) * (np.sign(Wb[:, -1])[:, np.newaxis])
            W_in = Wb[:, :-1]
            bW_in = Wb[:, -1]
            U_in = Ub[:, :-1]
            bU_in = Ub[:, -1]
            W = self.compute_W(Wb, Ub, X1, X, Y-X1)
            self.net.inner_W[i].weight = nn.Parameter(torch.Tensor(W_in))
            self.net.inner_W[i].bias = nn.Parameter(torch.Tensor(bW_in))
            self.net.inner_U[i].weight = nn.Parameter(torch.Tensor(U_in))
            self.net.inner_U[i].bias = nn.Parameter(torch.Tensor(bU_in))
            self.net.outer[i].weight = nn.Parameter(torch.Tensor(W))
            # X1 = W @ np.tanh(0.5 * (Wb @ Z1 + Ub @ Z))
            # Z1 = np.vstack([X1, np.ones(X1.shape[-1])])
        
    