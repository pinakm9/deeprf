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

class EulerSN(nn.Module):
    def __init__(self, D, D_r, B, alpha=0.9, eps=1e-5):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.alpha = 0.9
        self.eps = eps
        self.inner_W = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=True) for _ in range(B)])
        self.inner_U = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=True) for _ in range(B)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.D, bias=False) for _ in range(B)])
        
    def forward(self, x):
        y = x + 0.
        x1 = (x - torch.mean(x, axis=0)) / (torch.std(x, axis=0) + self.eps)
        for i in range(self.B):
            y1 = (y - torch.mean(y, axis=0)) / (torch.std(y, axis=0) + self.eps)
            y += self.outer[i](torch.tanh(self.alpha * self.inner_W[i](y1) + (1-self.alpha)*self.inner_U[i](x1)))
        return y



class DeepRF(chain.DeepRF):
    def __init__(self, D_r, B, alpha, L0, L1, Uo, beta, name='nn', save_folder='.'):
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
        self.net = EulerSN(self.sampler.dim, D_r, B, alpha)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})

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
        R = np.tanh(self.net.alpha * Wb_in @ X1_ + (1-self.net.alpha) * Ub_in @ X_)
        return (Y@R.T) @ np.linalg.solve(R@R.T + self.beta*self.I_r, self.I_r) 

    def init(self):
        with torch.no_grad():
            Z = self.sampler.Uo[:, self.net.B:]
            Y = self.sampler.Uo[:, self.net.B-1:-1]
            ZdiffY = Z - Y
            for i in range(self.net.B):
                Wb = self.sampler.sample(self.net.D_r)
                Ub = self.sampler.sample(self.net.D_r) 
                Ub *=  (np.sign(Wb[:, -1] * np.sign(Ub[:, -1]))[:, np.newaxis])
                W_in = Wb[:, :-1]
                bW_in = Wb[:, -1]
                U_in = Ub[:, :-1]
                bU_in = Ub[:, -1]

                W = self.compute_W(Wb, Ub, Y, self.sampler.Uo[:, self.net.B-1-i:-(1+i)], ZdiffY)

                self.net.inner_W[i].weight = nn.Parameter(torch.Tensor(W_in))
                self.net.inner_W[i].bias = nn.Parameter(torch.Tensor(bW_in))
                self.net.inner_U[i].weight = nn.Parameter(torch.Tensor(U_in))
                self.net.inner_U[i].bias = nn.Parameter(torch.Tensor(bU_in))
                self.net.outer[i].weight = nn.Parameter(torch.Tensor(W))

    def init_xy(self, x, y):
        x1 = x + 0.
        x_ = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + self.net.eps)
        with torch.no_grad():
            for i in range(self.net.B):
                x1_ = (x1 - np.mean(x1, axis=0)) / (np.std(x1, axis=0) + self.net.eps)

                self.sampler.update(x1_)
                Wb = self.sampler.sample(self.net.D_r)
                self.sampler.update(x_)
                Ub = self.sampler.sample(self.net.D_r) 
         
                # Ub *=  (np.sign(Wb[:, -1] * np.sign(Ub[:, -1]))[:, np.newaxis])
                W_in = Wb[:, :-1]
                bW_in = Wb[:, -1]
                U_in = Ub[:, :-1]
                bU_in = Ub[:, -1]
              
                W = self.compute_W(Wb, Ub, x1_, x_, y-x1)

                self.net.inner_W[i].weight = nn.Parameter(torch.Tensor(W_in))
                self.net.inner_W[i].bias = nn.Parameter(torch.Tensor(bW_in))
                self.net.inner_U[i].weight = nn.Parameter(torch.Tensor(U_in))
                self.net.inner_U[i].bias = nn.Parameter(torch.Tensor(bU_in))
                self.net.outer[i].weight = nn.Parameter(torch.Tensor(W))

                z = self.net.alpha * (W_in @ x1_ + bW_in[:, np.newaxis]) + (1.-self.net.alpha) * (U_in @ x_ + bU_in[:, np.newaxis])
                x1 += W @ np.tanh(z)

            
        