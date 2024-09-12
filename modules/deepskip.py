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

class DeepSkip(nn.Module):
    def __init__(self, D, D_r, B):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.inner = nn.ModuleList([nn.Linear(2*self.D, self.D_r, bias=True) for _ in range(B)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.D, bias=False) for _ in range(B)])

    # @ut.timer  
    def forward(self, x):
        y = torch.hstack((x, x))
        for i in range(self.B):
            y[..., self.D:] += self.outer[i](torch.tanh(self.inner[i](y)))
        return y[..., self.D:]
    
class DeepRF(chain.DeepRF):
    def __init__(self, D_r, B, L0, L1, Uo, beta, name='nn', save_folder='.', normalize=False):
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
        self.net = DeepSkip(self.sampler.dim, D_r, B)
        self.net.to(self.device)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})
        self.sampler.update(torch.vstack((Uo, Uo)))
        self.arch = self.net.__class__
    
    # @ut.timer
    def compute_W(self, Wb, X, Y):
        """
        Description: computes W with Ridge regression

        Args:
            Wb_in: internal weights
            X: input
            Y: output 
        """
        R = torch.tanh(self.augmul(Wb, X))
        # print(Wb_in.shape, X.shape, Y.shape, R.shape)
        return torch.linalg.solve(R@R.T + self.beta*self.I_r, R@Y.T).T

    # @ut.timer
    def learn(self, train, seed):
        # self.set_stats(train)
        X0 = train[:, :-1]
        Y = train[:, 1:]
        X1 = torch.vstack((X0, X0))
        with torch.no_grad():
            for i in range(self.net.B):
                Wb = self.sampler.sample_vec(self.net.D_r, seed=seed)
                self.net.inner[i].weight = nn.Parameter(Wb[:, :-1])
                self.net.inner[i].bias = nn.Parameter(Wb[:, -1])
                self.net.outer[i].weight = nn.Parameter(self.compute_W(Wb, X1, Y-X1[self.net.D:, :]))
                X1[self.net.D:, :] +=  self.net.outer[i](torch.tanh(self.net.inner[i](X1.T))).T



class BatchDeepRF(chain.BatchDeepRF):
    def __init__(self, train: np.array, test: np.array, *drf_args):
        super().__init__(DeepRF, train, test, *drf_args) 

class BetaTester(chain.BetaTester):
    def __init__(self, D_r_list: list, B_list: list, train_list: list, test: np.array, *drf_args):
        super().__init__(DeepRF, D_r_list, B_list, train_list, test, *drf_args) 