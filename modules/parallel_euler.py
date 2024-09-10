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

class ParallelEuler(nn.Module):
    def __init__(self, D, D_r, B, G=2, I=2):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.G = G 
        self.I = I
        self.Ng = int(self.D / self.G)
        self.idx = 2 * (torch.arange(-self.I, self.I+1).reshape(-1, 1) % self.Ng)
        self.idx = torch.cat([self.idx, self.idx+1], dim=1).flatten()
        self.idx = torch.vstack([(self.idx + 2*i) % self.D for i in range(self.Ng)])
        self.idx0 = torch.arange(0, 2)
        self.idx0 = torch.vstack([(self.idx0 + 2*i) % 40 for i in range(20)])
        self.inner = nn.ModuleList([nn.Linear((2*self.I + 1)*self.G, self.D_r, bias=True)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.G, bias=False)])

    # @ut.timer
    def forward(self, x):
        return x + self.outer[0](torch.tanh(self.inner[0](x[..., self.idx]))).reshape(x.shape)

    
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
        self.net = ParallelEuler(self.sampler.dim, D_r, B)
        self.net.to(self.device)
        self.sampler.update((Uo.T[:-1][..., self.net.idx]).flatten(0, 1).T)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})
        self.arch = self.net.__class__
        
    

    def compute_W(self, Wb_in, X, Y):
        """
        Description: computes W with Ridge regression

        Args:
            Wb_in: internal weights
            X: input
            Y: output 
        """
        R = torch.tanh(self.augmul(Wb_in, X))
        # print(Wb_in.shape, X.shape, Y.shape, R.shape)
        return torch.linalg.solve(R@R.T + self.beta*self.I_r, R@Y.T).T

    
    # @ut.timer
    def learn(self, train, seed):
        X = (train.T[:-1][..., self.net.idx]).flatten(0, 1).T
        Y = (train.T[1:][..., self.net.idx0] - train.T[:-1][..., self.net.idx0]).flatten(0, 1).T
        indices = torch.randperm(X.shape[1])
        X = X[:, indices[:100000]]
        Y = Y[:, indices[:100000]]
        print(X.shape, Y.shape)
        self.set_stats(X)
        with torch.no_grad():
            Wb = self.sampler.sample_vec(self.net.D_r, seed=seed)
            self.net.inner[0].weight = nn.Parameter(Wb[:, :-1])
            self.net.inner[0].bias = nn.Parameter(Wb[:, -1])
            self.net.outer[0].weight = nn.Parameter(self.compute_W(Wb, X, Y))


class BatchDeepRF(chain.BatchDeepRF):
    def __init__(self, train: np.array, test: np.array, *drf_args):
        super().__init__(DeepRF, train, test, *drf_args) 

class BetaTester(chain.BetaTester):
    def __init__(self, D_r_list: list, B_list: list, train_list: list, test: np.array, *drf_args):
        super().__init__(DeepRF, D_r_list, B_list, train_list, test, *drf_args) 