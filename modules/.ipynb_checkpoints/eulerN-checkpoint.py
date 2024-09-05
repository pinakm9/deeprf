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

class Euler(nn.Module):
    def __init__(self, D, D_r, B):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.inner = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=True) for _ in range(B)])
        self.outer = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=False) for _ in range(B)])
        
    def forward(self, x):
        y = x + 0.
        for i in range(self.B):
            y += self.outer[i](nn.Tanh()(self.inner[i](y)))
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
        self.net = Euler(self.sampler.dim, D_r, B)
    

    def compute_W(self, Wb_in, X, Y):
        """
        Description: computes W with Ridge regression

        Args:
            Wb_in: internal weights
            X: input
            Y: output 
        """
        R = np.tanh(Wb_in @ np.vstack([X, np.ones(X.shape[-1])]))
        return (Y@R.T) @ np.linalg.solve(R@R.T + self.beta*self.I_r, self.I_r)  

    def init(self):
        Uo = self.sampler.Uo
        X = Uo[:, :self.sampler.Uo.shape[-1]-self.net.B]
        y = torch.Tensor(X.T)
        for i in range(self.net.B):
            Y = Uo[:, i+1:Uo.shape[-1]-(self.net.B-i-1)]
            self.sampler.update(X)
            Wb = self.sampler.sample(self.net.D_r)
            W = self.compute_W(Wb, X, Y-X)
            W_in = Wb[:, :-1]
            bW_in = Wb[:, -1]
            self.net.inner[i].weight = nn.Parameter(torch.Tensor(W_in))
            self.net.inner[i].bias = nn.Parameter(torch.Tensor(bW_in))
            self.net.outer[i].weight = nn.Parameter(torch.Tensor(W))
            y +=  self.net.outer[i](nn.Tanh()(self.net.inner[i](torch.from_numpy(X).T)))
            X = y.detach().numpy().T

 
            
            
           
        
     

