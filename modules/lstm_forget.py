# load necessary modules
import numpy as np 
import os, sys, time
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import utility as ut
import modules.oneshot as sm
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



DTYPE = 'float32'
torch.set_default_dtype(torch.float32)

class Forget_Block(nn.Module):
    def __init__(self, D, D_r):
        super(Forget_Block, self).__init__()
        # d = D if i==0 else D_r
        self.W_f = nn.Linear(D, D_r, bias=True)
        self.U_f = nn.Linear(D_r, D_r, bias=False)
        self.W_g = nn.Linear(D, D_r, bias=True)
        self.U_g = nn.Linear(D_r, D_r, bias=False)
        self.W_r = nn.Linear(D, D_r, bias=True)
        self.U_r = nn.Linear(D_r, D_r, bias=False)
        self.W_s = nn.Linear(D, D_r, bias=True)
        self.U_s = nn.Linear(D_r, D_r, bias=False)
    
    def forward(self, x, h, c):
        f = torch.tanh(self.W_f(x) + self.U_f(h))
        g = torch.tanh(self.W_g(x) + self.U_g(h))
        r = torch.tanh(self.W_r(x) + self.U_r(h))
        s = torch.tanh(self.W_s(x) + self.U_s(h))
        c = f*c + g*s
        return r*torch.tanh(c), c


class LSTM(nn.Module):
    def __init__(self, D, D_r, B):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.blocks = nn.ModuleList([Forget_Block(self.D, self.D_r) for _ in range(B)])
        self.W = nn.Linear(D_r, D, bias=False)
    
        
    def forward(self, x):
        h = torch.zeros(self.D_r)
        c = torch.zeros(self.D_r)
        for block in self.blocks:
            h, c = block(x, h, c)
            # h = self.batch_norm(h)
            # c = self.batch_norm(c)
        return self.W(h)



class DeepRF(chain.DeepRF):
    def __init__(self, D_r, B, L0, L1, Uo, name='nn', save_folder='.'):
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
        super().__init__(D_r, B, L0, L1, Uo, 0., name, save_folder)
        self.net = LSTM(self.sampler.dim, D_r, B)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})


    def forecast(self, u):
        """
        Description: forecasts for a single time step

        Args:
            u: state at current time step 

        Returns: forecasted state
        """
        return self.net(torch.from_numpy(u)).detach().numpy()
    
       
    
    