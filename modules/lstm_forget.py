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

class Forget_Block(nn.Module):
    def __init__(self, D, D_r, i):
        super(Forget_Block, self).__init__()
        d = D if i==0 else D_r
        self.W_f = nn.Linear(D, D_r, bias=True)
        self.U_f = nn.Linear(d, D_r, bias=False)
        self.W_g = nn.Linear(D, D_r, bias=True)
        self.U_g = nn.Linear(d, D_r, bias=False)
        self.W_r = nn.Linear(D, D_r, bias=True)
        self.U_r = nn.Linear(d, D_r, bias=False)
        self.W_s = nn.Linear(D, D_r, bias=True)
        self.U_s = nn.Linear(d, D_r, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, x, h, c):
        f = self.tanh(self.W_f(x) + self.U_f(h))
        g = self.tanh(self.W_g(x) + self.U_g(h))
        r = self.tanh(self.W_r(x) + self.U_r(h))
        s = self.tanh(self.W_s(x) + self.U_s(h))
        c = f*c + g*s
        return r*self.tanh(c), c


class LSTM(nn.Module):
    def __init__(self, D, D_r, B):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.blocks = nn.ModuleList([Forget_Block(self.D, self.D_r, i) for i in range(B)])
        self.W = nn.Linear(D_r, D, bias=False)
    
        
    def forward(self, x):
        h = torch.zeros_like(x)
        c = torch.zeros(x.size(0), self.D_r)
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
        self.X = torch.from_numpy(Uo[:, :-1].T)
        self.Y = torch.from_numpy(Uo[:, 1:].T)

    def count_params(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def forecast(self, u):
        """
        Description: forecasts for a single time step

        Args:
            u: state at current time step 

        Returns: forecasted state
        """
        return self.net(torch.from_numpy(u[np.newaxis, :])).detach().numpy()[0]
    
       
    
    def train(self, epochs=50, learning_rate=1e-4, batch_size=100):
        self.logger.update(start=False, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        train_row = {'epoch': 0, 'loss': 0, 'change':0, 'lr': 0, 'time':0}
        permutation = torch.randperm(self.X.size(0))
        last_loss = np.nan

        for j in range(epochs):
            start = time.time()
            for i in range(0, self.X.size(0), batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = self.X[indices], self.Y[indices]
                loss = torch.sum(self.net(batch_x) - batch_y)**2
                loss.backward()
                optimizer.step()
            end = time.time()
            
            train_row['epoch'] = [j]
            train_row['loss'] = [loss.item()]
            train_row['change'] = [(loss.item() - last_loss) / last_loss]
            train_row['lr'] = [optimizer.param_groups[0]['lr']]
            train_row['time'] = [end - start]
            last_loss = loss.item() 

            self.logger.log(j==0, **train_row)
            self.logger.print(**train_row)



    
    