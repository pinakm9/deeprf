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


class SeqRFM(nn.Module):
    def __init__(self, D, D_r, B):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.inner = nn.ModuleList([nn.Linear(self.D*self.B, self.D_r, bias=True) for _ in range(1)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.D, bias=False) for _ in range(1)])
        
    def forward(self, x):
        y = x + 0.
        for i in range(1):
            y = self.outer[i](torch.tanh(self.inner[i](y)))
        return y
    

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
        self.net = SeqRFM(int(self.sampler.dim/B), D_r, B)
        self.net.to(self.device)
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
        return torch.linalg.solve(R@R.T + self.beta*self.I_r, R@Y.T).T
    
    
    @ut.timer
    def learn(self, train, seed):
        X, Y = self.parse_data(train)
        self.set_stats(X)
        with torch.no_grad():
            for i in range(1):
                Wb = torch.tensor(self.sampler.sample(self.net.D_r, seed=seed))
                self.net.inner[i].weight = nn.Parameter(Wb[:, :-1])
                self.net.inner[i].bias = nn.Parameter(Wb[:, -1])
                self.net.outer[i].weight = nn.Parameter(self.compute_W(Wb, X, Y))

    def parse_data(self, Uo):
        d, n = Uo.shape
        X = torch.zeros((d*self.net.B, n-self.net.B-1))
        Y = torch.zeros((d, n-self.net.B-1))
        for i in range(n-self.net.B-1):
            X[:, i] = Uo.T[i:i+self.net.B].flatten()
            Y[:, i] = Uo.T[i+self.net.B]
        return X, Y
    
    def multistep_forecast(self, u, n_steps):
        """
        Description: forecasts for multiple time steps

        Args:
            u: state at current time step 
            n_steps: number of steps to propagate u

        Returns: forecasted state
        """ 
        with torch.no_grad():
            trajectory = torch.zeros((self.net.D, n_steps))
            trajectory[:, 0] = u[:self.net.D]
            for step in range(n_steps - 1):
                v = self.forecast(u)
                u = torch.hstack((u[self.net.D:], v))
                trajectory[:, step+1] = v
            return trajectory


            
class BatchDeepRF(chain.BatchDeepRF):
    def __init__(self, train: np.array, test: np.array, *drf_args):
        super().__init__(DeepRF, train, test, *drf_args) 

    def get_tau_f(self, drf, test, error_threshold=0.05, dt=0.02, Lyapunov_time=1/0.91):
        validation_points = test.shape[-1]
        Vo = test.T
        u = Vo[:drf.net.B].flatten()
        se_flag, nmse_flag = False, False
        tau_f_se, tau_f_nmse = validation_points-1, validation_points-1
        se, nmse = 0., 0.

        for i in range(1, validation_points-drf.net.B):
            v0 = Vo[drf.net.B+i-1]
            v = drf.forecast(u)
            u = torch.hstack((u[drf.net.D:], v))
            difference = v0 - v
            se_ = torch.sum(difference**2) / torch.sum(v0**2)
            nmse_ = ((difference / self.std)**2).mean()
            if se_ > error_threshold and se_flag == False:
                tau_f_se = i-1
                se_flag = True
            if nmse_ > error_threshold and nmse_flag == False:
                tau_f_nmse = i-1
                nmse_flag = True
            if se_flag and nmse_flag:
                break
            else:
                se, nmse = se_ + 0., nmse_ + 0.

        tau_f_nmse *= (dt / (drf.net.B * Lyapunov_time))
        tau_f_se *= (dt / (drf.net.B * Lyapunov_time))
        return tau_f_nmse, tau_f_se, nmse, se

class BetaTester(chain.BetaTester):
    def __init__(self, D_r_list: list, B_list: list, train_list: list, test: np.array, *drf_args):
        super().__init__(DeepRF, D_r_list, B_list, train_list, test, *drf_args)   