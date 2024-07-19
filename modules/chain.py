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
import log
import learning_rate as rate

DTYPE = 'float64'
torch.set_default_dtype(torch.float64)

class Chain(nn.Module):
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
            y = x + self.outer[i](nn.Tanh()(self.inner[i](y)))
        return y



class DeepRF:
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
        self.device = ("cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu")
        self.sampler = sm.GoodRowSampler(L0, L1, Uo)
        self.net = Chain(self.sampler.dim, D_r, B)
        self.beta = beta
        self.name = name
        self.I_r = np.identity(D_r)
        self.save_folder = save_folder 
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        config = {'D': self.sampler.dim, 'D_r': D_r, 'B': B, 'name': name}
        config |= {'L0': L0, 'L1': L1, 'beta': beta, 'training_points': Uo.shape[-1], 'parameters': self.count_params()}
        self.logger = log.Logger(self.save_folder)
        self.logger.update(start=True, kwargs=config)
        

        self.mean = 0
        self.std = Uo.T.std(axis=0)

        self.X = torch.from_numpy((Uo[:, :-1].T -0.) / 1.)
        self.Y = torch.from_numpy((Uo[:, 1:].T -0.) /1.)


    def count_params(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    
    def forecast(self, u):
        """
        Description: forecasts for a single time step

        Args:
            u: state at current time step 

        Returns: forecasted state
        """
        return (self.net(torch.from_numpy(u)).detach().numpy()) 
    
    def loss_wo_reg(self):
        return torch.sum((self.net(self.X) - self.Y)**2) 
    
    
    def multistep_forecast(self, u, n_steps):
        """
        Description: forecasts for multiple time steps

        Args:
            u: state at current time step 
            n_steps: number of steps to propagate u

        Returns: forecasted state
        """    
        trajectory = np.zeros((self.sampler.dim, n_steps))
        trajectory[:, 0] = u
        for step in range(n_steps - 1):
            u = self.forecast(u)
            trajectory[:, step+1] = u
        return trajectory
    
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
        X = self.sampler.Uo[:, :-1]
        Y = self.sampler.Uo[:, 1:] - X
        Z = np.vstack([X, np.ones(X.shape[-1])])
        for i in range(self.net.B):
            Wb = self.sampler.sample(self.net.D_r)
            W_in = Wb[:, :-1]
            b_in = Wb[:, -1]
            W = self.compute_W(Wb, X, Y)
            self.net.inner[i].weight = nn.Parameter(torch.Tensor(W_in))
            self.net.inner[i].bias = nn.Parameter(torch.Tensor(b_in))
            self.net.outer[i].weight = nn.Parameter(torch.Tensor(W))
            X = self.sampler.Uo[:, :-1] + W @ np.tanh(Wb @ Z)
            Z = np.vstack([X, np.ones(X.shape[-1])])
        
    def read(self):
        return pd.read_csv(f'{self.save_folder}/train_log.csv')
    
    
    @ut.timer
    def compute_tau_f_(self, test, error_threshold=0.05, dt=0.02, Lyapunov_time=1/0.91):
        """
        Description: computes forecast time tau_f for the computed surrogate model

        Args:
            test: list of test trajectories
        """
        tau_f_se, tau_f_rmse = np.zeros(len(test)), np.zeros(len(test))
        self.validation_points = test.shape[-1]
        self.error_threshold = error_threshold
        self.dt = dt
        self.Lyapunov_time = Lyapunov_time
        se, rmse = np.zeros(len(test)), np.zeros(len(test))
        for validation_index in range(len(test)):
            validation_ = test[validation_index]
            prediction = self.multistep_forecast(validation_[:, 0], self.validation_points)
            se_ = np.linalg.norm(validation_ - prediction, axis=0)**2 / np.linalg.norm(validation_, axis=0)**2
            mse_ = np.cumsum(se_) / np.arange(1, len(se_)+1)
    
            
            l = np.argmax(mse_ > self.error_threshold)
            if l == 0:
                tau_f_rmse[validation_index] = self.validation_points
            else:
                tau_f_rmse[validation_index] = l-1


            l = np.argmax(se_ > self.error_threshold)
            if l == 0:
                tau_f_se[validation_index] = self.validation_points
            else:
                tau_f_se[validation_index] = l-1
            
            rmse[validation_index] = np.sqrt(mse_[-1])
            se[validation_index] = se_.mean()
 
            
        
        tau_f_rmse *= (self.dt / self.Lyapunov_time)
        tau_f_se *= (self.dt / self.Lyapunov_time)

        np.savetxt(f'{self.save_folder}/tau_f_rmse.csv', tau_f_rmse, delimiter=',')
        np.savetxt(f'{self.save_folder}/tau_f_se.csv', tau_f_se, delimiter=',') 

        if len(test) == 1:
            return tau_f_rmse[0], tau_f_se[0], rmse[0], se[0]
        else:
            return tau_f_rmse, tau_f_se, rmse, se
    

    def get_save_idx(self):
        return sorted([int(f.split('_')[-1]) for f in os.listdir(self.save_folder) if f.startswith(self.name)])
    
    def save(self, idx):
        torch.save(self.net, self.save_folder + f'/{self.name}_{idx}')
    
    def load(self, idx):
        self.net = torch.load(self.save_folder + f'/{self.name}_{idx}')


    def train(self, epochs=50, learning_rate=1e-4, batch_size=100, **rate_scheduler_params):
        self.logger.update(start=False, kwargs={"epochs":epochs, "learning_rate":learning_rate, "batch_size":batch_size})
        self.logger.update(start=False, kwargs=rate_scheduler_params)
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        lr_scheduler = rate.SimpleLRScheduler(optimizer, **rate_scheduler_params)
        # lr_scheduler = rate.PiecewiseLinearScheduler(optimizer, **rate_scheduler_params)
        
        train_row = {'epoch': 0, 'loss': 0, 'change':0, 'lr': 0, 'time':0}
        permutation = torch.randperm(self.X.size(0))
        last_loss = np.nan
        num_batchs = np.ceil(self.X.size(0) / batch_size)
       

        for epoch in range(epochs):
            start = time.time()
            avg_loss = 0.
            for i in range(0, self.X.size(0), batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = self.X[indices], self.Y[indices]
                loss = torch.sum((self.net(batch_x) - batch_y)**2)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            end = time.time()
            avg_loss /= num_batchs 

            train_row['epoch'] = [epoch]
            train_row['loss'] = [avg_loss]
            train_row['change'] = [(avg_loss - last_loss) / last_loss]
            train_row['lr'] = [optimizer.param_groups[0]['lr']]
            train_row['time'] = [end - start]
            last_loss = avg_loss 

            self.logger.log(epoch==0, **train_row)
            self.logger.print(**train_row)
            lr_scheduler.step(train_row['change'][0])

        self.save(epoch)


    
    

