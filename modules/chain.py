# load necessary modules
import numpy as np 
import os, sys, time
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import utility as ut
from joblib import Parallel, delayed, parallel_config

# from scipy.linalg import eigh
import pandas as pd
import oneshot as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter 
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch.nn.functional as tfn
import torch.optim.lr_scheduler as lr_scheduler
import log
import glob
import learning_rate as rate
from joblib import Parallel, delayed
import itertools

DTYPE = 'float64'
torch.set_default_dtype(torch.float64)

class Chain(nn.Module):
    def __init__(self, D, D_r, B):
        super().__init__()
        self.D = D
        self.D_r = D_r
        self.B = B
        self.inner = nn.ModuleList([nn.Linear(self.D, self.D_r, bias=True) for _ in range(B)])
        self.outer = nn.ModuleList([nn.Linear(self.D_r, self.D, bias=False) for _ in range(B)])
        
    def forward(self, x):
        y = x + 0.
        for i in range(self.B):
            y = x + self.outer[i](nn.Tanh()(self.inner[i](y)))
        return y
    
    def multistep_forecast(self, u, n_steps):
        trajectory = torch.zeros((self.sampler.dim, n_steps))
        trajectory[:, 0] = u
        for step in range(n_steps - 1):
            u = self.forward(u)
            trajectory[:, step+1] = u
        return trajectory



class DeepRF:
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # "mps" if torch.backends.mps.is_available() else "cpu")
        Uo.to(self.device)
        self.normalize = normalize
        self.sampler = sm.GoodRowSampler(L0, L1, Uo)
     
        Uo_ = self.set_stats(Uo)
        self.X = Uo_[:, :-1].T
        self.Y = Uo_[:, 1:].T
        self.net = Chain(self.sampler.dim, D_r, B)
        self.net.to(self.device)
        self.beta = beta
        self.name = name
        self.I_r = torch.eye(D_r, device=self.device)
        self.save_folder = save_folder 
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        config = {'D': self.sampler.dim, 'D_r': D_r, 'B': B, 'name': name}
        config |= {'L0': L0, 'L1': L1, 'beta': beta, 'normalize': normalize}
        self.logger = log.Logger(self.save_folder)
        self.logger.update(start=True, kwargs=config)
        self.logger.update(start=False, kwargs={'parameters': self.count_params()})
        self.arch = self.net.__class__
    
    # @ut.timer
    def set_stats(self, train):
        self.mean = torch.mean(train, axis=1) if self.normalize else torch.zeros(train.shape[0], device=self.device)
        self.std = torch.std(train, axis=1) + 0e-2 if self.normalize else torch.ones(train.shape[0], device=self.device)
        Uo_ = (train - self.mean[:, None]) / self.std[:, None] 
        self.sampler.update(Uo_)
        return Uo_

    def count_params(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
    
    
    def correct_(self, u, flag: bool):
        return (u - self.mean) / self.std if flag  else u * self.std + self.mean 
    
    
    def correct(self, u, flag: bool):
        return (u - self.mean[:, None]) / self.std[:, None] if flag\
                else u * self.std[:, None] + self.mean[:, None] 

    
    def forecast(self, u):
        """
        Description: forecasts for a single time step

        Args:
            u: state at current time step 

        Returns: forecasted state
        """
        # with torch.no_grad():
        #     if correction_flag:
        #         return self.correct_(self.net(self.correct_(u, True)), False)
        #     else:
        #         return self.net(u)
        # with torch.no_grad():
        return self.net(u)

    
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
        with torch.no_grad():
            if len(u.shape) < 2:
                trajectory = torch.zeros((n_steps, self.sampler.dim), device=self.device)

            else:
                trajectory = torch.zeros((n_steps, u.shape[0], u.shape[1]), device=self.device)
            trajectory[0] = u
            for step in range(n_steps - 1):
                u = self.forecast(u)
                trajectory[step+1] = u
            return torch.movedim(trajectory, 0, -1)
    
    def compute_W(self, Wb_in, X, Y):
        """
        Description: computes W with Ridge regression

        Args:
            Wb_in: internal weights
            X: input
            Y: output 
        """
        R = torch.tanh(Wb_in @ torch.vstack([X, torch.ones(X.shape[-1])]))
        return (Y@R.T) @ torch.linalg.solve(R@R.T + self.beta*self.I_r, self.I_r) 
    
    def augmul(self, Wb, X):
        # print(Wb.device, X.device)
        return Wb @ torch.vstack([X, torch.ones(X.shape[-1], device=self.device)])
    

    def init(self):
        X0 = self.X.T
        X = X0 + 0.
        Y = self.Y.T - X0
        one = torch.ones(X.shape[-1])
        Z = torch.vstack([X, one])
        for i in range(self.net.B):
            Wb = torch.tensor(self.sampler.sample(self.net.D_r))
            self.net.inner[i].weight = nn.Parameter(Wb[:, :-1])
            self.net.inner[i].bias = nn.Parameter(Wb[:, -1])
            W = self.compute_W(Wb, X, Y)
            self.net.outer[i].weight = nn.Parameter(W)
            X = X0 + W @ torch.tanh(Wb @ Z)
            Z = torch.vstack([X, one])

    def learn(self, train):
        X0 = self.X.T
        X = X0 + 0.
        Y = self.Y.T - X0
        one = torch.ones(X.shape[-1])
        Z = torch.vstack([X, one])
        for i in range(self.net.B):
            Wb = torch.tensor(self.sampler.sample(self.net.D_r))
            self.net.inner[i].weight = nn.Parameter(Wb[:, :-1])
            self.net.inner[i].bias = nn.Parameter(Wb[:, -1])
            W = self.compute_W(Wb, X, Y)
            self.net.outer[i].weight = nn.Parameter(W)
            X = X0 + W @ torch.tanh(Wb @ Z)
            Z = torch.vstack([X, one])
        
    def read(self):
        return pd.read_csv(f'{self.save_folder}/train_log.csv')
    
    
    @ut.timer
    def compute_tau_f(self, test, error_threshold=0.05, dt=0.02, Lyapunov_time=1/0.91, name=''):
        """
        Description: computes forecast time tau_f for the computed surrogate model

        Args:
            test: list of test trajectories
        """
        self.validation_points = test.shape[-1]
        self.error_threshold = error_threshold
        self.dt = dt
        self.Lyapunov_time = Lyapunov_time
        std = torch.movedim(test, -2, -1).reshape(-1, test.shape[1]).std(axis=0)


        difference = test - self.multistep_forecast(test[:,:,0], self.validation_points)
        se = torch.sum(difference**2, axis=1) / torch.sum(test**2, axis=1)
        nmse = torch.mean((difference/std[None, :, None])**2, axis=1)

        
        l = torch.argmax((se > self.error_threshold).to(torch.long), dim=1)
        l[l==0] = self.validation_points
        l[l>0] -= 1
        tau_f_se = l * (self.dt / self.Lyapunov_time)


        l = torch.argmax((nmse > self.error_threshold).to(torch.long), dim=1)
        l[l==0] = self.validation_points
        l[l>0] -= 1
        tau_f_nmse = l * (self.dt / self.Lyapunov_time)
        
        if name != '':
            connector = '' if name == '' else '_'
            np.savetxt(f'{self.save_folder}/{name}{connector}tau_f_nmse.csv', tau_f_nmse.cpu().numpy(), delimiter=',')
            np.savetxt(f'{self.save_folder}/{name}{connector}tau_f_se.csv', tau_f_se.cpu().numpy(), delimiter=',') 

        return tau_f_nmse, tau_f_se, nmse, se
    
    def read_tau_f(self, name=''):
        connector = '' if name == '' else '_'
        tau_f_se = np.genfromtxt(f'{self.save_folder}/{name}{connector}tau_f_se.csv', delimiter=',')
        return torch.tensor(tau_f_se)

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


    def get_block_params(self, block_index):
        return self.net.inner[block_index].weight, self.net.inner[block_index].bias, self.net.outer[block_index].weight

    def block_update_(self, block_index, W_in, b_in, W):
        self.net.inner[block_index].weight = nn.Parameter(W_in)
        self.net.inner[block_index].bias = nn.Parameter(b_in)
        self.net.outer[block_index].weight = nn.Parameter(W)

    def block_update(self, block_idx, W_in, b_in, W):
        for i, j in enumerate(block_idx):
            self.block_update_(j, W_in[i], b_in[i], W[i])

    def get_block_model(self, block_index):
        model = self.arch(self.sampler.dim, self.net.D_r, 1)
        W_in, b_in, W = self.get_block_params(block_index)
        self.net.inner[block_index].weight = nn.Parameter(W_in) + 0.
        self.net.inner[block_index].bias = nn.Parameter(b_in) + 0.
        self.net.outer[block_index].weight = nn.Parameter(W) + 0.
        return model
    

    







class BatchDeepRF:
    def __init__(self, drf_type, train: np.array, test: np.array, *drf_args):
        self.train = train
        self.test = test
        self.drf_args = list(drf_args)
        self.drf_type = drf_type
        self.drf = self.drf_type(*self.drf_args)
        self.std = torch.std(train, axis=1)
    
    # @ut.timer
    def get_tau_f(self, drf, test, error_threshold=0.05, dt=0.02, Lyapunov_time=1/0.91):
        with torch.no_grad():
            validation_points = test.shape[-1]
            u_hat = test[:, 0] + 0.
            se_flag, nmse_flag = False, False
            tau_f_se, tau_f_nmse = validation_points-1, validation_points-1
            se, nmse = 0., 0.

            for i in range(1, validation_points):
                u_hat = drf.forecast(u_hat)
                difference = test[:, i] - u_hat
                se_ = torch.sum(difference**2) / torch.sum(test[:, i]**2)
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

            tau_f_nmse *= (dt / Lyapunov_time)
            tau_f_se *= (dt / Lyapunov_time)
            return tau_f_nmse, tau_f_se, nmse, se
        


    def compute_tau_f(self, drf, test, **tau_f_kwargs):
        tau_f_nmse, tau_f_se, nmse, se = drf.compute_tau_f(test, **tau_f_kwargs)
        return float(tau_f_nmse[0]), float(tau_f_se[0]), float(nmse[0]), float(se[0])
    


    def run_single(self, exp_idx:int, model_seed: int, train_idx: int, test_idx: int, **tau_f_kwargs):
        deep_rf = self.drf_type(*self.drf_args)
        deep_rf.learn(self.train[:, train_idx:train_idx+self.training_points], model_seed)
        return [exp_idx, model_seed, train_idx, test_idx] + list(self.get_tau_f(deep_rf, self.test[test_idx], **tau_f_kwargs))
    

    @ut.timer
    def run(self, training_points: int, n_repeats: int, batch_size: int, save_best=True, **tau_f_kwargs):
        self.tau_f_kwargs = tau_f_kwargs
        file_path = '{}/batch_data.csv'.format(self.drf.save_folder)
        if os.path.exists(file_path):
            os.remove(file_path)
        columns = ['l', 'model_seed', 'train_index', 'test_index',\
                   'tau_f_nmse', 'tau_f_se', 'nmse', 'se']
        
        model_seeds: int = np.random.randint(1e8, size=n_repeats)
        self.training_points: int = training_points
        self.n_repeats: int = n_repeats
        train_indices = np.random.randint(self.train.shape[1] - self.training_points, size=self.n_repeats)
        test_indices = np.random.randint(len(self.test), size=self.n_repeats)
        num_batches = int(n_repeats/batch_size)
        exp_indices = list(range(self.n_repeats))
        k = 0 
        for batch in range(num_batches):
            print(f'Running experiments for batch {batch}...')
            start = time.time()
            results = [self.run_single(k, model_seeds[k], train_indices[k], test_indices[k], **tau_f_kwargs) for k in exp_indices[k:k+batch_size]]
            pd.DataFrame(results, columns=columns, dtype=float)\
                        .to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
            end = time.time()
            print(f'Time taken = {end-start:.2E}s')
            del results
            k += batch_size
        if save_best:
            print('Saving the best and worst models ...')
            data = self.get_data()
            idx = np.argmax(data['tau_f_nmse'].to_numpy())
            train_idx = int(data['train_index'][idx])
            deep_rf = self.drf_type(*self.drf_args)
            deep_rf.learn(self.train[:, train_idx:train_idx+self.training_points], int(data['model_seed'][idx]))
            deep_rf.save('best')

            idx = np.argmin(data['tau_f_nmse'].to_numpy())
            train_idx = int(data['train_index'][idx])
            deep_rf = self.drf_type(*self.drf_args)
            deep_rf.learn(self.train[:, train_idx:train_idx+self.training_points], int(data['model_seed'][idx]))
            deep_rf.save('worst')

    
    def get_data(self):
        return pd.read_csv('{}/batch_data.csv'.format(self.drf.save_folder))
    

    def get_beta_data(self):
        return pd.read_csv(f'{self.drf.save_folder}/beta_test_D_r-{self.drf.net.D_r}_depth-{self.drf.net.B}.csv')

    def try_beta(self, beta, model_seed, train_idx, test_idx, **tau_f_kwargs):
        self.drf_args[5] = beta
        drf = self.drf_type(*self.drf_args)
        drf.learn(self.train[:, train_idx:train_idx+self.training_points], model_seed)
        del drf
        return self.get_tau_f(drf, self.test[test_idx], **tau_f_kwargs) #[beta, model_seed, train_idx, test_idx] +
       
    
    @ut.timer
    def search_beta(self, negative_log10_range:list, resolution:int, n_repeats: int, training_points: int, **tau_f_kwargs):
        self.tau_f_kwargs = tau_f_kwargs
        # file_path = f'{self.drf.save_folder}/beta_test_D_r-{self.drf.net.D_r}_depth-{self.drf.net.B}.csv'
        file_path_agg = f'{self.drf.save_folder}/beta_D_r-{self.drf.net.D_r}_depth-{self.drf.net.B}.csv'
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        if os.path.exists(file_path_agg):
            os.remove(file_path_agg)
        # columns = ['beta', 'seed', 'train_index',\
        #             'test_index', 'tau_f_nmse', 'tau_f_se', 'nmse', 'se']
        columns_agg = ['beta', 'tau_f_nmse_mean', 'tau_f_se_mean', 'nmse_mean', 'se_mean',\
                       'tau_f_nmse_std', 'tau_f_se_std', 'nmse_std', 'se_std']
        
        array = np.linspace(1., 10., resolution, endpoint=False)
        betas = np.array([array * 10.**(-i) for i in range(int(negative_log10_range[0]), int(negative_log10_range[1]+1), 1)]).flatten()
        betas.sort()
        self.n_exps: int = len(betas) * n_repeats

        model_seeds: int = np.random.randint(1e8, size=self.n_exps)
        self.training_points: int = training_points
        self.n_repeats = n_repeats
        train_indices = np.random.randint(self.train.shape[1] - self.training_points, size=self.n_exps)
        test_indices = np.random.randint(len(self.test), size=self.n_exps)

  
        k = 0
        r = torch.zeros(size=(4, n_repeats), device=self.drf.device)
        for beta in betas:
            print(f'Running experiments for (D_r, B, beta) = ({self.drf_args[0]}, {self.drf_args[1]}, {beta:.2E})...')
            start = time.time()
            for j in range(k, k+n_repeats):
                r[:, j] = self.try_beta(beta, model_seeds[j], train_indices[j], test_indices[j], **tau_f_kwargs)
            results = [[beta, float(r[0].mean()), float(r[1].mean()), float(r[2].mean()), float(r[3].mean()),\
                        float(r[0].std()), float(r[1].std()), float(r[2].std()), float(r[3].std())]]
            # results_agg = beta
            # results_agg[:, [1, 2, 3, 4]] = np.mean(results[:, [5, 6, 7, 8]], axis=0) 
            # results_agg[:, [5, 6, 7, 8]] = np.std(results[:, [5, 6, 7, 8]], axis=0)
            # print(results)
            # pd.DataFrame(results, columns=columns, dtype=float)\
            #             .to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
            pd.DataFrame(results, columns=columns_agg, dtype=float)\
                        .to_csv(file_path_agg, mode='a', index=False, header=not os.path.exists(file_path_agg))
            end = time.time()
            print(f'Time taken = {end-start:.2E}s')
            k += n_repeats

        # data = self.get_beta_data()
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # ax.plot(np.log10(data['beta'].to_numpy()), data['tau_f_nmse_mean'])
        # ax.set_ylabel('Mean VPT')#r'$\mathbb{E}[\text{VPT}]$') Colab friendly
        # ax.set_xlabel('beta (log scale in base 10)')#r'$\log_{10}(\beta)$')
        # plt.savefig(f'{self.drf.save_folder}/beta_test_D_r-{self.drf.net.D_r}_depth-{self.drf.net.B}.png', bbox_inches='tight', dpi=300)
        

    def get_model(self, idx):
        self.drf.load(idx)
        return self.drf



class BetaTester:

    def __init__(self, drf_type, D_r_list: list, B_list: list, train_list: list, test: np.array, *drf_args):
        self.train_list = train_list
        self.test = test
        self.D_r_list = D_r_list 
        self.B_list = B_list
        self.drf_args = list(drf_args)
        self.drf_type = drf_type



    def get_drf_args(self, D_r, B):
        new_drf_args = self.drf_args.copy()
        new_drf_args[:2] = D_r, B 
        # new_drf_args[7] += f'/D_r-{D_r}/depth-{B}'
        new_drf_args[4] = self.train_list[self.B_list.index(B)]
        return new_drf_args


    def search_beta(self, negative_log10_range:list, resolution:int, n_repeats: int,\
                    training_points: int, **tau_f_kwargs):
        # file_path = '{}/beta_grid.csv'.format(self.drf_args[7])
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        # columns = ['D_r', 'B', 'beta_star']
        for D_r in self.D_r_list:
            for B in self.B_list:
                drf_args = self.get_drf_args(D_r, B)
                batch = BatchDeepRF(self.drf_type, drf_args[4], self.test, *drf_args)
                batch.search_beta(negative_log10_range, resolution, n_repeats, training_points, **tau_f_kwargs)
                # data = batch.get_beta_data()
                # idx = np.argmax(data['tau_f_nmse_mean'].to_numpy())
                # pd.DataFrame([[D_r, B, data['beta'].to_numpy()[idx]]], columns=columns, dtype=float)\
                #         .to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
        


    def search_beta_(self, negative_log10_range:list, resolution:int, n_repeats: int,\
                    training_points: int, **tau_f_kwargs):
        for i, config in enumerate(itertools.product(self.D_r_list, self.B_list)):
            drf_args = self.get_drf_args(*config)
            batch = BatchDeepRF(self.drf_type, drf_args[4], self.test, *drf_args)
            batch.search_beta(negative_log10_range[i], resolution, n_repeats, training_points, **tau_f_kwargs)
    

    
        
def agg_beta(folder):
    files = glob.glob(folder + '/*.csv')
    agg = []
    for file in files:
        filename = os.path.basename(file)
        if filename != 'beta.csv':
            D_r = int(filename[9:].split('_')[0])
            data = pd.read_csv(file)
            idx = np.argmax(data['tau_f_nmse_mean'])
            agg.append([D_r] + data.iloc[idx].to_list())
    pd.DataFrame(sorted(agg, key=lambda x:x[0]), columns=['D_r'] + list(data.columns))\
                .to_csv(folder + '/beta.csv', index=False, mode='w')




        