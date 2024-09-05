# load necessary modules
import numpy as np 
import os, sys 
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import utility as ut
import oneshot as sm
# from scipy.linalg import eigh
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import torch
from torch.autograd.functional import jacobian

def uniform_W_in(D, D_r, w):
    return np.random.uniform(low=-w, high=w, size=(D_r, D)) 

def uniform_b_in(D_r, b):
    return np.random.uniform(low=-b, high=b, size=D_r)



class SurrogateModel_LR:
    """
    Description: A class for computing surrogate models for deterministic dynamical systems based on RAFDA paper https://doi.org/10.1016/j.physd.2021.132911
    """
    def __init__(self, D, D_r, W_in_fn, b_in_fn, W=None, name="RFM"):
        """
        Args: 
            D: dimension of the underlying dynamical system
            D_r: dimension of the reservoir
            W_in_fn: a function for generating W_in
            b_in_fn: a function for generating b_in 
            W: value of the W matrix, default = None
        """
        self.D = D
        self.D_r = D_r
        self.name = name
        if isinstance(W_in_fn, np.ndarray):
            self.W_in = W_in_fn
        elif isinstance(W_in_fn, (float, int)):
            self.W_in = uniform_W_in(D, D_r, W_in_fn)
        else:
            self.W_in = W_in_fn(D, D_r)

        if isinstance(b_in_fn, np.ndarray):
            self.b_in = b_in_fn
        elif isinstance(b_in_fn, (float, int)):
            self.b_in = uniform_b_in(D_r, b_in_fn)
        else:
            self.b_in = b_in_fn(D_r)
        self.W = W
        self.identity_r = np.identity(D_r)

    def phi(self, uo):
        """
        Description: calculates random features

        Args:
            uo: observation vector
    
        Returns: tanh(W_in uo + b_in)
        """
        return np.tanh(self.W_in @ uo + self.b_in)

    def compute_W(self, obs, beta):
        """
        Description: computes W with Ridge regression

        Args:
            obs: observation matrix of shape DxN, referred to as U^o in the RAFDA paper
            beta: regularization paramter     
        """
        self.beta = beta
        Uo = obs[:, 1:]#np.hstack([uo.reshape(-1, 1) for uo in obs])
        R = np.tanh(self.W_in @ obs[:, :-1] + self.b_in[:, np.newaxis])#np.hstack([self.phi(uo).reshape(-1, 1) for uo in obs.T])
        self.W = (Uo@R.T) @ np.linalg.solve(R@R.T + beta*self.identity_r, self.identity_r) 
        # return self.W
        
    def compute_W_xy(self, x, y, beta):
        """
        Description: computes W with Ridge regression

        Args:
            x: input
            y: output
            beta: regularization paramter     
        """
        self.beta = beta
        R = np.tanh(self.W_in @ x + self.b_in[:, np.newaxis])#np.hstack([self.phi(uo).reshape(-1, 1) for uo in obs.T])
        self.W = (y@R.T) @ np.linalg.solve(R@R.T + beta*self.identity_r, self.identity_r)
        self.final_loss = np.sum((self.W@R-y)**2) + beta*np.sum(self.W**2)  
        # return self.W

    def validation_error(self, x, y):
        self.avg_validation_error = np.sum((self.forecast_m(x) - y)**2) / len(x)
    
    def forecast(self, u):
        """
        Description: forecasts for a single time step

        Args:
            u: state at current time step 

        Returns: forecasted state
        """     
        return self.W @ self.phi(u)
    
    def get_tensor_map(self):
        """
        Description: returns a torch-tensorized version of the one step surrogate map 
        """
        W_in = torch.from_numpy(self.W_in)
        b_in = torch.from_numpy(self.b_in)
        W = torch.from_numpy(self.W)
        def psi(x):
            return W@torch.tanh(W_in@x+b_in)
        return psi
        
    
    def forecast_m(self, u):
        """
        Description: one step forecasts for multiple inputs
        """
        return self.W @ np.tanh(self.W_in @ u + self.b_in[:, np.newaxis])
    
    
    def multistep_forecast(self, u, n_steps):
        """
        Description: forecasts for multiple time steps

        Args:
            u: state at current time step 
            n_steps: number of steps to propagate u

        Returns: forecasted state
        """    
        trajectory = np.zeros((self.D, n_steps))
        trajectory[:, 0] = u
        for step in range(n_steps - 1):
            u = self.forecast(u)
            trajectory[:, step+1] = u
        return trajectory
    

    def error(self, true, predicted):
        """
        Description: computes squared l2 error at each time step for a predicted trajectory

        Arg:
            true: the true trajectory (every column corresponds to a unique time step)
            predicted: the predicted trajectorie

        Returns: the computed squared l2 error
        """
        se = np.linalg.norm(true - predicted, axis=0)**2 / np.linalg.norm(true, axis=0)**2
        mse = np.cumsum(se) / np.arange(1, len(se)+1) 
        return se, mse
    
    # @ut.timer
    def compute_forecast_time(self, trajectories, error_threshold, dt, Lyapunov_time=1/0.91):
        """
        Description: computes forecast time tau_f for the computed surrogate model

        Args:
            trajectories: true trajectories to compute forecast errors
            error_threshold: threshold to determine forecast time
            dt: time step of the surrogate model
            Lyapunov_time: Lyapunov time for the underlying dynamical system, default=1
        
        Returns: computed forecast times in Lyapunov units and errors
        """
        if len(trajectories.shape) < 3:
            trajectories = trajectories[np.newaxis, :, :]
        tau_f_se = np.zeros(len(trajectories))
        tau_f_mse = np.zeros(len(trajectories))
        N = trajectories[0].shape[-1]
        ses, mses = np.zeros((len(trajectories), N)), np.zeros((len(trajectories), N))
        for i, trajectory in enumerate(trajectories):
            predicted = self.multistep_forecast(u=trajectory.T[0], n_steps=len(trajectory.T))
            ses[i, :], mses[i, :] = self.error(trajectory, predicted)
            # print(error, error.shape)
            j = np.argmax(ses[i] > error_threshold)
            if j == 0:
                tau_f_se[i] = N
            else:
                tau_f_se[i] = j
        
            j = np.argmax(mses[i] > error_threshold)
            if j == 0:
                tau_f_mse[i] = N
            else:
                tau_f_mse[i] = j
        return tau_f_se * dt / Lyapunov_time, ses,  tau_f_mse * dt / Lyapunov_time, mses
    


    
class BatchUniform_SMLR:
    """
    Description: runs batches of surrogate models using uniformly random features
    """
    def __init__(self, save_folder, D, D_r, w_max, b_max, sqrt_n_models, beta, error_theshold, dt, Lyapunov_time=1.):
        """
        Args:
            save_folder: folder where all the generated data will be saved
            D: dimension of the underlying dynamical system
            D_r: dimension of the reservoir
            w_max: maximum possible value of an entry of W_in
            b_max: maximum possible value of an entry of b_in
            sqrt_n_models: square root of the number of total surrogate models 
            beta: regularization parameters for ridge regression
            error_threshold: threshold to determine forecast time
            dt: time step of the surrogate model
            Lyapunov_time: Lyapunov time for the underlying dynamical system, default=1
        """
        self.D = D
        self.D_r = D_r
        self.w, self.b = np.linspace(0., w_max, num=sqrt_n_models+1)[1:], np.linspace(0., b_max, num=sqrt_n_models+1)[1:]
        self.sqrt_n_models = sqrt_n_models
        self.beta = beta
        self.error_threshold = error_theshold
        self.dt = dt 
        self.Lyapunov_time = Lyapunov_time

        folders = [save_folder, save_folder + '/W_in', save_folder + '/b_in', save_folder + '/W']

        attributes = ['save_folder', 'W_in_folder', 'b_in_folder', 'W_folder']

        for i, folder in enumerate(folders):
            if not os.path.exists(folder):
                os.makedirs(folder)
            setattr(self, attributes[i], folder)

        self.data = {'i':[], 'j':[], 'k':[], 'w':[], 'b': [], '||W_in||':[], '||b_in||':[], '||W||':[]}    
        
        config = {'D': D, 'D_r': D_r, 'Lyapunov_time': Lyapunov_time, 'dt': dt, 'w_max': w_max, 'b_max':b_max, 'beta': beta,\
                  'error_threshold': error_theshold, 'sqrt_n_models': sqrt_n_models}
        with open('{}/config.json'.format(self.save_folder), 'w') as f:
            f.write(json.dumps(config))
       

    @ut.timer
    def run_single(self, train, i, j, n_repeats=1):
        """
        Description: runs experiments for a single value of (w, b) multiple times 

        Args: 
            train: training data
            i: upper bound for W_in is given by self.w[i]
            j: lower bound for b_in is given by self.b[j]
            n_repeats: number of times to repeat experiment for a single value of (w, b), the upper bounds for W_in, b_in, default=1
        """
        w, b = self.w[i], self.b[j]
        
        for k in range(n_repeats):
            model = SurrogateModel_LR(self.D, self.D_r, lambda d, d_r: uniform_W_in(d, d_r, w), lambda d_r: uniform_b_in(d_r, b))
            model.compute_W(train, beta=self.beta)
            
            # tse, se, tmse, mse = model.compute_forecast_time(test, error_threshold=self.error_threshold, dt=self.dt, Lyapunov_time=self.Lyapunov_time)
            W_in_norm = np.linalg.norm(model.W_in)
            b_in_norm = np.linalg.norm(model.b_in)
            W_norm = np.linalg.norm(model.W)


            np.savetxt(self.W_in_folder + '/W_in_{}_{}_{}.csv'.format(i, j, k), model.W_in, delimiter=',')
            np.savetxt(self.b_in_folder + '/b_in_{}_{}_{}.csv'.format(i, j, k), model.b_in, delimiter=',')
            np.savetxt(self.W_folder + '/W_{}_{}_{}.csv'.format(i, j, k), model.W, delimiter=',')
          

            self.data['i'].append(i)
            self.data['j'].append(j)
            self.data['k'].append(k)
            self.data['w'].append(w)
            self.data['b'].append(b)
            self.data['||W_in||'].append(W_in_norm)
            self.data['||b_in||'].append(b_in_norm)
            self.data['||W||'].append(W_norm)

    
    
    @ut.timer
    def run(self, train, n_repeats=1):
        """
        Description: runs all the experiments, documents W_in, b_in, W, forecast times and errors

        Args: 
            train: training data
            n_repeats: number of times to repeat experiment for a single value of (w, b), the upper bounds for W_in, b_in, default=1
        """
        with open('{}/config.json'.format(self.save_folder), 'r') as f:
            config = json.loads(f.read())
        config['training_points'] = train.shape[1]
        # config['validation_points'] = test.shape[1]
        config['n_repeats'] = n_repeats
        with open('{}/config.json'.format(self.save_folder), 'w') as f:
            f.write(json.dumps(config))

        for i in range(self.sqrt_n_models):
            for j in range(self.sqrt_n_models):
                print('working on choice#{}'.format((i, j)), end='\r')
                self.run_single(train, i, j, n_repeats)
        
        pd.DataFrame.from_dict(self.data).to_csv('{}/batch_data.csv'.format(self.save_folder), index=False)





class BatchSingle_SMLR(BatchUniform_SMLR):
    """
    Description: runs batches of surrogate models using uniformly random features
    """
    def __init__(self, save_folder, D, D_r, w, b, beta, error_theshold, dt, Lyapunov_time=1.):
        """
        Args:
            save_folder: folder where all the generated data will be saved
            D: dimension of the underlying dynamical system
            D_r: dimension of the reservoir
            w: maximum possible value of an entry of W_in
            b: maximum possible value of an entry of b_in
            beta: regularization parameters for ridge regression
            error_threshold: threshold to determine forecast time
            dt: time step of the surrogate model
            Lyapunov_time: Lyapunov time for the underlying dynamical system, default=1
        """
        super().__init__(save_folder, D=D, D_r=D_r, w_max=w, b_max=b, sqrt_n_models=1, beta=beta,\
                               error_theshold=error_theshold, dt=dt, Lyapunov_time=Lyapunov_time)




class BatchRunAnalyzer_SMLR:
    """
    Description: A class for analyzing the results of a batch run
    """
    def __init__(self, save_folder) -> None:
        """
        Args: 
            save_folder: folder where the results of batch run experiments were stored
        """
        self.save_folder = save_folder
        folders = [save_folder, save_folder + '/W_in', save_folder + '/b_in', save_folder + '/W', save_folder + '/tau_f',\
                    save_folder + '/errors', save_folder + '/plots']
        attributes = ['save_folder', 'W_in_folder', 'b_in_folder', 'W_folder', 'tau_f_folder', 'errors_folder', 'plot_folder']
        for i, attribute in enumerate(attributes):
            setattr(self, attribute, folders[i])
        
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        
        self.get_idx()
        self.get_config()
        self.get_wb()

    
    def get_config(self):
        with open('{}/config.json'.format(self.save_folder), 'r') as f:
            config = json.loads(f.read())
        for key in config:
            setattr(self, key, config[key])
        self.n_models = len(self.get_data()['||W||'].to_numpy())

    
    def get_wb(self):
        self.w = np.linspace(0., self.w_max, num=self.sqrt_n_models+1)[1:] 
        self.b = np.linspace(0., self.b_max, num=self.sqrt_n_models+1)[1:]
    
    
    # @ut.timer
    def get_idx(self):
        """
        Description: finds the indices of w, b and randomness
        """
        self.w_idx, self.b_idx, self.random_idx = [],  [], []
        for folder in os.listdir(self.W_folder):
            if folder.count('_') > 2:
                w_idx, b_idx, random_idx = list(map(int, folder.split('.')[0].split('_')[1:]))
                self.w_idx.append(w_idx)
                self.b_idx.append(b_idx)
                self.random_idx.append(random_idx)
                # print(w_idx, b_idx, random_idx)
        self.w_idx = list(set(self.w_idx))
        self.b_idx = list(set(self.b_idx))
        self.random_idx = list(set(self.random_idx))
        self.w_idx.sort()
        self.b_idx.sort()
        self.random_idx.sort()

    def count_zero_rows_norm(self, matrix, threshold):
        for i, row in enumerate(matrix):
            if np.linalg.norm(row) < threshold:
                matrix[i] = np.zeros_like(row)
        return np.sum(~(matrix.any(1)))
    
    def count_zero_rows_entry(self, matrix, limits):
        new_matrix = matrix.copy()
        new_matrix[(new_matrix > limits[0]) & (new_matrix < limits[1])] = 0. 
        return np.sum(~(new_matrix.any(1)))
    
    
    def count_bad_features(self, i, j, k, validation, threshold=0.998):
        model = self.get_model(i, j, k)
        features = np.abs(np.tanh(model.W_in@validation + model.b_in[:,np.newaxis]))
        bad = np.sum(features > threshold, axis=0) #/ self.D_r
        return bad.mean()


    def get_extreme_vector_same(self, xlims, row):
        return np.array([xlims[i, j] for i, j in enumerate((row > 0).astype(int))])
    
    def get_extreme_vector_opposite(self, xlims, row):
        return np.array([xlims[i, j] for i, j in enumerate((row < 0).astype(int))])

    def check_if_lesser(self, xlims, row, M, b):
        x_plus = self.get_extreme_vector_opposite(xlims, row)
        x_minus = self.get_extreme_vector_same(xlims, row)
        return not ((np.dot(row, x_plus) + b < - M) or (np.dot(row, x_minus) + b > M)) 
    
    def check_if_greater(self, xlims, row, m, b):
        x = - b * row / np.dot(row, row)
        if (x > xlims[:, 0]).all() and (x < xlims[:, 1]).all():
            return False
        else:
            x = np.sum(xlims, axis=1) / 2.
            if np.dot(row, x) + b < 0:
                x = self.get_extreme_vector_same(xlims, row)
                return not (-np.dot(row, x) - b < m)
            else:
                x = self.get_extreme_vector_opposite(xlims, row)
                return not (np.dot(row, x) + b < m)
    
    
    
    def count_good_rows(self, W_in, b_in, xlims, threshold=3.5):
        dgr = 0
        for i, row in enumerate(W_in):
            if self.check_if_lesser(xlims, row, threshold, b_in[i]):
                dgr += 1
        return dgr
    


    def get_xlims(self, data):
        xlims = np.zeros((self.D, 2))
        for i, dim in enumerate(data):
            xlims[i, :] = [min(dim), max(dim)]
        return xlims

    
    @ut.timer
    def count(self, validation, limits_in, limits, threshold=3.5):
        zr = np.zeros(self.n_models)
        zr = zr.reshape(self.sqrt_n_models, self.sqrt_n_models, -1)
        zc = np.zeros_like(zr)
        bf = np.zeros_like(zr)
        avg_abs_col_sum = np.zeros_like(zc)
        n_good_rows = np.zeros_like(zc)
        xlims = self.get_xlims(validation)
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        l = 0
        for i in self.w_idx:
            for j in self.b_idx:
                for k in self.random_idx:
                    W = np.genfromtxt(self.W_folder + '/W_{}_{}_{}.csv'.format(i, j, k), delimiter=',')
                    W_in = np.genfromtxt(self.W_in_folder + '/W_in_{}_{}_{}.csv'.format(i, j, k), delimiter=',')
                    b_in = np.genfromtxt(self.b_in_folder + '/b_in_{}_{}_{}.csv'.format(i, j, k), delimiter=',')
                    zr[i][j][k], zc[i][j][k] = self.count_zero_rows_entry(W_in, limits_in), self.count_zero_rows_entry(W.T, limits)
                    bf[i][j][k] = self.count_bad_features(i, j, k, validation, np.tanh(threshold))
                    avg_abs_col_sum[i][j][k] = np.sum(np.abs(W), axis=1).mean()
                    n_good_rows[i][j][k] = self.count_good_rows(W_in, b_in, xlims, threshold)
                    data.loc[l, '0-rows-W_in'] = zr[i][j][k]
                    data.loc[l, '0-cols-W'] = zc[i][j][k]
                    l += 1
            print('working on experiment#{}'.format(i), end='\r')
        data['avg_bad_features'] = bf.flatten()
        data['avg_abs_col_sum_W'] = avg_abs_col_sum.flatten()
        data['good_rows_W_in'] = n_good_rows.flatten()
        data.to_csv('{}/batch_data.csv'.format(self.save_folder), index=None)

                    

    @ut.timer
    def get_mean_std(self, quantity):
        mean, std = np.zeros((len(self.w_idx), len(self.b_idx))), np.zeros((len(self.w_idx), len(self.b_idx))) 
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for i in self.w_idx:
            for j in self.b_idx: 
                array = data.loc[(data['i'] == i) & (data['j'] == j)][quantity]
                mean[i][j], std[i][j] = np.mean(array), np.std(array)
        return mean, std
    
    def get_W_in(self, i, j, k):
        return np.genfromtxt(self.W_in_folder + '/W_in_{}_{}_{}.csv'.format(i, j, k), delimiter=',')
    
    def get_b_in(self, i, j, k):
        return np.genfromtxt(self.b_in_folder + '/b_in_{}_{}_{}.csv'.format(i, j, k), delimiter=',')
    
    def get_W(self, i, j, k):
        return np.genfromtxt(self.W_folder + '/W_{}_{}_{}.csv'.format(i, j, k), delimiter=',')
    
    
    def get_model(self, i, j, k):
        W_in_fn = lambda d, d_r: np.genfromtxt(self.W_in_folder + '/W_in_{}_{}_{}.csv'.format(i, j, k), delimiter=',')
        b_in_fn  = lambda d_r: np.genfromtxt(self.b_in_folder + '/b_in_{}_{}_{}.csv'.format(i, j, k), delimiter=',')
        W = np.genfromtxt(self.W_folder + '/W_{}_{}_{}.csv'.format(i, j, k), delimiter=',')
        model = SurrogateModel_LR(self.D, self.D_r, W_in_fn, b_in_fn, W)
        return model
    
    def get_data(self):
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        return data
    
    @ut.timer
    def compute_error(self, validation, rmse_threshold):
        """
        Description: computes forecast time tau_f for the computed surrogate model

        Args:
            validation: a validation trajectory
            rmse_threshold: threshold to determine forecast time
        """
        self.validation_points = validation.shape[-1]
        tau_f_rmse = np.zeros((self.sqrt_n_models, self.sqrt_n_models, self.n_repeats))
        tau_f_mse = np.zeros_like(tau_f_rmse)
        tau_f_se =np.zeros_like(tau_f_rmse)
        rmse = np.zeros_like(tau_f_rmse)
        se = np.zeros_like(tau_f_rmse)
        

        for i in self.w_idx:
            for j in self.b_idx:
                for k in self.random_idx:
                    model = self.get_model(i, j, k)
                    prediction = model.multistep_forecast(validation[:, 0], self.validation_points)
                    se_ = np.linalg.norm(validation - prediction, axis=0)**2 / np.linalg.norm(validation, axis=0)**2
                    mse_ = np.cumsum(se_) / np.arange(1, len(se_)+1)
                    rmse_ = np.sqrt(mse_)
                    
                    l = np.argmax(rmse_ > rmse_threshold)
                    if l == 0:
                        tau_f_rmse[i][j][k] = self.validation_points
                    else:
                        tau_f_rmse[i][j][k] = l-1

                    l = np.argmax(mse_ > rmse_threshold**2)
                    if l == 0:
                        tau_f_mse[i][j][k] = self.validation_points
                    else:
                        tau_f_mse[i][j][k] = l-1

                    l = np.argmax(se_ > rmse_threshold**2)
                    if l == 0:
                        tau_f_se[i][j][k] = self.validation_points
                    else:
                        tau_f_se[i][j][k] = l-1
                    
                    rmse[i][j][k] = rmse_[-1]
                    se[i][j][k] = se_.mean()
            print('working on experiment#{}'.format(i), end='\r')
            
        
        tau_f_rmse *= (self.dt / self.Lyapunov_time)
        tau_f_mse  *= (self.dt / self.Lyapunov_time)
        tau_f_se *= (self.dt / self.Lyapunov_time)
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        data['tau_f_rmse'] = tau_f_rmse.flatten()
        data['tau_f_mse'] = tau_f_mse.flatten()
        data['tau_f_se'] = tau_f_se.flatten()
        data['rmse'] = rmse.flatten()
        data['mse'] = rmse.flatten()**2
        data['se'] = se.flatten()
        data.to_csv('{}/batch_data.csv'.format(self.save_folder), index=None)
        

        with open('{}/config.json'.format(self.save_folder), 'r') as f:
            config = json.loads(f.read())
        config['validation_points'] = self.validation_points
        with open('{}/config.json'.format(self.save_folder), 'w') as f:
            f.write(json.dumps(config))

    
    @ut.timer
    def plot(self, reduction_factor):
        xlabel_size = 13
        ylabel_size = 13
        title_size = 13
        tick_size = 5

        # plot the tau_f heat map
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(121)
        ax_sd = fig.add_subplot(122)
        w, b = self.w, self.b
        tau_f_se, tau_f_se_std = self.get_mean_std('tau_f_se')
        im = ax.contourf(w, b, tau_f_se.T, cmap='viridis')
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=tick_size+5)
        ax.set_xlabel('w', fontsize=xlabel_size+5)
        ax.set_ylabel('b', fontsize=ylabel_size+5)
        ax.set_title(r'$\tau_f$ (in Lyapunov units)', fontsize=ylabel_size+5)


        im = ax_sd.contourf(w, b, tau_f_se_std.T, cmap='viridis')
        cbar = fig.colorbar(im, ax=ax_sd)
        cbar.ax.tick_params(labelsize=tick_size+5)
        ax_sd.set_xlabel('w', fontsize=xlabel_size+5)
        ax_sd.set_ylabel('b', fontsize=ylabel_size+5)
        ax_sd.set_title(r'standard deviation in $\tau_f$', fontsize=ylabel_size+5)
        plt.savefig('{}/tau_f_se.png'.format(self.plot_folder))
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(121)
        ax_sd = fig.add_subplot(122)
        w, b = self.w, self.b
        tau_f_mse, tau_f_mse_std = self.get_mean_std('tau_f_mse')
        im = ax.contourf(w, b, tau_f_mse.T, cmap='viridis')
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=tick_size+5)
        ax.set_xlabel('w', fontsize=xlabel_size+5)
        ax.set_ylabel('b', fontsize=ylabel_size+5)
        ax.set_title(r'$\tau_f$ (in Lyapunov units)', fontsize=ylabel_size+5)


        im = ax_sd.contourf(w, b, tau_f_mse_std.T, cmap='viridis')
        cbar = fig.colorbar(im, ax=ax_sd)
        cbar.ax.tick_params(labelsize=tick_size)
        ax_sd.set_xlabel('w', fontsize=xlabel_size)
        ax_sd.set_ylabel('b', fontsize=ylabel_size)
        ax_sd.set_title(r'standard deviation in $\tau_f$', fontsize=ylabel_size)
        plt.savefig('{}/tau_f_mse.png'.format(self.plot_folder))
        plt.close(fig)

        # plot grid quantities
        data = self.get_data()
        total = self.n_repeats * self.sqrt_n_models**2
        random_idx = np.random.choice(total, size=int(total/reduction_factor), replace=False)
        data_list = []
        data_list.append(data['tau_f_se'][random_idx])
        data_list.append(data['0-cols-W'][random_idx] / self.D_r)
        data_list.append(data['avg_abs_col_sum_W'][random_idx] / self.D_r)
        data_list.append(data['||W||'][random_idx] / self.D_r)
        data_list.append(data['avg_bad_features'][random_idx] / self.D_r)
        data_list.append(data['good_rows_W_in'][random_idx] / self.D_r)
        # data_list.append(data['0-rows-W_in'][random_idx] / self.D_r)

        label_list = []
        label_list.append(r'$\tau_f$')
        label_list.append(r'zero cols $W$')
        label_list.append(r'$E|\text{col|sum } W$')
        label_list.append(r'$\|W\|$')
        label_list.append(r'bad features')
        label_list.append(r'good rows $W_{\rm in}$')
        # label_list.append(r'zero rows $W_{\rm in}$')


        # pairs = [[1, 0], [2, 0], [3, 0],\
        #          [4, 0], [5, 0], [6, 0],\
        #          [2, 4], [3, 4], [5, 4],\
        #          [4, 1], [5, 1], [6, 1],\
        #          [4, 6], [2, 5], [6, 5],\
        #          [2, 1], [3, 1], [3, 5],\
        #          [3, 2], [3, 6], [2, 6]]
        
        pairs = []
        for i in range(len(data_list)):
            for j in range(len(data_list)):
                if i < j:
                    pairs.append([j, i])
        color_list = [data_list[0]] * len(pairs)
        
        n_rows, n_cols = 6, 4
        fig_name = '{}/relationships.png'.format(self.plot_folder)
        s = 1
        figsize = 4
        wspace = 0.3
        hspace = 0.3 

        ut.grid_scatter_plotter(data_list, label_list, color_list, pairs, n_rows, n_cols, fig_name,\
						 xlabel_size, ylabel_size, title_size, tick_size,\
						 wspace, hspace, s, figsize)


        plt.hist(data_list[0], weights=np.ones(len(data_list[0])) / len(data_list[0]))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel(r'$\tau_f$')
        plt.savefig('{}/distribution_of_realizations.png'.format(self.plot_folder))



         


class BatchStrategy_SMLR:
    """
    Description: runs batches of surrogate models using uniformly random features. Initialization involves picking 
    either good, linear or extreme rows randomly.
    """
    def __init__(self, save_folder, D, D_r, n_models, beta, error_theshold, train, dt, Lyapunov_time=1.,\
                 L0=0.4, L1=3.5, percents=3, row_selection='good_linear_extreme', train_option='constant'):
        """
        Args:
            save_folder: folder where all the generated data will be saved
            D: dimension of the underlying dynamical system
            D_r: dimension of the reservoir
            n_models: the number of total surrogate models 
            beta: regularization parameters for ridge regression
            train: training data
            error_threshold: threshold to determine forecast time
            dt: time step of the surrogate model
            Lyapunov_time: Lyapunov time for the underlying dynamical system, default=1
            L0: lower bound for determining good rows
            L1: upper bound for determining good rows
            percents: an integer determining the allowed percentages (uniform, starting at 0. and ending at 100.)
                      for any type of row
            row_selection: row selection strategy
            train_option: training data selection strategy
        """
        self.D = D
        self.D_r = D_r

        self.n_models = n_models
        self.beta = beta
        self.train = train
        self.error_threshold = error_theshold
        self.dt = dt 
        self.Lyapunov_time = Lyapunov_time
        dx = 100./percents
        self.percents = np.arange(0., 100. + dx, dx)
        self.remaining = ((100. + dx - self.percents) / dx).astype(int)
        self.L0 = L0
        self.L1 = L1
        self.sampler = sm.MatrixSampler(L0, L1, train.T)
        self.row_selection = row_selection
        self.train_option = train_option
        if self.train_option == 'constant':
            self.training_points = self.train.shape[1]
        elif self.train_option.startswith('random'):
            self.training_points = int(self.train_option.split('_')[-1])

        folders = [save_folder, save_folder + '/W_in', save_folder + '/b_in', save_folder + '/W']

        attributes = ['save_folder', 'W_in_folder', 'b_in_folder', 'W_folder']

        for i, folder in enumerate(folders):
            if not os.path.exists(folder):
                os.makedirs(folder)
            setattr(self, attributes[i], folder)

        self.data = {'l':[], '||W_in||':[], '||b_in||':[], '||W||':[], 'n_good_rows_W_in':[], 'n_linear_rows_W_in':[],\
                     'n_extreme_rows_W_in': []}    
        
        config = {'D': D, 'D_r': D_r, 'Lyapunov_time': Lyapunov_time, 'dt': dt, 'beta': beta,\
                  'error_threshold': error_theshold, 'n_models': n_models, 'percents': list(self.percents),\
                  'L0':L0, 'L1':L1, 'training_points': self.training_points, 'row_selection': row_selection,\
                  'train_option': train_option}

        with open('{}/config.json'.format(self.save_folder), 'w') as f:
            f.write(json.dumps(config))
    


    def get_row_partitions(self, n_rows):
        i = np.random.randint(len(self.percents))
        good = int(self.percents[i]* n_rows / 100.)
        j = np.random.randint(self.remaining[i])
        linear = int(self.percents[j]* n_rows / 100.)
        extreme = n_rows - good - linear
        return [good, linear, extreme]



    @ut.timer
    def run_single(self, l):
        """
        Description: runs experiments for a single value of (w, b) multiple times 

        Args: 
            l: index/identifier for the single experiment
        """
        if self.row_selection == 'good_linear_extreme':
            partition = self.get_row_partitions(self.D_r)
        elif self.row_selection == 'good_50_50':
            i = np.random.randint(len(self.percents))
            good = int(self.percents[i]* self.D_r / 100.)
            linear = int((100. - self.percents[i]) * self.D_r / 200.)
            extreme = self.D_r - good - linear
            partition = [good, linear, extreme]
        if self.train_option == 'constant':
            train = self.train
        elif self.train_option.startswith('random'):
            i = np.random.randint(self.train.shape[1] - self.training_points)
            train = self.train[:, i:i+self.training_points]

        W_in, b_in = self.sampler.sample_(partition);
        model = SurrogateModel_LR(self.D, self.D_r, W_in, b_in)
        model.compute_W(train, beta=self.beta);
        
        
        W_in_norm = np.linalg.norm(model.W_in)
        b_in_norm = np.linalg.norm(model.b_in)
        W_norm = np.linalg.norm(model.W)


        np.savetxt(self.W_in_folder + '/W_in_{}.csv'.format(l), model.W_in, delimiter=',')
        np.savetxt(self.b_in_folder + '/b_in_{}.csv'.format(l), model.b_in, delimiter=',')
        np.savetxt(self.W_folder + '/W_{}.csv'.format(l), model.W, delimiter=',')
        

        self.data['l'].append(l)
        self.data['||W_in||'].append(W_in_norm)
        self.data['||b_in||'].append(b_in_norm)
        self.data['||W||'].append(W_norm)
        self.data['n_good_rows_W_in'].append(partition[0])
        self.data['n_linear_rows_W_in'].append(partition[1])
        self.data['n_extreme_rows_W_in'].append(partition[2])

    
    
    @ut.timer
    def run(self):
        """
        Description: runs all the experiments, documents W_in, b_in, W, forecast times and errors
        """

        for l in range(self.n_models):
            print('working on experiment#{}'.format(l), end='\r')
            self.run_single(l)
        
        pd.DataFrame.from_dict(self.data).to_csv('{}/batch_data.csv'.format(self.save_folder), index=False)



class BatchStrategyAnalyzer_SMLR:
    """
    Description: A class for analyzing the results of a batch run
    """
    def __init__(self, save_folder) -> None:
        """
        Args: 
            save_folder: folder where the results of batch run experiments were stored
        """
        self.save_folder = save_folder
        folders = [save_folder, save_folder + '/W_in', save_folder + '/b_in', save_folder + '/W', save_folder + '/tau_f',\
                    save_folder + '/errors', save_folder + '/plots']
        attributes = ['save_folder', 'W_in_folder', 'b_in_folder', 'W_folder', 'tau_f_folder', 'errors_folder', 'plot_folder']
        for i, attribute in enumerate(attributes):
            setattr(self, attribute, folders[i])
        
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        
        self.get_config()
        
    
    def get_config(self):
        with open('{}/config.json'.format(self.save_folder), 'r') as f:
            config = json.loads(f.read())
        for key in config:
            setattr(self, key, config[key])
        self.percents = np.array(self.percents)

    def count_zero_rows_entry(self, matrix, limits):
        new_matrix = matrix.copy()
        new_matrix[(new_matrix > limits[0]) & (new_matrix < limits[1])] = 0. 
        return np.sum(~(new_matrix.any(1)))
    
    
    def count_bad_features(self, l, validation, threshold=0.998):
        model = self.get_model(l)
        features = np.abs(np.tanh(model.W_in@validation + model.b_in[:,np.newaxis]))
        bad = np.sum(features > threshold, axis=0) #/ self.D_r
        return bad.mean()

    
    @ut.timer
    def count(self, validation, limits_in, limits, threshold=3.5):
        zr, zc = np.zeros(self.n_models), np.zeros(self.n_models)
        bf = np.zeros_like(zc)
        avg_abs_col_sum = np.zeros_like(zc)
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))

        for l in range(self.n_models):
            W = np.genfromtxt(self.W_folder + '/W_{}.csv'.format(l), delimiter=',')
            W_in = np.genfromtxt(self.W_in_folder + '/W_in_{}.csv'.format(l), delimiter=',')
            zr[l], zc[l] = self.count_zero_rows_entry(W_in, limits_in), self.count_zero_rows_entry(W.T, limits)
            bf[l] = self.count_bad_features(l, validation, np.tanh(threshold))
            avg_abs_col_sum[l] = np.sum(np.abs(W), axis=1).mean()
            data.loc[l, '0-rows-W_in'] = zr[l]
            data.loc[l, '0-cols-W'] = zc[l]

            print('working on experiment#{}'.format(l), end='\r')
        data['avg_bad_features'] = bf.flatten()
        data['avg_abs_col_sum_W'] = avg_abs_col_sum.flatten()
        data.to_csv('{}/batch_data.csv'.format(self.save_folder), index=None)

                    

    @ut.timer
    def get_mean_std(self, quantity):
        mean, std = np.zeros(self.n_models), np.zeros(self.n_model) 
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for l in range(self.n_model):
            array = data.loc[(data['l'] == l)][quantity]
            mean[l], std[l] = np.mean(array), np.std(array)
        return mean, std
    
    def get_W_in(self, l):
        return np.genfromtxt(self.W_in_folder + '/W_in_{}.csv'.format(l), delimiter=',')
    
    def get_b_in(self, l):
        return np.genfromtxt(self.b_in_folder + '/b_in_{}.csv'.format(l), delimiter=',')
    
    def get_W(self, l):
        return np.genfromtxt(self.W_folder + '/W_{}.csv'.format(l), delimiter=',')
    
    
    def get_model(self, l):
        W_in_fn = lambda d, d_r: np.genfromtxt(self.W_in_folder + '/W_in_{}.csv'.format(l), delimiter=',')
        b_in_fn  = lambda d_r: np.genfromtxt(self.b_in_folder + '/b_in_{}.csv'.format(l), delimiter=',')
        W = np.genfromtxt(self.W_folder + '/W_{}.csv'.format(l), delimiter=',')
        model = SurrogateModel_LR(self.D, self.D_r, W_in_fn, b_in_fn, W)
        return model
    

    @ut.timer
    def get_grid(self, quantity):
        m = len(self.percents)
        mean, std = np.zeros((m, m)), np.zeros((m, m))
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for i in range(m):
            good = int(self.D_r*self.percents[i]/100.)
            for j in range(m): 
                extreme = int(self.D_r*self.percents[j]/100.)
                array = data.loc[(data['n_good_rows_W_in'] == good) & (data['n_extreme_rows_W_in'] == extreme)][quantity]
                mean[i][j], std[i][j] = np.mean(array), np.std(array)
        return mean, std 
    
    @ut.timer
    def get_grid_bad(self, quantity):
        m = len(self.percents)
        mean, std = np.zeros((m, m)), np.zeros((m, m))
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for i in range(m):
            linear = int(self.D_r*self.percents[i]/100.)
            for j in range(m): 
                extreme = int(self.D_r*self.percents[j]/100.)
                array = data.loc[(data['n_linear_rows_W_in'] == linear) & (data['n_extreme_rows_W_in'] == extreme)][quantity]
                mean[i][j], std[i][j] = np.mean(array), np.std(array)
        return mean, std 


    def get_grid_abc(self, a, b, c, arr_a, arr_b):
        m, n = len(arr_a), len(arr_b)
        num, mean, std = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for i in range(m):
            
            for j in range(n): 
                
                array = data.loc[(data[a] <= arr_a[i]) & (data[b] <= arr_b[j])][c]
                mean[i][j], std[i][j] = np.mean(array), np.std(array)
                num[i][j] = len(array)
        return mean, std, num 


    def get_data(self):
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        return data
    
    @ut.timer
    def compute_error(self, validation, rmse_threshold):
        """
        Description: computes forecast time tau_f for the computed surrogate model

        Args:
            validation: multiple validation trajectories out of which one is randomly picked for calculating tau_f
            rmse_threshold: threshold to determine forecast time
        """
        n_trajectories = len(validation)
        self.validation_points = validation.shape[-1]
        tau_f_rmse = np.zeros(self.n_models)
        tau_f_mse = np.zeros_like(tau_f_rmse)
        tau_f_se =np.zeros_like(tau_f_rmse)
        rmse = np.zeros_like(tau_f_rmse)
        se = np.zeros_like(tau_f_rmse)
        

        for m in range(self.n_models):
            validation_ = validation[np.random.randint(n_trajectories)]
            model = self.get_model(m)
            prediction = model.multistep_forecast(validation_[:, 0], self.validation_points)
            se_ = np.linalg.norm(validation_ - prediction, axis=0)**2 / np.linalg.norm(validation_, axis=0)**2
            mse_ = np.cumsum(se_) / np.arange(1, len(se_)+1)
            rmse_ = np.sqrt(mse_)
            
            l = np.argmax(rmse_ > rmse_threshold)
            if l == 0:
                tau_f_rmse[m] = self.validation_points
            else:
                tau_f_rmse[m] = l-1

            l = np.argmax(mse_ > rmse_threshold**2)
            if l == 0:
                tau_f_mse[m] = self.validation_points
            else:
                tau_f_mse[m] = l-1

            l = np.argmax(se_ > rmse_threshold**2)
            if l == 0:
                tau_f_se[m] = self.validation_points
            else:
                tau_f_se[m] = l-1
            
            rmse[m] = rmse_[-1]
            se[m] = se_.mean()
            print('working on experiment#{}'.format(m), end='\r')
            
        
        tau_f_rmse *= (self.dt / self.Lyapunov_time)
        tau_f_mse  *= (self.dt / self.Lyapunov_time)
        tau_f_se *= (self.dt / self.Lyapunov_time)
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        data['tau_f_rmse'] = tau_f_rmse
        data['tau_f_mse'] = tau_f_mse
        data['tau_f_se'] = tau_f_se
        data['rmse'] = rmse
        data['mse'] = rmse**2
        data['se'] = se
        data.to_csv('{}/batch_data.csv'.format(self.save_folder), index=None)
        

        with open('{}/config.json'.format(self.save_folder), 'r') as f:
            config = json.loads(f.read())
        config['validation_points'] = self.validation_points
        config['validation_option'] = n_trajectories
        
        with open('{}/config.json'.format(self.save_folder), 'w') as f:
            f.write(json.dumps(config))

    
    @ut.timer
    def plot(self, reduction_factor=10, s=20):
        xlabel_size = 13
        ylabel_size = 13
        title_size = 13
        tick_size = 5

        # plot grid quantities
        data = self.get_data()
        total = self.n_models
        random_idx = np.random.choice(total, size=int(total/reduction_factor), replace=False)
        data_list = []
        data_list.append(data['tau_f_se'][random_idx])
        data_list.append(data['0-cols-W'][random_idx] / self.D_r)
        # data_list.append(data['avg_abs_col_sum_W'][random_idx] / self.D_r)
        data_list.append(data['||W||'][random_idx] / self.D_r)
        data_list.append(data['avg_bad_features'][random_idx] / self.D_r)
        data_list.append(data['n_good_rows_W_in'][random_idx] / self.D_r)
        data_list.append(data['n_linear_rows_W_in'][random_idx] / self.D_r)
        data_list.append(data['n_extreme_rows_W_in'][random_idx] / self.D_r)
   
        label_list = []
        label_list.append(r'$\tau_f$')
        label_list.append(r'zero cols $W$')
        # label_list.append(r'$E|\text{col|sum } W$')
        label_list.append(r'$\|W\|$')
        label_list.append(r'bad features')
        label_list.append(r'good rows $W_{\rm in}$')
        label_list.append(r'linear rows $W_{\rm in}$')
        label_list.append(r'extreme rows $W_{\rm in}$')

        pairs = []
        for i in range(len(data_list)):
            for j in range(len(data_list)):
                if i < j:
                    pairs.append([j, i])
        
        n_rows, n_cols = 6, 4
        fig_name = '{}/relationships.png'.format(self.plot_folder)
        # s = 20
        figsize = 4
        wspace = 0.3
        hspace = 0.3 
        color_list = [data_list[0]] * len(pairs)
        size_list = [data_list[4]] * len(pairs)
        ut.grid_scatter_plotter(data_list, label_list, color_list, pairs, n_rows, n_cols, fig_name,\
						 xlabel_size, ylabel_size, title_size, tick_size,\
						 wspace, hspace, s, figsize, size_list)

       
       
       
        grid_data_list = []
        grid_data_list.append(self.get_grid('tau_f_se')[0])
        grid_data_list.append(self.get_grid('0-cols-W')[0] / self.D_r)
        grid_data_list.append(self.get_grid('||W||')[0] / self.D_r)
        grid_data_list.append(self.get_grid('avg_bad_features')[0] / self.D_r)
 
        x = np.array(self.percents) / 100.
        xlabel = 'good rows'
        ylabel = 'extreme rows'

        n_rows, n_cols = 2, 2
        fig_name = '{}/good-bad.png'.format(self.plot_folder)

       
        ut.grid_heat_plotter(x, x, grid_data_list, xlabel, ylabel, label_list, n_rows, n_cols, fig_name,\
						xlabel_size, ylabel_size, title_size, tick_size,\
						wspace, hspace, figsize)
        
        bad_grid_data_list = []
        bad_grid_data_list.append(self.get_grid('tau_f_se')[0])
        bad_grid_data_list.append(self.get_grid('0-cols-W')[0] / self.D_r)
        bad_grid_data_list.append(self.get_grid('||W||')[0] / self.D_r)
        bad_grid_data_list.append(self.get_grid('avg_bad_features')[0] / self.D_r)

        xlabel = 'linear rows'
        fig_name = '{}/bad-bad.png'.format(self.plot_folder)
        ut.grid_heat_plotter(x, x, bad_grid_data_list, xlabel, ylabel, label_list, n_rows, n_cols, fig_name,\
						xlabel_size, ylabel_size, title_size, tick_size,\
						wspace, hspace, figsize)

        plt.hist(data_list[0], weights=np.ones(len(data_list[0])) / len(data_list[0]))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel(r'$\tau_f$')
        plt.savefig('{}/distribution_of_realizations.png'.format(self.plot_folder))

        self.plot_extras()


    def get_generalize_overfit_superb(self, quantity, gen=4., over=1.5, super=5.):
        data = self.get_data()
        generalize = np.zeros(len(self.percents))
        overfit = np.zeros(len(self.percents))
        superb = np.zeros(len(self.percents)) 

        generalize_std = np.zeros(len(self.percents))
        overfit_std = np.zeros(len(self.percents))
        superb_std = np.zeros(len(self.percents))

        num_gen = np.zeros(len(self.percents))
        num_over = np.zeros(len(self.percents))
        num_super = np.zeros(len(self.percents))

        for i, p in enumerate(self.percents):
            subset = data.loc[(data['n_good_rows_W_in'] == int(p*self.D_r/100.))]
            arr_gen = subset.loc[subset['tau_f_se'] >= gen][quantity]
            arr_over = subset.loc[subset['tau_f_se'] <= over][quantity]
            arr_super = subset.loc[subset['tau_f_se'] >= super][quantity]

            generalize[i] = arr_gen.mean()
            overfit[i] = arr_over.mean()
            superb[i] = arr_super.mean()

            generalize_std[i] = arr_gen.std()
            overfit_std[i] = arr_over.std()
            superb_std[i] = arr_super.std()

            num_gen[i] = len(arr_gen)
            num_over[i] = len(arr_over)
            num_super[i] = len(arr_super)


        return [[generalize, overfit, superb], [generalize_std, overfit_std, superb_std], [num_gen, num_over, num_super]]


    
    
    
    
    def plot_extras(self):
        n_rows, n_cols = 5, 2
        fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
        axes = [fig.add_subplot(n_rows, n_cols, j+1) for j in range(9)]
        quantities = ['tau_f_se', '0-cols-W', '||W||', 'avg_bad_features']
        labels = [r'$\tau_f$', 'zero cols W', '||W||', 'bad features']
  
        for i in range(0, 8, 2):
            j = int(i/2)
            mean, std, num = self.get_generalize_overfit_superb(quantities[j])
            if i > 0:
                mean = [e/self.D_r for e in mean]
                std = [e/self.D_r for e in std]

            axes[i].plot(self.percents/100., mean[0], label='generalize') 
            axes[i].plot(self.percents/100., mean[1], label='overfit') 
            axes[i].plot(self.percents/100., mean[2], label='superb')

            axes[i].set_xlabel('good rows')
            axes[i].set_ylabel(labels[j])
            axes[i].legend()

            axes[i+1].plot(self.percents/100., std[0], label='generalize') 
            axes[i+1].plot(self.percents/100., std[1], label='overfit') 
            axes[i+1].plot(self.percents/100., std[2], label='superb') 

            axes[i+1].set_xlabel('good rows')
            axes[i+1].set_ylabel('std ' + labels[j])
            axes[i+1].legend()

        axes[-1].plot(self.percents/100., num[0], label='generalize') 
        axes[-1].plot(self.percents/100., num[1], label='overfit') 
        axes[-1].plot(self.percents/100., num[2], label='superb') 

        axes[-1].set_xlabel('good rows')
        axes[-1].set_ylabel('number of occurences')
        axes[-1].legend()

        fig.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.savefig('{}/generalize-overfit-superb.png'.format(self.plot_folder))