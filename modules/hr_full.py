import os, sys 
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import numpy as np
import matplotlib.pyplot as plt
import utility as ut
from joblib import Parallel, delayed

class InequalitySampler3:
    """
    A class for sampling from w.x_+ + b < l+ and w.x_- + b > l-
    """
    def __init__(self, x_pm, l_minus, l_plus, dim) -> None:
        self.x_pm = x_pm
        self.l_minus = l_minus
        self.l_plus = l_plus
        self.dim = dim

    
    def is_feasible(self, x):
        """
        Description: Determines if an input belongs to the feasible region
        """
        w, b = x[:-1], x[-1]
        x_plus = self.x_pm(np.sign(w), True)
        x_minus = self.x_pm(np.sign(w), False)
        y_plus = np.dot(x_plus, w)
        y_minus = np.dot(x_minus, w)
        return (y_plus + b < self.l_plus) & (y_minus + b > self.l_minus)

    
    
    def intersection_with_bisection(self, x0, d, tol=1e-2, max_iters=100):
        """
        Description: Finds the intersection of a line originating from x0 (lying inside the feasible region) having direction d with the feasible region
                    within the specified tolerance within the max_iters iterations
        """
        
        self.x0 = x0
        self.weights = [0., 0.]
        
        if self.is_feasible(self.x0):
            t = 1.
            while self.is_feasible(self.x0 + t*d):
                t *= 2 

            l, r = 0., t
            iter = 0

            while abs(r-l) > tol and iter < max_iters:
                m = (r+l)/2.
                if not self.is_feasible(self.x0 + m*d):
                    r = m
                else:
                    l = m
                iter += 1
            
            # self.int_pts[0, :] = self.x0 + l*d
            self.weights[1] = l

        if self.is_feasible(self.x0):
            t = -1.
            while self.is_feasible(self.x0 + t*d):
                t *= 2 

            l, r = 0., t
            iter = 0

            while abs(r-l) > tol and iter < max_iters:
                m = (r+l)/2.
                if not self.is_feasible(self.x0 + m*d):
                    r = m
                else:
                    l = m
                iter += 1
            
            # self.int_pts[0, :] = self.x0 + l*d
            self.weights[0] = l
            return 1
        else:
           self.int_pts, self.weights = [], []
           return 0
    
    
    def single_sample(self, x, steps, **kwargs):
        for _ in range(steps):
            d = np.random.normal(size=self.dim+1)
            # d = np.abs(d) * self.signs
            d /= np.linalg.norm(d)
            # find the intersection points 
            while self.intersection_with_bisection(x, d, **kwargs) == 0:
                d = np.random.normal(size=self.dim+1)
                # d = np.abs(d) * self.signs
                d /= np.linalg.norm(d)
            t = np.random.uniform(*self.weights)
            x += t*d
        return x













class GoodRowSampler:
    """
    A class for sampling a row R such that R satisfies m<|R.x + b|<M for x in C = a convex set
    """
    def __init__(self, m, M, data):
        self.m = m 
        self.M = M 
        self.data = np.array(data)
        self.dim = self.data.shape[-1]
        self.lims = np.array([[min(self.data[:, d]), max(self.data[:, d])] for d in range(self.dim)])
        


    def get_vector(self, signs, option):
        for i in range(len(signs)):
            if signs[i] == -1:
                signs[i] = 0 
        signs = list(map(int, signs))  
        if option:
            return np.array([self.lims[d, signs[d]] for d in range(self.dim)])
        else:
            return np.array([self.lims[d, (1+signs[d]) % 2] for d in range(self.dim)])
        

    def sample_(self, steps=10):
        flag = np.random.randint(2)
        # assign signs for the entries of the row
        s = np.random.randint(2, size=self.dim)
        # set up inequalities
        if flag:
            lims = [self.m, self.M]
        else:
            lims = [-self.M, -self.m]

        b = np.random.uniform(*lims)
        sampler = InequalitySampler3(self.get_vector, lims[0], lims[1], self.dim)
        return sampler.single_sample(x=np.hstack((np.zeros(self.dim), b)), steps=steps)
    
    # @ut.timer
    def sample(self, n_sample, steps=10):
        rows, bs = np.zeros((n_sample, self.dim)), np.zeros(n_sample)
        for n in range(n_sample):
            wb = self.sample_(steps)
            rows[n, :], bs[n] = wb[:-1], wb[-1]
        return rows, bs
    
    # @ut.timer
    def sample_parallel(self, n_sample):
        results = Parallel(n_jobs=-1)(delayed(self.sample_)() for _ in range(n_sample))
        rows = np.vstack([item[0] for item in results])
        bs = np.hstack([item[1] for item in results])
        return rows, bs


    def test_rows(self, rows, bs):
        Y = np.abs(self.data@(np.array(rows).T) + np.array(bs))
        return np.all((Y < self.M) & (Y > self.m), axis=0)
    
    
    def is_row(self, row, b):
        if b < self.M and b > self.m:
            sign = ((np.sign(row) + 1) / 2).astype(int)
            x_plus = self.get_vector(sign, True)
            x_minus = self.get_vector(sign, False)
            if x_plus @ row + b < self.M and x_minus @ row + b > self.m:
                return True
            else:
                return False
        elif b < -self.m and b > -self.M:
            sign = ((np.sign(row) + 1) / 2).astype(int)
            x_plus = self.get_vector(sign, True)
            x_minus = self.get_vector(sign, False)
            if x_plus @ row + b < -self.m and x_minus @ row + b > -self.M:
                return True
            else:
                return False
        else:
            return False
        
    def are_rows(self, rows, bs):
        flags = np.full(len(rows), True)
        for i in range(len(rows)):
            flags[i] = self.is_row(rows[i], bs[i])
        return flags