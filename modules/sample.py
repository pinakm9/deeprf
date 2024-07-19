import numpy as np
import utility as ut
from joblib import Parallel, delayed

class GoodRowSampler:
    """
    Restricted hit and run sampler for good rows
    """
    def __init__(self, L0, L1, Uo):
        """
        Args:
            L0: left limit of tanh input for defining good rows
            L1: right limit tanh input for defining good rows
            Uo: training data
        """
        self.L0 = L0
        self.L1 = L1 
        self.Uo = Uo
        self.dim = self.Uo.T.shape[-1]
        self.lims = np.array([[min(self.Uo.T[:, d]), max(self.Uo.T[:, d])] for d in range(self.dim)])

    
    def update(self, Uo):
        self.Uo = Uo


    def x_plus(self, s):
        """
        Args:
            s: sign vector
        """
        return np.array([self.lims[d, s[d]] for d in range(self.dim)])
    
    
    def x_minus(self, s):
        """
        Args:
            s: sign vector
        """
        return np.array([self.lims[d, (1+s[d]) % 2] for d in range(self.dim)])
    

    def sample_(self, sample_b=True):
     
        # choose bias
        b = np.random.uniform(self.L0, self.L1)
        
        # choose an orthant
        s = np.random.randint(2, size=self.dim)
        # choose a direction in the orthant
        d = np.abs(np.random.normal(size=self.dim)) * np.array([1 if e else -1 for e in s])
        d /= np.linalg.norm(d)
        
        # determine line segment
        d_plus = d @ self.x_plus(s)
        d_minus = d @ self.x_minus(s)
        q = (self.L0 - b) / d_minus
        p = (self.L1 - b) / d_plus


        if d_plus > 0:
            if d_minus < 0:
                a1 = np.min([p, q]) 
            else:
                a1 = p 
        else:
            if d_minus < 0:
                a1 = q
            else:
                a1 = 1e6             

        # pick a point on the line 
        a = np.random.uniform(0, a1)
        if sample_b:
            wb = np.hstack([a*d, b])
            # decide which subset of the solution set we want to sample
            if np.random.randint(2) == 1:
                return wb
            else:
                return -wb
        else:
            if np.random.randint(2) == 1:
                return a*d
            else:
                return -a*d
    
    @ut.timer
    def sample_parallel(self, n_rows, sample_b=True):
        """
        Args:
            n_rows: number of rows to sample
        """
        results = Parallel(n_jobs=-1)(delayed(self.sample_)(sample_b) for _ in range(n_rows))
        return np.array(results)
    
    @ut.timer
    def sample(self, n_rows, sample_b=True):
        """
        Args:
            n_rows: number of rows to sample
        """
        results = [self.sample_(sample_b) for _ in range(n_rows)]
        return np.array(results)
    
    def test_rows(self, rows):
        return np.all([self.is_row(row) for row in rows])
    
    def is_row(self, row):
        if row[-1] < 0:
            row *= -1
        # find orthant
        s = ((np.sign(row) + 1) / 2).astype(int)
        x_minus = np.hstack([self.x_minus(s), 1])
        x_plus = np.hstack([self.x_plus(s), 1])
        # print((x_minus @ row > self.L0), (x_plus @ row < self.L1))
        return (x_minus @ row > self.L0) and (x_plus @ row < self.L1)
    
    def range_(self, row):
        y = self.Uo.T @ row[:-1] + row[-1]
        return np.min(y), np.max(y)
    
    @ut.timer
    def range(self, rows):
        return np.array([self.range_(row) for row in rows])
    
    @ut.timer
    def range_parallel(self, rows):
        return np.array(Parallel(n_jobs=-1)(delayed(self.range_)(row) for row in rows))