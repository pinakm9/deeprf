import numpy as np


class SimpleLRScheduler:
    def __init__(self, optimizer, update_fraction=0.1, update_threshold=-1e-4):
        self.optimizer = optimizer
        self.update_fraction = update_fraction
        self.update_threshold = update_threshold 

    def step(self, change):
        if change > self.update_threshold:
            if change > 0:
                self.optimizer.param_groups[0]['lr'] *= (1-self.update_fraction)
            else:
                self.optimizer.param_groups[0]['lr'] *= (1+self.update_fraction)


class PiecewiseLinearScheduler:
    def __init__(self, optimizer, milestones):
        self.optimizer = optimizer
        self.milestones = np.array(milestones)

    def step(self, step):
        for i, stone in enumerate(self.milestones[:, 1]):
            if step < stone:
                self.optimizer.param_groups[0]['lr'] = self.milestones[i][0]
                break 
        # if i == len(self.milestones[:, 0]):
        #     self.optimizer.param_groups[0]['lr'] = self.milestones[-1][1]
