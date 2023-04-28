
# Quoted from
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
#自己改了些，从loss换成recall了

import torch

from Params import args


class EarlyStopping:
    def __init__(self, filename='',patience=5, verbose=False, delta=0, path='./Saved/', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_min = 0
        self.delta = delta
        self.path = path+filename+"-checkpoint.pth"
        self.trace_func = trace_func
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, state):
        if self.verbose:
            self.trace_func.info(f'metrics ({self.score_min:{args.outAcc}} --> {score:{args.outAcc}}).  Saving model ...')
        torch.save(state, self.path)
        self.score_min = score