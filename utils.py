import importlib
import logging
import os


import numpy as np
import torch
from torch.utils.data import Dataset


from args import args


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


class ForecastingData:
    def __init__(self):
        raw_X = self.load_data()
        assert raw_X.ndim == 2
        valid_set = int((1-args.valid_ratio-args.test_ratio)*raw_X.shape[0])
        test_set = int((1-args.test_ratio)*raw_X.shape[0])
        self.raw_Xs = np.split(raw_X, [valid_set, test_set])  # trn, val, tst split

        if args.local_norm:
            axis = 0
        else:
            axis = None

        if args.norm_type == 'none':
            self.sc = np.ones((1, 1))
            self.mn = np.zeros((1, 1))
        elif args.norm_type == 'minmax':
            self.mn = self.raw_Xs[0].min(axis=axis, keepdims=True)
            self.sc = self.raw_Xs[0].max(axis=axis, keepdims=True) - self.mn
        elif args.norm_type == 'standard':
            self.mn = self.raw_Xs[0].mean(axis=axis, keepdims=True)
            self.sc = self.raw_Xs[0].std(axis=axis, keepdims=True)

    def load_data(self):
        data_path = os.path.join(args.data_dir, f'{args.dataset}.npy')
        raw_X = np.load(data_path, allow_pickle=True)
        args.n_series = raw_X[0].shape[-1]
        return raw_X

    def get_dataset(self, i):
        return ForecastingDataset(self.raw_Xs[i], self.sc, self.mn)


class ForecastingDataset(Dataset):
    def __init__(self, raw_X, sc, mn):
        self.sc = sc
        self.mn = mn
        self.rse = np.sum((raw_X[args.series_len-1:] - np.mean(raw_X[args.series_len-1:]))**2)
        self.X = self.norm(raw_X).astype(np.float32)
        self.avg = self.X.mean(axis=0).astype(np.float32)

    def norm(self, X):
        return (X - self.mn) / self.sc

    def renorm(self, X):
        return X * self.sc + self.mn

    def __len__(self):
        return self.X.shape[0] - args.series_len

    def __getitem__(self, idx):
        return (self.X[idx:idx+args.series_len], self.X[idx+args.series_len])
