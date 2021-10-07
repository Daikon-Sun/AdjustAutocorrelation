import logging
import numpy as np
import sys
import os
import copy
import pickle
import importlib


from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


from args import args
from utils import create_dir, ForecastingData


class Runner:

    def __init__(self, model, rho, data):
        self.model = model
        self.rho = rho
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lmbd)
        self.rho_opt = torch.optim.Adam((self.rho,), lr=args.rho_lr)
        self.dataloaders = []
        for i in range(3):
            dataset = data.get_dataset(i)
            logging.info('Length of dataset: {}'.format(len(dataset)))
            self.dataloaders.append(DataLoader(dataset, batch_size=args.batch_size*(1+1*(i!=0)),
                                               shuffle=(i==0), num_workers=4))

        self.model_scheduler = LambdaLR(self.model_opt, lr_lambda=lambda e:args.model_decay_rate**e)
        self.rho_scheduler = LambdaLR(self.rho_opt, lr_lambda=lambda e:args.rho_decay_rate**e)

        self.bst_val_err = 1e9
        self.bad_limit = 0
        self.bst_model = model

    def run(self):
        pass

    def one_epoch(self, epoch, mode):
        pass
