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
from runner.runner import Runner


class forecastingRunner(Runner):
    def __init__(self, model, rho, data):
        super().__init__(model, rho, data)
        self.criterion = nn.MSELoss()

    def run(self):
        bad_limit = 0

        for epoch in range(1, args.n_epochs+1):
            trn_results = self.one_epoch(epoch, 0)
            self.model_scheduler.step()
            self.rho_scheduler.step()

            epoch_info = 'epoch = {} , trn loss = {:.6f}'.format(epoch, trn_results['loss'])
            epoch_info += ' , trn err = {:.6f}'.format(trn_results['err'])
            epoch_info += ' , rho = {:.6f}'.format(np.mean(np.tanh(self.rho.cpu().detach().numpy())))

            with torch.no_grad():
                val_results = self.one_epoch(epoch, 1)
                epoch_info += ' , val loss = {:.6f}'.format(val_results['loss'])
                epoch_info += ' , val err = {:.6f}'.format(val_results['err'])

                tst_results = self.one_epoch(epoch, 2)
                epoch_info += ' , tst loss = {:.6f}'.format(tst_results['loss'])
                epoch_info += ' , tst err = {:.6f}'.format(tst_results['err'])

            logging.info(epoch_info)
            # print(trn_results['recon'].mean())

            pickle.dump(trn_results, open(os.path.join(args.output_dir, f'trn_results_{epoch}.pkl'), 'wb'))
            pickle.dump(val_results, open(os.path.join(args.output_dir, f'val_results_{epoch}.pkl'), 'wb'))
            pickle.dump(tst_results, open(os.path.join(args.output_dir, f'tst_results_{epoch}.pkl'), 'wb'))

            if val_results['err'] < self.bst_val_err:
                self.bst_val_err = val_results['err']
                bad_limit = 0
                self.bst_model = copy.deepcopy(self.model)
            else:
                bad_limit += 1
            if args.bad_limit > 0 and bad_limit >= args.bad_limit:
                break

    def one_epoch(self, epoch, mode):

        if mode == 0:
            self.model.train()
        else:
            self.model.eval()

        avg = torch.tensor(self.dataloaders[0].dataset.avg).to(args.device)
        sc = torch.tensor(self.dataloaders[0].dataset.sc).to(args.device)
        rse = self.dataloaders[mode].dataset.rse

        results = dict()
        epoch_err = 0
        epoch_loss = 0
        with torch.autograd.set_grad_enabled(mode==0):
            for i, (x, y) in enumerate(self.dataloaders[mode]):

                bs = x.shape[0]
                x = x.to(args.device)
                y = y.to(args.device)
                rho = torch.tanh(self.rho)

                if args.inp_adj:
                    inp = torch.cat([avg[None].repeat(bs, 1, 1), x], dim=1)
                    inp = inp[:, 1:] - rho * inp[:, :-1]
                else:
                    inp = x
                prd_y = self.model(inp)
                if args.out_adj:
                    prd_y += rho * x[:, -1]
                loss = self.criterion(y, prd_y)
                epoch_loss += loss.item() * bs

                epoch_err += torch.sum((y*sc - prd_y*sc)**2).item()

                if mode == 0:
                    loss.backward()
                    self.model_opt.step()
                    self.model_opt.zero_grad()
                    self.rho_opt.step()
                    self.rho_opt.zero_grad()

        results['err'] = (epoch_err/rse)**0.5
        results['loss'] = epoch_loss / len(self.dataloaders[mode].dataset)
        return results
