import logging
import numpy as np
import sys
import os
import importlib


import torch
import torch.nn as nn


from args import args
from utils import create_dir, ForecastingData


def run():

    if args.task_type == 'forecasting':
        data = ForecastingData()

    # datasets = []
    # for i in range(3):
    #     datasets.append(data.get_dataset(i))

    def model_decay(epoch):
        return args.model_decay_rate**epoch
    def rho_decay(epoch):
        return args.rho_decay_rate**epoch

    model_package = importlib.import_module(f'models.{args.task_type}.{args.model_type}')
    org_model = getattr(model_package, args.model_type)().to(args.device)
    model = org_model

    if args.model_type == 'AGCRN':
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    # for name, param in model.named_parameters():
    #     logging.info(name, param.shape, param.requires_grad)

    total_num = sum([param.nelement() for param in model.parameters()])
    logging.info('total num of parameters: {}'.format(total_num))

    if args.task_type == 'forecasting':
        n_rho = 1 if args.one_rho else args.n_series

    if n_rho == 1:
        rho = torch.tensor(args.init_rho, device=args.device, requires_grad=not args.fix_rho)
    else:
        init_rho = np.ones(n_rho, dtype=np.float32) * args.init_rho
        rho = torch.tensor(init_rho, device=args.device, requires_grad=not args.fix_rho)

    runner_package = importlib.import_module(f'runner.{args.task_type}_runner')
    runner = getattr(runner_package, f'{args.task_type}Runner')(model, rho, data)
    runner.run()


if __name__ == '__main__':

    # torch.backends.cudnn.benchmark = True

    if not os.path.isdir(args.output_dir):
        create_dir(args.output_dir)

    # FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    output_file_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'), mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)

    logging.info(args)

    run()
