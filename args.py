import os
import argparse


import task_args


def all_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_decay_rate', type=float, default=1)
    parser.add_argument('--rho_decay_rate', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rho_lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='data')

    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--fix_rho', action='store_true', help='fix rho at initial value')
    parser.add_argument('--lmbd', type=float, default=0.0, help='lambda value for regularization')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--bad_limit', type=int, default=0, help='# of non-improving epochs before early-stopping')
    parser.add_argument('--n_series', type=int, help='# of series in the data (assign automatically)')
    parser.add_argument('--series_len', type=int, default=64, help='window size of a single example')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--inp_adj', action='store_true', help='adjust for input')
    parser.add_argument('--out_adj', action='store_true', help='adjust for output')
    parser.add_argument('--norm_type', type=str, default='none', choices=['none', 'minmax', 'standard'])
    parser.add_argument('--local_norm', action='store_true', help='normalize data per series')
    parser.add_argument('--one_rho', action='store_true', help='one rho for all series')
    parser.add_argument('--init_rho', type=float, default=0.0, help='initial value of rho')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='ratio of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of data for testing')

    subparsers = parser.add_subparsers(dest='task_type', help='task_type')

    for name, task_arg in task_args.__dict__.items():
        if callable(task_arg):
            task_arg(subparsers)

    args = parser.parse_args()
    args.data_dir = os.path.join(args.data_dir, args.task_type)
    args.output_dir = os.path.join(args.output_dir, args.task_type)

    return args

args = all_args()
