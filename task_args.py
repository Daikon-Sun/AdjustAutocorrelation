import argparse


def forecasting_args(subparsers):
    p = subparsers.add_parser('forecasting')

    p.add_argument('--horizon', type=int, default=1, help='prediction horizon')
    p.add_argument('--model_type', type=str, default='LSTM', choices=['LSTM', 'LSTNet', 'DSANet', 'AGCRN', 'TCN', 'TPA', 'Trans'])
    p.add_argument('--dataset', type=str, default='solar')

    # LSTM
    p.add_argument('--lstm_n_layers', type=int, default=2)
    p.add_argument('--lstm_hidden_size', type=int, default=64)

    # DSANet
    p.add_argument('--dsanet_local', type=int, default=3)
    p.add_argument('--dsanet_n_kernels', type=int, default=32)
    p.add_argument('--dsanet_w_kernel', type=int, default=1)
    p.add_argument('--dsanet_d_model', type=int, default=512)
    p.add_argument('--dsanet_d_inner', type=int, default=2048)
    p.add_argument('--dsanet_d_k', type=int, default=64)
    p.add_argument('--dsanet_d_v', type=int, default=64)
    p.add_argument('--dsanet_n_head', type=int, default=8)
    p.add_argument('--dsanet_n_layers', type=int, default=1)
    p.add_argument('--dsanet_dropout', type=float, default=0.1)

    # AGCRN
    p.add_argument('--agcrn_n_layers', type=int, default=1)
    p.add_argument('--agcrn_hidden_size', type=int, default=64)
    p.add_argument('--agcrn_embed_dim', type=int, default=10)

    # TCN
    p.add_argument('--tcn_n_layers', type=int, default=9)
    p.add_argument('--tcn_hidden_size', type=int, default=64)
    p.add_argument('--tcn_dropout', type=int, default=0)

    # TPA
    p.add_argument('--tpa_n_layers', type=int, default=1)
    p.add_argument('--tpa_hidden_size', type=int, default=64)
    p.add_argument('--tpa_ar_len', type=int, default=24)

    # Trans
    p.add_argument('--trans_n_layers', type=int, default=3)
    p.add_argument('--trans_n_head', type=int, default=8)
    p.add_argument('--trans_hidden_size', type=int, default=256)
    p.add_argument('--trans_kernel_size', type=int, default=6)
