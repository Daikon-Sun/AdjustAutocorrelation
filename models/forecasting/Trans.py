import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(x.shape, self.pe.shape)
        x = x + self.pe[:x.size(0), None, :]
        return self.dropout(x)


class Trans(nn.Module):

    def __init__(self):
        super(Trans, self).__init__()

        self.conv = nn.Conv1d(args.n_series, args.trans_hidden_size, kernel_size=args.trans_kernel_size)
        self.pos_encoder = PositionalEncoding(args.trans_hidden_size, max_len=args.series_len)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=args.trans_hidden_size, nhead=args.n_trans_head)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=args._trans_n_layers)
        self.fc = nn.Linear(args.trans_hidden_size, args.n_series)

        self.kernel_size = args.trans_kernel_size

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size-1,0))
        x = self.conv(x).permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)[-1]
        output = self.fc(x)
        return output

