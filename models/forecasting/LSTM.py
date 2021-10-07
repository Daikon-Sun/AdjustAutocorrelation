from args import args


import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_size = args.lstm_hidden_size
        self.lstm = nn.LSTM(input_size=args.n_series, hidden_size=self.hidden_size,
                            num_layers=args.lstm_n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, args.n_series)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(x)
        output = output.view(batch_size, seq_len, self.hidden_size)[:, -1]
        output = self.fc(output)
        return output

