import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.l1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(1, self.batch_size, self.hidden_dim).cuda())

    def forward(self, sample):
        l1_out = F.elu(self.l1(sample))
        l2_out = F.elu(self.l2(l1_out))
        lstm_out, self.hidden = self.lstm(l2_out, self.hidden)
        out = self.out(lstm_out)
        return out
