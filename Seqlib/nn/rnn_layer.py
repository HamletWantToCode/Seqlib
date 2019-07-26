import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel

# import torchsnooper

class EncoderRNN(BaseModel):
    def __init__(self, input_size, hidden_size, batch_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)

    # @torchsnooper.snoop()
    def forward(self, input_seq, hidden=None):
        """
        :input: batch_size * 1
        """
        if hidden is None:
            hidden = self.initHidden()
        
        embedded = self.embedding(input_seq)      # embedded: n_batch * 1 * n_hidden
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers, self.batch_size, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    # @torchsnooper.snoop()
    def forward(self, input_seq, hidden=None):
        """
        :input: batch_size * 1
        """

        if hidden is None:
            hidden = self.initHidden()

        output = self.embedding(input_seq)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)    # output: n_batch * 1 * n_hidden
        output = self.softmax(self.out(output.squeeze(dim=1)))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers, self.batch_size, self.hidden_size)