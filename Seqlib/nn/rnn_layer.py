import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel

# import torchsnooper

class EncoderRNN(BaseModel):
    def __init__(self, input_size, hidden_size, batch_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    # @torchsnooper.snoop()
    def forward(self, input_seq, hidden=None):
        """
        :input: batch_size * feature_size
        """
        if hidden is None:
            hidden = self.initHidden()
        
        embedded = self.embedding(input_seq).view(self.batch_size, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    # @torchsnooper.snoop()
    def forward(self, input_seq, hidden=None):
        """
        :input: batch_size * feature_size
        """

        if hidden is None:
            hidden = self.initHidden()

        output = self.embedding(input_seq).view(self.batch_size, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output.squeeze(dim=1)))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size)