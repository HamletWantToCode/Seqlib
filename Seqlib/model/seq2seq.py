import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel
from ..nn import EncoderRNN, DecoderRNN
from ..data_loader import EOS_token, SOS_token

# import torchsnooper


class Seq2Seq(BaseModel):
    def __init__(self, input_size, hidden_size, output_size, batch_size, teaching=False):
        super(Seq2Seq, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size

        self.teaching = teaching
        self.encoder = EncoderRNN(input_size, hidden_size, batch_size)
        self.decoder = DecoderRNN(hidden_size, output_size, batch_size)
    
    # @torchsnooper.snoop()
    def forward(self, input_seq, target_seq, hidden=None):
        """
        :input_seq: batch_size * seq_size
        :target_seq: batch_size * seq_size
        :hidden: layer_size * batch_size
        """
        input_length = input_seq.size(1)
        target_length = target_seq.size(1)

        for ix in range(input_length):
            _input_word = input_seq[:, ix].unsqueeze(dim=1)   # _input_word: n_batch * 1
            _, hidden = self.encoder(_input_word, hidden)
        
        decoder_input = torch.repeat_interleave(
            torch.tensor([[SOS_token]]),
            self.batch_size,
            dim=0
        )
        
        output_seq = torch.zeros(self.batch_size, target_length, self.output_size)

        if self.teaching:
            for ix in range(target_length):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                decoder_input = target_seq[:, ix].unsqueeze(dim=1)
                output_seq[:, ix, :] = decoder_output
        else:
            for ix in range(target_length):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                topv, topi = decoder_output.topk(1, dim=1)       # topi: n_batch * 1
                decoder_input = topi.detach()
                output_seq[:, ix, :] = decoder_output

        return output_seq, hidden
        
