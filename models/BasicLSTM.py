import torch

from torch import nn, einsum
from random import randint
from torch.nn import functional as f


class BasicLSTM(nn.Module):
    ''' Basic single-LSTM with output calculated using a single
        linear layer with a softmax activation '''
    
    def __init__(self, input_size, hidden_size, output_size):
        ''' Init the LSTM '''
        super(BasicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, input, hidden):
        ''' Run a forward path for a single number of sequence of numbers
        
        Returns:
            output and hidden layer after last sequence input
        '''
        # Hidden is made up of (hidden, cell_state)
        history, hidden = self.lstm.forward(input, hidden)
        output = self.hidden_to_output(hidden[0])
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        ''' Returns new hidden layers for the start of a new sequence '''
        model_device = next(self.parameters()).device
        return (
            torch.randn(1, 1, self.hidden_size).to(model_device),
            torch.randn(1, 1, self.hidden_size).to(model_device)
        )
