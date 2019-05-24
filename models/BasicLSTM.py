import torch

from torch import nn, einsum
from random import randint
from torch.nn import functional as f


class BasicLSTM(nn.Module):
    ''' Basic single-LSTM with output calculated using a single
        linear layer with a softmax activation '''
    
    def __init__(self, input_size, hidden_size, output_size, output_length = 1):
        ''' Init the LSTM '''
        super(BasicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.output_length = output_length
        self.output_size = output_size
        self.softmax = nn.LogSoftmax(dim=2)
        
        
    def forward(self, input, hidden):
        ''' Run a forward path for a single number of sequence of numbers
        
        Hidden is made up of (hidden, cell_state)
        
        Returns:
            output and hidden layer after last sequence input
        '''
        output = []
        
        history, hidden = self.lstm.forward(input, hidden)
        
        output.append(self.softmax(self.hidden_to_output(hidden[0])))
        
        for j in range(1, self.output_length):
            history, hidden = self.lstm.forward(output[-1], hidden)
            output.append(self.softmax(self.hidden_to_output(hidden[0])))
            
        output = torch.cat(output, dim=0)
        return output, hidden
    
    
    def init_hidden(self):
        ''' Returns new hidden layers for the start of a new sequence '''
        model_device = next(self.parameters()).device
        return (
            torch.randn(1, 1, self.hidden_size).to(model_device),
            torch.randn(1, 1, self.hidden_size).to(model_device)
        )