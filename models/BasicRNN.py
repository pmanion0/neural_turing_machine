import torch

from torch import nn, einsum
from random import randint
from torch.nn import functional as f


class BasicRNN(nn.Module):
    ''' Basic RNN with a softmax output layer '''
    
    def __init__(self, input_size, hidden_size, output_size):
        ''' Init the RNN '''
        super(BasicRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.input_to_output = nn.Linear(input_size + hidden_size, output_size)
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, input, hidden):
        ''' Run a forward path for a single number of sequence of numbers
        
        Returns:
            output and hidden layer after last sequence input
        '''
        for i in range(input.shape[0]):
            combined_input = torch.cat((input[i].unsqueeze(0), hidden), dim=2)
            hidden = self.input_to_hidden(combined_input)
            
        output = self.input_to_output(combined_input)
        output = self.softmax(output)
        
        return output, hidden
    
    def init_hidden(self):
        ''' Returns new hidden layers for the start of a new sequence '''
        model_device = next(self.parameters()).device
        return torch.zeros(1, 1, self.hidden_size).to(model_device)