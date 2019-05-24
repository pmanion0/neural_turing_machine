import torch

from torch import nn, einsum
from random import randint
from torch.nn import functional as f

from .MemoryNN import MemoryNN

class NTM_LSTM(nn.Module):
    ''' Implementation of a Neural Turing Machine from the Graves et al paper (2014) 
    
    https://arxiv.org/pdf/1410.5401.pdf
    '''
    
    def __init__(self, input_size, hidden_size, output_size,
                 memory_banks, memory_dim, output_length = 1):
        ''' Init the NTM-LSTM '''
        super(NTM_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.memory_banks = memory_banks
        self.memory_dim = memory_dim
        self.output_length = output_length
        
        # Core LSTM and Memory
        self.lstm = nn.LSTM(input_size + memory_dim, hidden_size)
        self.add_module('mem_nn', MemoryNN(hidden_size, memory_banks, memory_dim))
        
        # Output Layer
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        
    
    def forward(self, input, state):
        ''' Run a forward path for a single number of sequence of numbers
        
        Args:
            state: contains [hidden, cell, memory, last_weight]
        
        Returns:
            output and hidden layer after last sequence input
        '''
        output = []
        hidden, cell, memory, weight = state
        
        for i in range(input.shape[0]):
            ntm_input = torch.cat((input[i].unsqueeze(0), memory.view(1,1,-1)), dim=2)
        
            _, (hidden, cell) = self.lstm.forward(ntm_input, (hidden, cell))
            memory, weight = self.mem_nn.forward(hidden, weight)
        
        output.append(self.softmax(self.hidden_to_output(hidden)))
        
        for j in range(1, self.output_length):
            ntm_input = torch.cat((output[-1], memory.view(1,1,-1)), dim=2)
        
            _, (hidden, cell) = self.lstm.forward(ntm_input, (hidden, cell))
            memory, weight = self.mem_nn.forward(hidden, weight)
            output.append(self.softmax(self.hidden_to_output(hidden)))
            
        output = torch.cat(output, dim=0)
        return output, (hidden, cell, memory, weight)
    
    
    def init_hidden(self):
        ''' Returns new hidden layers for the start of a new sequence '''
        memory, weight = self.mem_nn.reset_memory()
        
        model_device = next(self.parameters()).device

        return (
            torch.randn(1, 1, self.hidden_size).to(model_device),
            torch.randn(1, 1, self.hidden_size).to(model_device),
            memory.to(model_device),
            weight.to(model_device)
        )