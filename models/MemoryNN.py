import torch

from torch import nn, einsum
from random import randint
from torch.nn import functional as f

from .Memory import Memory

class MemoryNN(nn.Module):
    def __init__(self, hidden_size, memory_banks, memory_dim):
        ''' Initialize the Memory NN
        
        Args:
            hidden_size: input size of the LSTM hidden layer
            memory_banks: count of memory banks to include
            memory_dim: dimension of each memory bank
        '''
        super(MemoryNN, self).__init__()
        self.hidden_size = hidden_size
        self.banks = memory_banks
        self.dim = memory_dim
        
        self.add_module('memory', Memory(memory_banks, memory_dim))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        
        # For key_strength(1), sharpening(1), shift(3), key, erase, add
        self.hidden_to_dim = nn.Linear(hidden_size, 1+1+3+3*memory_dim)
        self.hidden_to_gate = nn.Linear(hidden_size, memory_banks)
        
    
    def split_outputs(self, dim_output, banks_output):
        ''' Splits network output into intuitive groupings
        
        Args:
            dim_output: output in the memory_dim space
            banks_output: output in the memory_banks space
            
        Returns:
            dictionary with each conceptual group referenceable by key
        '''
        o = {}
        o['key_strength'] = dim_output[0]
        o['sharpening'] = dim_output[1]
        o['shift'] = dim_output[2:5]
        o['key'], o['erase'], o['add'] = self._split(dim_output[5:], self.dim)
        o['gate'] = banks_output
        
        return o
        
        
    def calculate_memory_weight(self, w_old, k, β, g, s, γ):
        ''' Calculates the read/write weight for the memory
        
        Args:
            w_old: output weight from last processes token
            k: content key to use for similarity lookup in memory
            β: coef to attenuate/amplify content key attention
            g: interpolation gate to raise/lower mix of w_c/w_old
            s: shift weights to convolve with w_g
            γ: sharpening exponent for final weights
        '''
        w_c = self.memory.get_content_similarity(k, β)
        w_g = self._interpolate(w_c, w_old, g)
        w_tilde = self._convolve(w_g, s)
        weight = self._sharpen(w_tilde, γ)
        
        return weight
    
        
    def forward(self, hidden, last_weight):
        ''' Runs the Memory network using the hidden layer output
        
        Args:
            hidden: hidden layer output from LSTM to use as input
        '''
        dim_output = self.hidden_to_dim(hidden).squeeze()
        gate_output = self.hidden_to_gate(hidden).squeeze()
        
        split_output = self.split_outputs(dim_output, gate_output)
        split_output['shift'] = self.softmax(split_output['shift'])
        split_output['gate'] = self.sigmoid(split_output['gate'])
        
        weight = self.calculate_memory_weight(
            w_old = last_weight,
            k = split_output['key'],
            β = split_output['key_strength'],
            g = split_output['gate'],
            s = split_output['shift'],
            γ = split_output['sharpening']
        )
        
        memory_read = self.memory.read(weight)
        
        self.memory.write(
            weight,
            split_output['erase'],
            split_output['add']
        )
        
        return memory_read, weight
    
    
    def reset_memory(self):
        ''' Resets the memory network for a new sequence
        
        Returns:
            pair of (initial_memory, initial_weights)
        '''
        self.add_module('memory', Memory(self.banks, self.dim))
        
        model_device = next(self.parameters()).device
        
        memory_read = torch.zeros(self.dim).to(model_device)
        weight = torch.zeros(self.banks).to(model_device)
        
        return (memory_read, weight)
    
    def _split(self, tensor, size):
        ''' Return a tensor split into chunks of length `size` '''
        count = tensor.shape[0] // size
        split = [tensor[i*size : (i+1)*size] for i in range(count)]
        return split
        
    def _interpolate(self, w1, w2, w1_mix):
        ''' Interpolate between two weights '''
        return w1_mix*w1 + (1-w1_mix)*w2
    
    def _sharpen(self, w, exponent):
        ''' Sharpen weight distribution '''
        w_sharpened = w ** exponent
        return w_sharpened / w_sharpened.sum()
    
    def _convolve(self, x, s):
        ''' Implement circular convolution between w and s '''
        # Create circular padding and reshape for conv2d
        x2 = torch.cat((x[-1:], x, x[:1])).view(1,1,1,-1)
        s2 = s.view(1,1,1,-1)

        # Run circular convolution and undo reshaping
        out = f.conv2d(x2, s2, padding=(0,1)).view(-1)

        # Return without padding
        return out[1:-1]
