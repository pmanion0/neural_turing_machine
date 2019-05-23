import torch

from torch import nn, einsum
from torch.nn import functional as f

class Memory:
    ''' Implements an attention-based memory system as outlined in
        the Neural Turing Machine paper '''
    
    def __init__(self, banks, dim):
        ''' Initialize the memory store
        
        Args:
            banks: number of memory banks
            dim: dimension of each memory bank
        '''
        self.memory = torch.zeros(banks, dim)
        
        
    def read(self, weights):
        ''' Reads stored memory based on the attention
        
        Args:
            weights: attention for each cell (size: banks)
            
        Returns:
            weighted memory output (size: dim)
        '''
        attention = f.normalize(weights, p=1, dim=0)
        read = einsum('ij,i->j', (self.memory, attention))
        return read
    
    
    def write(self, weights, erase, add):
        ''' Updates attented memory with the erase and add
        
        Args:
            weights: attention paid to each cell (size: banks)
            erase: forget percent for each position (size: dim)
            add: amount to remember in each position (size: dim)
        '''
        attention = f.normalize(weights, p=1, dim=0)
        forget = einsum('i,j->ij', (attention, erase.clamp(0,1)))
        decay = torch.ones_like(forget) - forget
        
        remember = einsum('i,j->ij', (attention, add))
        
        self.memory = self.memory * decay + remember
        
    
    def get_content_similarity(self, query, key_strength = 1):
        ''' Provides similarity of query with each memory cell
        
        Args:
            query: reference memory to search for (size: dim)
            key_strength: scalar to amplify/attenuate attention
        '''
        similarity = f.cosine_similarity(self.memory, query.view(1,-1), dim=1)
        content_attention = f.softmax(key_strength * similarity, dim=0)
        return content_attention
