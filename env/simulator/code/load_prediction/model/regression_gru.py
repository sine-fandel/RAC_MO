# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:40:36 2022

@author: Gavin
"""

import torch

from torch import nn

class RegressionGRU(nn.Module):
    
    def __init__(self, in_dims, out_dims,
                 num_layers=2, layer_dims=128,
                 dropout_prob=0.99, cuda=False):
        super().__init__()
        
        self.cuda = cuda
        
        self.in_dims = in_dims
        self.layer_dims = layer_dims
        self.out_dims = out_dims
        
        self.num_layers = num_layers
        
        self.dropout_prob = dropout_prob
        
        self.gru = nn.GRU(input_size=in_dims, hidden_size=layer_dims,
                          num_layers=num_layers, dropout=dropout_prob,
                          batch_first=True)
        
        self.fcout = nn.Linear(layer_dims, out_dims)
        #print(f'{in_dims}        {out_dims}')
        if cuda:
            self.gru = self.gru.cuda()
            self.fcout = self.fcout.cuda()
            
            
            
    def forward(self, x):
        #x = torch.unsqueeze(x, 0)
        batch_size = x.size(0)
        #batch_size = x.size(1)
        #print(self.gru)
        self.h = torch.zeros([
                self.num_layers,
                batch_size,
                self.layer_dims
            ], dtype=torch.float).requires_grad_()
        
        #print(x.shape)
        if self.cuda: self.h = self.h.cuda()
        
        x, self.h = self.gru(x, self.h.detach())
        #print(self.h)
        #print('------------------------------------------------------------------')
        #x = torch.tanh(x)
        #print('here')
        #print(self.h.shape)
        
        #x = self.fcout(self.h[0])
        
        #print(x.shape)
        #x = x.squeeze()
        
        x = self.fcout(x[:, -1, :])
        
        return x