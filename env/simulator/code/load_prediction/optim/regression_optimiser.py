# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:41:57 2022

@author: Gavin
"""

import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader

from typing import Dict, Any, Callable, Iterable

class RegressionOptimiser():
    
    def __init__(self,
                 model: nn.Module,
                 optimiser: optim.Optimizer,
                 optim_params: Dict[str, Any],
                 loss_func: Callable
                 ) -> None:
        
        self.model = model
        self.optimiser = optimiser(model.parameters(), **optim_params)
        self.loss_func = loss_func
        
        
        
    def train(self, 
              train_dataloader: Iterable[DataLoader],
              test_dataloader: Iterable[DataLoader]=None,
              epochs: int=100
              ) -> None:
        
        self.training_history = {}
        
        print('--beginning training--')
        
        for epoch in range(epochs):  # loop over the dataset multiple times
            if epoch % 20 == 0:   
                print(f'training epoch {epoch}...')
        
            epoch_history = {'train' : {'loss' : None}}
            losses = []
            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                
                self.optimiser.zero_grad()
                
                outputs = self.model(inputs)
                
                loss = self.loss_func(outputs, labels)
                loss.backward()
                
                self.optimiser.step()
                
                losses.append(loss.item())
            
            epoch_history['train']['loss'] = np.mean(losses)
            
            if test_dataloader is not None:
                epoch_history['test'] = {'loss' : None}
                
                losses = []
                for i, data in enumerate(test_dataloader):
                    inputs, labels = data
                
                    outputs = self.model(inputs)
                        
                    loss = self.loss_func(outputs, labels)
                        
                    losses.append(loss.item())
                    
            epoch_history['test']['loss'] = np.mean(losses)
                    
            self.training_history[epoch] = epoch_history

                