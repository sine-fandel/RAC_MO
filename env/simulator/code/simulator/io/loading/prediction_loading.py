# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:07:46 2022

@author: Gavin
"""

import torch

import numpy as np

import pandas as pd

from .data_loading import load_container_data

from torch.utils.data import DataLoader, Dataset

# from env.cloud_allocation.lib.simulator.code.load_prediction import transforms
# from env.cloud_allocation.lib.simulator.code.load_prediction import util
from env.simulator.code.load_prediction import transforms
from env.simulator.code.load_prediction import util

class ContainersDataset(Dataset):
    
    def __init__(self,
                 test_cases,
                 window_width,
                 cuda=False,
                 rules=None,
                 ys_features=['counts'],
                 formatting_transforms=None,
                 ys_transforms=None,
                 sample_frac=0.1):
        self.window_width = window_width
        
        datasets = [load_container_data(test_case) for test_case in test_cases]
        
        dataset = pd.concat(datasets)
        
        if sample_frac < 1:
            dataset = dataset.sample(frac=sample_frac)
            
            dataset = dataset.sort_index().reset_index(drop=True)
        
        if not formatting_transforms is None:
            for formatting_transform in formatting_transforms:
                dataset = formatting_transform(dataset)
        
        if not rules is None: self.__add_rules_transforms(dataset, rules)
        
        xs = []
        ys = []
        
        self.num_features = len(dataset.columns)
        
        for i, (_, row) in enumerate(dataset.iterrows()):
            ys.append(row[ys_features])
            
            window = [np.zeros(self.num_features) for _ in range(window_width)]
            
            for j in range(window_width):
                window_index = window_width - j - 1
                
                lookup_index = i - j - 1
                
                if lookup_index >= 0:
                    window[window_index] = dataset.iloc[lookup_index].to_numpy()
            
            window = np.array(window)
            
            xs.append(window)
            
        xs = torch.from_numpy(np.array(xs)).float()
        ys = torch.from_numpy(np.array(ys)).float()
        
        if cuda:
            self.xs = xs.cuda()
            self.ys = ys.cuda()
            
        else:
            self.xs = xs
            self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = self.xs[idx]
        y = self.ys[idx]

        return x, y
    
    def __add_rules_transforms(self, dataset, rules):
        cols = list(dataset.columns)
        
        if any([isinstance(rule, transforms.BollPriceChannel) for rule in rules]):
            for col in cols:
                upper, lower, n_upper, n_lower = transforms.bollinger_bands(dataset, self.window_width, col)
            
                util.add_columns_to_df(dataset, [upper, lower], [n_upper, n_lower])
        
        for rule in rules:
            for col in cols:
                scores = rule.execute(dataset, tracked_value=col)
                
                util.add_column_to_df(dataset, scores, rule)
            
        dataset.fillna(0, inplace=True)
        
        
            
def load_container_files(test_cases, window_width, batch_size=512, **kwargs):
    dataset = ContainersDataset(test_cases, window_width, **kwargs)
       
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    dataloader.num_features = dataset.num_features
    
    return dataloader
