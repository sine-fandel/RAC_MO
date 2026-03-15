# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:42:32 2022

@author: Gavin
"""

from .regression_gru import RegressionGRU

class LoadPredictor(RegressionGRU):
    
    def __init__(self, dataset, out_dims, **kwargs):
        super().__init__(dataset.num_features, out_dims, **kwargs)