# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 15:45:29 2022

@author: Gavin
"""

def timestamp_aggregation(dataset):
    dataset = dataset.groupby(by=['timestamp'])
    
    counts = dataset.count()
    counts = counts.drop(columns=counts.columns[1:])
    counts = counts.rename(columns={counts.columns[0] : 'counts'})
    
    avgs = dataset.mean()
    
    dataset = avgs
    dataset['counts'] = counts['counts']
    
    return dataset