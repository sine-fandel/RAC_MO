# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:12:21 2022

@author: Gavin
"""

import json

import pandas as pd

from pandas import DataFrame

def parse_parameters_file(fname: str, fdir: str) -> dict:
    file = open(fdir + '/' + fname, 'r')
    
    params = json.load(file)
    
    file.close()
    
    return params

def load_pm_types(fdir: str) -> DataFrame:
    return pd.read_csv(fdir + '/PMConfig_Amazon.csv',
                       header=None,
                       names=['cpu-max', 'memory-max', 'idle-power', 'max-power', 'cores-num'])

def load_vm_types(fdir: str) -> DataFrame:
    return pd.read_csv(fdir + '/VMConfig_c5New_cores.csv',
                       header=None,
                       names=['cpu-max', 'memory-max', 'cores-num'])