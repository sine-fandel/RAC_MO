# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 21:23:12 2022

@author: Gavin
"""

import os, json, torch

from env.cloud_allocation.lib.simulator.code.simulator.config import MODELS_DIR

def save_load_predictor(model, training_history, name):
    save_dir = f'{MODELS_DIR}/load_predictors/{name}'
    
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    
    stats_fname = f'{save_dir}/training_history.json'
    
    with open(stats_fname, 'w') as file:
        json.dump(training_history, file)
        
        file.close()
        
    model_fname = f'{save_dir}/model'
        
    torch.save(model.state_dict(), model_fname)