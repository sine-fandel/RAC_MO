# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:50:56 2022

@author: Gavin
"""

import os

from env.cloud_allocation.lib.simulator.code.simulator.config import SNAPSHOT_DIR

def save_model_snapshot(sim_env, generation, save_dir) -> None:
    simulator = sim_env.env
    
    containers = sim_env.containers
    vms = simulator.state.vm_resources
    pms = simulator.state.pm_resources
    
    num_containers = len(containers)
    
    start_allocation = containers.loc[0][2]
    end_allocation = containers.loc[num_containers - 1][2]
    
    delta_t = end_allocation - start_allocation
    E = simulator.running_energy_consumption
    
    avg_P = E / delta_t
    
    num_vms = len(vms)
    num_pms = len(pms)
    
    avg_P_per_container = avg_P / num_containers
    
    save_to = SNAPSHOT_DIR + '/' + save_dir
    
    if not os.path.isdir(save_to): os.mkdir(save_to)
    
    individual = 0
    
    fname = '/generation{:}_individual{:}_snapshot.txt'.format(generation, individual)
    fname = save_to + fname
    
    while os.path.isfile(fname):
        individual += 1
        
        fname = '/generation{:}_individual{:}_snapshot.txt'.format(generation, individual)
        fname = save_to + fname
    
    file = open(fname, 'w')
    
    data_to_write = ', '.join(map(str, [E, delta_t, avg_P, num_containers, num_vms, num_pms, avg_P_per_container]))
    
    file.write(data_to_write)
    
    file.close()