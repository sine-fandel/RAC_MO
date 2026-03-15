# -*- coding: utf-8 -*-
"""
Created on Wed May 11 21:43:03 2022

@author: Gavin
"""

# from env.cloud_allocation.lib.simulator.code.simulator.metrics import Metric

# from env.cloud_allocation.lib.simulator.code.simulator.config import (AMAZON_PM_TYPES,
#                                                                       VM_CPU_OVERHEAD_RATE,
#                                                                       VM_MEMORY_OVERHEAD)

from env.simulator.code.simulator.metrics import Metric

from env.simulator.code.simulator.config import (AMAZON_PM_TYPES,
                                                                      VM_CPU_OVERHEAD_RATE,
                                                                      VM_MEMORY_OVERHEAD)

import numpy as np

def vm_pm_compatible(pm_cpu_remaining: float, pm_memory_remaining: int, pm_core: int,
                     vm_cpu_capacity: float, vm_memory_capacity: int, vm_core: int
                     ) -> bool:
    return pm_cpu_remaining >= vm_cpu_capacity and pm_memory_remaining >= vm_memory_capacity and pm_core >= vm_core



class FirstFitPMAllocation(Metric):
    
    def __call__(self, *inputs) -> float:
        state, pm_selection_constraints = inputs
        vm_cpu_capacity, vm_memory_capacity, vm_used_cpu, vm_used_memory, vm_core = pm_selection_constraints
        
        pm_resources = state.pm_resources
        num_pms = len(pm_resources)
        
        vm_wrapped = (vm_cpu_capacity, vm_memory_capacity, vm_core)
        
        selected_pm = -1
        
        for pm_count, pm_stats in enumerate(pm_resources):
            pm_cpu_remaining = pm_stats[0]
            pm_memory_remaining = pm_stats[1]
            pm_core = pm_stats[4]
            
            pm_wrapped = (pm_cpu_remaining, pm_memory_remaining, pm_core)
            if vm_pm_compatible(*pm_wrapped, *vm_wrapped):
                # print(pm_stats)
                selected_pm = pm_count
                break
                
        if selected_pm == -1:
            
            for i, pm_stats in enumerate(AMAZON_PM_TYPES.values):
                
                pm_cpu_remaining, pm_memory_remaining, _, _, pm_cores = pm_stats
                
                pm_wrapped = (pm_cpu_remaining, pm_memory_remaining, pm_cores)
                
                if vm_pm_compatible(*pm_wrapped, *vm_wrapped):
                    selected_pm = num_pms + i
                    break

        return { 'pm_num' : selected_pm }
    
    
    
class BestFitCPUPMAllocation(Metric):
    
    def __call__(self, *inputs) -> float:
        state, pm_selection_constraints = inputs
        vm_cpu_capacity, vm_memory_capacity, vm_used_cpu, vm_used_memory, vm_core = pm_selection_constraints
        
        pm_resources = state.pm_resources
        running_machine = np.array(pm_resources)
        amazon_pm_types = np.array(AMAZON_PM_TYPES)
        total_pm_types = amazon_pm_types[:, : 2]
        amazon_pm_types = np.append(amazon_pm_types, total_pm_types, axis=1)
        
        if pm_resources == []:
            candidate_pms = amazon_pm_types
        else:
            candidate_pms = np.append(running_machine, amazon_pm_types, axis=0)

        pm_mapping = np.where((candidate_pms[:, 0] >= vm_cpu_capacity)&(candidate_pms[:, 1] >= vm_memory_capacity)&(candidate_pms[:, 4] >= vm_core))[0]
        avaiable_pms = candidate_pms[(candidate_pms[:, 0] >= vm_cpu_capacity)&(candidate_pms[:, 1] >= vm_memory_capacity)&(candidate_pms[:, 4] >= vm_core), :]

        vm_cpu_capacity = (vm_cpu_capacity) / (np.max(avaiable_pms[:, 0]))
        vm_memory_capacity = (vm_cpu_capacity) / (np.max(avaiable_pms[:, 1]))

        avaiable_pms[:, 0] = (avaiable_pms[:, 0]) / (np.max(avaiable_pms[:, 0]))
        avaiable_pms[:, 1] = (avaiable_pms[:, 1]) / (np.max(avaiable_pms[:, 1]))
        
        heuristic = vm_cpu_capacity / avaiable_pms[:, 0] + vm_memory_capacity / avaiable_pms[:, 1]

        action = { "pm_num": pm_mapping[np.argmax(heuristic)] }
        # print(f"heuristic = {heuristic}")
        # print(f"action = {action}")
        # print(f"pm_mapping = {pm_mapping}")
        # print(candidate_pms)
        # print("===============")
        return action


    


class BestFitMemoryPMAllocation(Metric):
    
    def __call__(self, *inputs) -> float:
        state, pm_selection_constraints = inputs
        vm_cpu_capacity, vm_memory_capacity, vm_used_cpu, vm_used_memory, vm_core = pm_selection_constraints
        
        pm_resources = state.pm_resources
        
        num_pms = len(pm_resources)
        
        vm_wrapped = (vm_cpu_capacity, vm_memory_capacity, vm_core)
        
        selected_pm = -1
        min_observed_diff = float('inf')
        
        for pm_count, pm_stats in enumerate(pm_resources):
            pm_cpu_remaining = pm_stats[0]
            pm_memory_remaining = pm_stats[1]
            pm_core = pm_stats[4]
            
            pm_wrapped = (pm_cpu_remaining, pm_memory_remaining, pm_core)
            
            candidate_diff = pm_memory_remaining - vm_memory_capacity
            
            is_new_min = candidate_diff < min_observed_diff 
            
            if vm_pm_compatible(*pm_wrapped, *vm_wrapped) and is_new_min:
                selected_pm = pm_count
                min_observed_diff = candidate_diff
                
        if selected_pm == -1:
            
            for i, pm_stats in enumerate(AMAZON_PM_TYPES.values):
                
                pm_cpu_remaining, pm_memory_remaining, _, _, pm_cores = pm_stats
                
                pm_wrapped = (pm_cpu_remaining, pm_memory_remaining, pm_cores)
                
                candidate_diff = pm_memory_remaining - vm_memory_capacity
                
                is_new_min = candidate_diff < min_observed_diff 
                
                if vm_pm_compatible(*pm_wrapped, *vm_wrapped) and is_new_min:
                    selected_pm = num_pms + i
                    min_observed_diff = candidate_diff
            
        return { 'pm_num' : selected_pm }