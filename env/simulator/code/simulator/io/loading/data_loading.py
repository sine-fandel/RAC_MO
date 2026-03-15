# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:00:28 2022

@author: Gavin
"""

import os

import pandas as pd
import numpy as np


# import env.cloud_allocation.lib.simulator.code.simulator.config as config 
import env.simulator.code.simulator.config as config 

from .application import Application

from typing import Dict, Union, List

from pandas import DataFrame

DATASET_DIR_LOOKUP = { 'auvergrid' : config.AUVERGRID_DIR,
                       'bitbrains' : config.BITBRAINS_DIR,
                       'bitbrains2' : config.BITBRAINS_DIR2 }

FULL_CONTAINER_COLS = ['cpu', 'memory', 'timestamp']
CONTAINER_COLS = ['cpu', 'memory']
OS_COLS = ['os-id']
PM_TYPE_COLS = ['pm-type-id']
VM_TYPE_COLS = ['vm-type-id']

COLUMN_NAMES_LOOKUP = { 'container-data' : FULL_CONTAINER_COLS,
                        'container' : CONTAINER_COLS,
                        'os' : OS_COLS,
                        'pmType' : PM_TYPE_COLS,
                        'vmType' : VM_TYPE_COLS }

def load_init_env_data(test_num: int, workload="bitbrains", os_dataset="3OS") -> Dict[str, DataFrame]:
    # dataset = config.RUNNING_PARAMS['dataset']
    # os_dataset = config.RUNNING_PARAMS['OS-dataset']
    os_dataset = os_dataset
    dataset = workload
    
    env_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/InitEnv/testCase' + str(test_num)
    init_env_data = {}
    
    fnames = os.listdir(env_data_dir)
    fnames.sort()
    
    for fname in fnames:
        key = fname[:-4]
        value = __read_init_env_data_file(fname, env_data_dir + '/' + fname)
        
        init_env_data[key] = value   
    
    return init_env_data

def load_valid_init_env_data() -> Dict[str, DataFrame]:
    dataset = config.RUNNING_PARAMS['dataset']
    os_dataset = config.RUNNING_PARAMS['OS-dataset']
    
    env_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/InitEnv/testCase' + '-1'
    
    init_env_data = {}
    
    fnames = os.listdir(env_data_dir)
    fnames.sort()
    
    for fname in fnames:
        key = fname[:-4]
        value = __read_init_env_data_file(fname, env_data_dir + '/' + fname)
        
        init_env_data[key] = value   
        
    return init_env_data

def load_container_data(test_num: int, workload="bitbrains", os_dataset="3OS") -> DataFrame:
    # dataset = config.RUNNING_PARAMS['dataset']
    # os_dataset = config.RUNNING_PARAMS['OS-dataset']
    os_dataset = os_dataset
    dataset = workload

    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/containerData/testCase{:}.csv'.format(test_num)

    containers = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['container-data'])

    container_data_dir_os = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/OSData/testCase{:}.csv'.format(test_num)

    os = pd.read_csv(container_data_dir_os, header=None, names=COLUMN_NAMES_LOOKUP['os'])

    '''
    Generate application
    '''
    start_id, end_id = 0, 0
    applications = {}
    applications_reverse = {}

    containers_os = pd.concat([containers, os], axis=1)
    groups = containers_os.groupby(["timestamp", "os-id"])
    num_containers = 0
    pattern = 0
    for name, group in groups: 
        num_containers += group.shape[0]        # number of containers
        group_id_list = group.index.to_list()
        start_id, end_id = 0, 0
        while start_id < len(group_id_list):
            """ the training instances
            randomly create applications by using the container at the same timestamp
            """
            if pattern < 7:
                pattern += 1
            else:
                pattern = 0
            if pattern == 0 and start_id + 5 < len(group_id_list):
                end_id = start_id + 5
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 1 and start_id + 8 < len(group_id_list):
                end_id = start_id + 8
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 2 and start_id + 6 < len(group_id_list):
                end_id = start_id + 6
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 3 and start_id + 9 < len(group_id_list):
                end_id = start_id + 9
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 4 and start_id + 10 < len(group_id_list):
                end_id = start_id + 10
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 5 and start_id + 11 < len(group_id_list):
                end_id = start_id + 11
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 6 and start_id + 12 < len(group_id_list):
                end_id = start_id + 12
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 7 and start_id + 13 < len(group_id_list):
                end_id = start_id + 13
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            else:
                break

            if end_id < num_containers:
                applications_reverse[app] = app.vector_id_list
                # print(app.vector_id_list)

            start_id = end_id + 1

    return containers, applications, applications_reverse

def load_os_data(test_num: int, workload="bitbrains", os_dataset="3OS") -> DataFrame:
    # dataset = config.RUNNING_PARAMS['dataset']
    # os_dataset = config.RUNNING_PARAMS['OS-dataset']
    os_dataset = os_dataset
    dataset = workload
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/OSData/testCase{:}.csv'.format(test_num)

    return pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['os'])


def load_test_container_data(test_num: int=0, start: int=0, end: int=None, run: int=0, dataset="bitbrains", os="3OS") -> DataFrame:
    import random
    random.seed(run + run * 199)
    # dataset = config.RUNNING_PARAMS['dataset']
    # os_dataset = config.RUNNING_PARAMS['OS-dataset']
    dataset = dataset
    os_dataset = os
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/Test/containerData/testCase{:}.csv'.format(test_num)

    containers = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['container-data'])[start: end]
    # all_containers = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['container-data'])
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/Test/OSData/testCase{:}.csv'.format(test_num)
    os = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['os'])

    
    '''
    Generate application
    '''
    start_id, end_id = start, start
    applications = {}
    applications_reverse = {}
    containers_os = pd.concat([containers, os], axis=1)
    groups = containers_os.groupby(["timestamp", "os-id"])
    num_containers = start
    pattern = 0
    for name, group in groups: 
        num_containers += group.shape[0]        # number of containers
        group_id_list = group.index.to_list()
        start_id, end_id = 0, 0
        while start_id < len(group_id_list):
            """ the training instances
            randomly create applications by using the container at the same timestamp
            """
            if pattern < 7:
                pattern += 1
            else:
                pattern = 0
            if pattern == 0 and start_id + 5 < len(group_id_list):
                end_id = start_id + 5
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 1 and start_id + 8 < len(group_id_list):
                end_id = start_id + 8
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 2 and start_id + 6 < len(group_id_list):
                end_id = start_id + 6
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 3 and start_id + 9 < len(group_id_list):
                end_id = start_id + 9
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 4 and start_id + 10 < len(group_id_list):
                end_id = start_id + 10
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 5 and start_id + 11 < len(group_id_list):
                end_id = start_id + 11
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 6 and start_id + 12 < len(group_id_list):
                end_id = start_id + 12
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 7 and start_id + 13 < len(group_id_list):
                end_id = start_id + 13
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            else:
                break
            
            if end_id < num_containers:
                applications_reverse[app] = app.vector_id_list
                # print(app.vector_id_list)

            start_id = end_id + 1
            # print("start_id = ", start_id)
            # print("end_id = ", end_id)
            # print("num = ", num_containers)

            # if pattern < 7:
            #     pattern += 1
            # else:
            #     pattern = 0

        # print("++++++++++++++++++++++++++++++")
        # start_id -= 1
    return containers, applications, applications_reverse

def load_test_os_data(start: int=None, end: int=None, test_num: int=0, dataset="bitbrains", os="3OS") -> DataFrame:
    # dataset = config.RUNNING_PARAMS['dataset']
    # os_dataset = config.RUNNING_PARAMS['OS-dataset']
    dataset = dataset
    os_dataset = os
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/Test/OSData/testCase{:}.csv'.format(test_num)

    return pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['os'])


def __read_init_env_data_file(fname: str, fdir: str) -> Union[List[List[int]], DataFrame]:
    fname = fname[:-4]
    
    if fname == 'pm' or fname == 'vm': 
        
        ids = []
        
        with open(fdir, 'r') as file:
            
            for line in file:
                
                line_ids = [int(val) for val in line.split(',')]
                ids.append(line_ids)
                
        return ids
    
    file_data = pd.read_csv(fdir, header=None, names=COLUMN_NAMES_LOOKUP[fname])
    
    return file_data

def load_be_container_data(test_num: int=-1, start: int=0, end: int=None, run: int=0) -> DataFrame:
    import random
    random.seed(run + run * 199)
    dataset = config.RUNNING_PARAMS['dataset']
    os_dataset = config.RUNNING_PARAMS['OS-dataset']
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/Test/containerData/testCase{:}.csv'.format(test_num)

    containers = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['container-data'])[start: end]
    # all_containers = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['container-data'])
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/Test/OSData/testCase{:}.csv'.format(test_num)

    os = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['os'])[start: end]

    
    '''
    Generate application
    '''
    start_id, end_id = start, start
    applications = {}
    applications_reverse = {}

    containers_os = pd.concat([containers, os], axis=1)
    groups = containers_os.groupby(["timestamp", "os-id"])
    num_containers = start
    pattern = 0
    for name, group in groups: 
        num_containers += group.shape[0]        # number of containers
        group_id_list = group.index.to_list()
        start_id, end_id = 0, 0
        while start_id < len(group_id_list):
            """ the training instances
            randomly create applications by using the container at the same timestamp
            """
            pattern = 0
            if pattern == 0 and start_id + 5 < len(group_id_list):
                end_id = start_id + 5
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 1 and start_id + 8 < len(group_id_list):
                end_id = start_id + 8
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 2 and start_id + 6 < len(group_id_list):
                end_id = start_id + 6
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 3 and start_id + 9 < len(group_id_list):
                end_id = start_id + 9
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 4 and start_id + 10 < len(group_id_list):
                end_id = start_id + 10
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 5 and start_id + 11 < len(group_id_list):
                end_id = start_id + 11
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 6 and start_id + 12 < len(group_id_list):
                end_id = start_id + 12
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            elif pattern == 7 and start_id + 13 < len(group_id_list):
                end_id = start_id + 13
                app = Application(group_id_list[start_id: end_id + 1], pattern=pattern, test_num=test_num)
            else:
                break
            
            if end_id < num_containers:
                applications_reverse[app] = app.vector_id_list
                # print(app.vector_id_list)

            start_id = end_id + 1
            # print("start_id = ", start_id)
            # print("end_id = ", end_id)
            # print("num = ", num_containers)

            # if pattern < 7:
            #     pattern += 1
            # else:
            #     pattern = 0

        # print("++++++++++++++++++++++++++++++")
        # start_id -= 1

    return containers, applications, applications_reverse

def load_be_os_data(start: int=None, end: int=None, test_num: int=-1) -> DataFrame:
    dataset = config.RUNNING_PARAMS['dataset']
    os_dataset = config.RUNNING_PARAMS['OS-dataset']
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/Test/OSData/testCase{:}.csv'.format(test_num)

    return pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['os'])[start: end]


