# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:02:08 2022

@author: Gavin
"""

import os, sys

from pathlib import Path

from .__config_loading import parse_parameters_file, load_pm_types, load_vm_types

CWD = Path(os.getcwd())

SIMULATOR_MODULE_DIR = str(CWD) + '/env/simulator'

PARAMETERS_FILES_DIR = SIMULATOR_MODULE_DIR + '/parameters_files'

ROOT_DATA_DIR = SIMULATOR_MODULE_DIR + '/data'

AUVERGRID_DIR = ROOT_DATA_DIR + '/auvergrid'
BITBRAINS_DIR = ROOT_DATA_DIR + '/bitbrains'
BITBRAINS_DIR2 = ROOT_DATA_DIR + '/bitbrains2'
RAWDATA_DIR = ROOT_DATA_DIR + '/rawData'
TEMPDATA_DIR = ROOT_DATA_DIR + '/tempData'

SNAPSHOT_DIR = SIMULATOR_MODULE_DIR + '/out/model_snapshots'

MODELS_DIR = SIMULATOR_MODULE_DIR + '/out/saved_models'

# RUNNING_PARAMS = {}#parse_parameters_file(sys.argv[1], PARAMETERS_FILES_DIR)
RUNNING_PARAMS = { 'dataset' : "bitbrains", 'OS-dataset' : "4OS"}

AMAZON_PM_TYPES = load_pm_types(RAWDATA_DIR)
AMAZON_VM_TYPES = load_vm_types(RAWDATA_DIR)

VM_CPU_OVERHEAD_RATE = 0.1
VM_MEMORY_OVERHEAD = 200

TRACKED_OPTIM_PARAMS = ['learning_rate'] 