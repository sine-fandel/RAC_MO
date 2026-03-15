# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:41:30 2022

@author: Gavin
"""

import numpy as np

from pandas import DataFrame

from typing import Tuple

# from env.cloud_allocation.lib.simulator.code.load_prediction.util import all_windows
from env.simulator.code.load_prediction.util import all_windows


def bollinger_bands(data: DataFrame, window_width: int, target: str) -> Tuple[list, list, str, str]:
    col = target
    
    windows = all_windows(data[col], window_width)
    
    stds = np.std(windows, axis=1)
    
    values = data[col].iloc[window_width - 1:]
    
    boll_upper = [value + (2 * std)for value, std in zip(values, stds)]
    boll_lower = [value - (2 * std)for value, std in zip(values, stds)]
    
    return boll_upper, boll_lower, f'bollinger_upper_w-{window_width}_t-{col}', f'bollinger_lower_w-{window_width}_t-{col}'



def delta_price(data: DataFrame, target: str) -> list:
    col = target
    
    values = data[col]
    
    return [values.iloc[i + 1] - values.iloc[i] for i in range(len(values) - 1)], 'delta'


