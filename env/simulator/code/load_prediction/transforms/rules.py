# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 19:47:53 2022

@author: Gavin
"""

import pandas as pd

from pandas import DataFrame

from abc import ABC, abstractmethod

# from env.cloud_allocation.lib.simulator.code.load_prediction.util import all_windows, format_new_column
from env.simulator.code.load_prediction.util import all_windows, format_new_column

class Rule(ABC):
    
    @abstractmethod
    def execute(self, data: DataFrame) -> list:...
    
    @abstractmethod
    def __str__(self) -> str:...
        
    

class PriceChannel(Rule):
    
    def __init__(self, window_width: int, tracked_value: str='counts') -> None:
        super().__init__()
        self.window_width = window_width
        self.tracked_value = tracked_value
        
    def execute(self, data: DataFrame, tracked_value: str=None) -> list:
        if not tracked_value is None: self.tracked_value = tracked_value
        
        values = data[self.tracked_value]       
        
        windows = all_windows(values, self.window_width)
        
        max_mins = [(max(window), min(window)) for window in windows]
        
        scores = []
        
        for i, pair in enumerate(max_mins):
            span = (pair[0] - pair[1])
            
            if span == 0: span = float('nan')
            
            score = (values.iloc[self.window_width + i - 1] - pair[1])  / span
            
            scores.append(score)
        
        return scores
    
    def __str__(self) -> str:
        return 'price_channel_w{:}_tv{:}'.format(self.window_width, self.tracked_value)
    
    
    
class BollPriceChannel(Rule):
    
    def __init__(self, tracked_value: str='counts') -> None:
        super().__init__()
        self.tracked_value = tracked_value
        
    def execute(self, data: DataFrame, tracked_value: str=None) -> list:
        if not tracked_value is None: self.tracked_value = tracked_value
        
        tracked_values = data[self.tracked_value]
        
        cols = data.columns
        
        upper_name = list(filter(lambda x: 'upper' in x and self.tracked_value in x, cols))[0]
        lower_name = list(filter(lambda x: 'lower' in x and self.tracked_value in x, cols))[0]
        
        upper_values = data[upper_name].values
        lower_values = data[lower_name].values
        
        self.tracked_value_upper = upper_name
        self.tracked_value_lower = lower_name
        
        max_mins = [(upper, lower) for upper, lower in zip(upper_values, lower_values)]
        
        scores = []
        
        for i, pair in enumerate(max_mins):
            span = (pair[0] - pair[1])
            
            if span == 0: span = float('nan')
            
            score = tracked_values.iloc[i] / span
            
            scores.append(score)
        
        format_new_column(data, scores)
        
        return scores
    
    def __str__(self) -> str:
        return 'boll_price_channel_tvu{:}_tvl{:}_tv{:}'.format(self.tracked_value_upper,
                                                               self.tracked_value_lower,
                                                               self.tracked_value)
    
    

class MovingAverage(Rule):
    
    def __init__(self, window_width_fast: int,
                 window_width_slow: int,
                 tracked_value: str='counts'
                 ) -> None:
        super().__init__()
        self.window_width_fast = window_width_fast
        self.window_width_slow = window_width_slow
        self.tracked_value = tracked_value

    def execute(self, data: DataFrame, tracked_value: str=None) -> list:    
        if not tracked_value is None: self.tracked_value = tracked_value
        
        tracked_values = data[self.tracked_value]
        
        fast_windows = all_windows(tracked_values, self.window_width_fast)
        slow_windows = all_windows(tracked_values, self.window_width_slow)
        
        fast_slow_delta = self.window_width_slow - self.window_width_fast
        
        for i in range(fast_slow_delta):
            slow_windows.insert(0, pd.Series([float('nan') for i in range(self.window_width_slow)]))
        
        fast_slow = [(sum(fast) / len(fast), sum(slow) / len(slow)) for fast, slow in zip(fast_windows, slow_windows)]
        
        scores = []
        
        for fast, slow in fast_slow:
            delta = fast - slow 
            
            score = delta / ((fast + slow) / 2)
            
            scores.append(score)
            
        return scores
    
    def __str__(self) -> str:
        return 'moving_average_wf{:}_ws{:}_tv{:}'.format(self.window_width_fast,
                                                         self.window_width_slow,
                                                         self.tracked_value)
    


class RateOfChange(Rule):
    
    def __init__(self, n: int, tracked_value: str='counts') -> None:
        super().__init__()
        self.n = n
        self.tracked_value = tracked_value
        
    def execute(self, data: DataFrame, tracked_value: str=None) -> list:
        if not tracked_value is None: self.tracked_value = tracked_value
        
        tracked_values = data[self.tracked_value]
        
        start_offset = self.n - 1
        
        num_instances = len(data)
        
        scores = [tracked_values.iloc[i] / tracked_values.iloc[i - self.n] for i in range(start_offset, num_instances)]
        
        return scores
    
    def __str__(self) -> str:
        return 'rate_of_change_n{:}_tv{:}'.format(self.n, self.tracked_value)
    