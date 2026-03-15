# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 19:52:34 2022

@author: Gavin
"""

from pandas import DataFrame

from typing import Iterable

def all_windows(data: Iterable, window_width: int) -> list:
    num_elems = len(data)
    
    assert isinstance(window_width, int)
    
    assert window_width <= num_elems
    assert window_width > 0
    
    assert isinstance(data, Iterable)
    
    num_windows = num_elems - window_width + 1
    
    return [data[i : i + window_width] for i in range(num_windows)]



def format_new_column(df: DataFrame, new_column: list) -> None:
    col_length = len(new_column)
    df_length = len(df)
    
    for i in range(df_length - col_length):
        new_column.insert(0, float('nan'))
    
    
    
def add_column_to_df(df: DataFrame, new_column: list, rule: object) -> None:
    format_new_column(df, new_column)
    
    df[str(rule)] = new_column
    


def add_columns_to_df(df: DataFrame,
                      new_columns: Iterable[list],
                      rules: Iterable[object]
                      ) -> None:
    for new_column, rule in zip(new_columns, rules):
        add_column_to_df(df, new_column, rule)