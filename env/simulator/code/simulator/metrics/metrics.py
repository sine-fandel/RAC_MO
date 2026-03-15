# -*- coding: utf-8 -*-
"""
Created on Wed May 11 21:36:49 2022

@author: Gavin
"""

from abc import ABC, abstractmethod

from typing import Callable

class Metric(Callable, ABC):

    @abstractmethod
    def __call__(self, stimulator: object) -> dict:...