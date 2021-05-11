"""
post_analysis.py

Classes and methods for the post analysis after model
training.

    - WeightChangeCalculator: interface for calculating the weight
    change
    - 
- 
"""

import pandas as pd
from typing import Callable, Optional, List
from src.utils import get_double_iqr
from src.utils import apply_median_filter, apply_double_median_filter
from src.data import load_weight_sensor


class WeightChangeCalculator:
    def __init__(self, user_id):
        self.user_id = user_id
        self.total_weight_clean = load_weight_sensor.get_total_weight_clean(
            user_id)

    def get_total_weight_smoothed(
        self,
        smooth_method: Optional[Callable] = apply_median_filter,
        **kwargs
    ) -> pd.Series:
        """
        Smooth the total weight data with the assigned method.
        """
        return smooth_method(self.total_weight_clean, **kwargs)

    def get_weight_change(
        self,
        start_stop_list: List[List[float]],
        smooth_method: Optional[Callable] = apply_double_median_filter,
        diff_method: Optional[Callable] = get_double_iqr,
        **kwargs
    ) -> float:
        """
        Get the weight change during a list of [start, stop]
        """
        total_weight_smooth = self.get_total_weight_smoothed(
            smooth_method, **kwargs)

        res = 0
        for start_stop in start_stop_list:
            start, stop = start_stop
            total_weight_within = total_weight_smooth[
                (total_weight_smooth.index >= start) &
                (total_weight_smooth.index <= stop)
            ]

            res += diff_method(total_weight_within.values)

        return res
