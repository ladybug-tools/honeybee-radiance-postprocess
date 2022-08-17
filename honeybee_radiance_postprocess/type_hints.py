"""Type hints for honeybee-radiance-postprocess."""
from typing import Tuple
import numpy as np
from ladybug.datacollection import HourlyContinuousCollection

annual_metric = Tuple[list[np.ndarray], list[dict]]
annual_metrics = Tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[dict]
]

point_in_time = Tuple[list[np.ndarray], list[dict]]
average_values = Tuple[list[np.ndarray], list[dict]]
cumulative_values = Tuple[list[np.ndarray], list[dict]]
peak_values = Tuple[list[np.ndarray], list[dict]]
annual_data = Tuple[list[list[HourlyContinuousCollection]], list[dict], dict]
