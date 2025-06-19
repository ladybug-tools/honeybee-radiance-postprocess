"""Type hints for honeybee-radiance-postprocess."""
from typing import Tuple, List, Union
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

from ladybug.datacollection import HourlyContinuousCollection

ArrayLike = Union[np.ndarray, "cp.ndarray"]
annual_metric = Tuple[List[ArrayLike], List[dict]]
annual_daylight_metrics = Tuple[
    List[ArrayLike],
    List[ArrayLike],
    List[ArrayLike],
    List[ArrayLike],
    List[ArrayLike],
    List[dict]
]
annual_irradiance_metrics = Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[dict]
]
spatial_daylight_autonomy = Tuple[List[np.float64], List[dict]]
annual_sunlight_exposure = Tuple[List[np.float64], List[ArrayLike], List[dict]]
total = Tuple[List[ArrayLike], List[dict]]
point_in_time = Tuple[List[ArrayLike], List[dict]]
average_values = Tuple[List[ArrayLike], List[dict]]
median_values = Tuple[List[ArrayLike], List[dict]]
cumulative_values = Tuple[List[ArrayLike], List[dict]]
peak_values = Tuple[List[ArrayLike], List[dict]]
annual_data = Tuple[List[List[HourlyContinuousCollection]], List[dict], dict]
annual_uniformity_ratio = \
    Tuple[List[float], List[HourlyContinuousCollection], List[dict]]
