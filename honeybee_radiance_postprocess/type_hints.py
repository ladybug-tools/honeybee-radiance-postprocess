"""Type hints for honeybee-radiance-postprocess."""
from typing import Tuple, List
import numpy as np
from ladybug.datacollection import HourlyContinuousCollection

annual_metric = Tuple[List[np.ndarray], List[dict]]
annual_metrics = Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[dict]
]
spatial_daylight_autonomy = Tuple[List[np.float64], List[dict]]
annual_sunlight_exposure = Tuple[List[np.float64], List[np.ndarray], List[dict]]
point_in_time = Tuple[List[np.ndarray], List[dict]]
average_values = Tuple[List[np.ndarray], List[dict]]
cumulative_values = Tuple[List[np.ndarray], List[dict]]
peak_values = Tuple[List[np.ndarray], List[dict]]
annual_data = Tuple[List[List[HourlyContinuousCollection]], List[dict], dict]
