"""Functions to calculate various metrics for 1D and 2D NumPy arrays."""

import numpy as np

from .util import check_array_dim


def da_array2d(
        array: np.ndarray, total_occ: int = None, threshold: float = 300) -> np.ndarray:
    """Calculate daylight autonomy for a 2D NumPy array.
    
    Args:
        array: A 2D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        threshold: Threshold value for daylight autonomy. Default: 300.
    
    Returns:
        A 1-dimensional NumPy array with the daylight autonomy for each row in the input
        array.
    """
    check_array_dim(array, 2)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.shape[1]

    da = np.apply_along_axis(
            da_array1d, 1, array, total_occ=total_occ, threshold=threshold)

    return da


def da_array1d(
        array: np.ndarray, total_occ: int = None, threshold: float = 300) -> np.float64:
    """Calculate daylight autonomy for a 1D NumPy array.
    
    Args:
        array: A 1D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        threshold: Threshold value for daylight autonomy. Default: 300.
        
    Returns:
        A NumPy float of the daylight autonomy.
    """
    check_array_dim(array, 1)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.size

    return np.float64((array >= threshold).sum() / total_occ * 100)


def cda_array2d(
        array: np.ndarray, total_occ: int = None, threshold: float = 300) -> np.ndarray:
    """Calculate continuos daylight autonomy for a 2D NumPy array.
    
    Args:
        array: A 2D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        threshold: Threshold value for continuos daylight autonomy. Default: 300.
    
    Returns:
        A 1-dimensional NumPy array with the continuos daylight autonomy for each row in
        the input array.
    """
    check_array_dim(array, 2)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.shape[1]

    cda = np.apply_along_axis(
            cda_array1d, 1, array, total_occ=total_occ, threshold=threshold)

    return cda


def cda_array1d(
        array: np.ndarray, total_occ: int = None, threshold: float = 300) -> np.float64:
    """Calculate continuos daylight autonomy for a 1D NumPy array.
    
    Args:
        array: A 1D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        threshold: Threshold value for continuos daylight autonomy. Default: 300.
        
    Returns:
        A NumPy float of the continuos daylight autonomy.
    """
    check_array_dim(array, 1)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.size

    return np.float64(
        np.where(array >= threshold, 1, array / threshold).sum() / total_occ * 100)


def udi_array2d(array: np.ndarray, total_occ: int = None, min_t: float = 100,
                max_t: float = 3000) -> np.ndarray:
    """Calculate useful daylight illuminance for a 2D NumPy array.
    
    Args:
        array: A 2D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        threshold: Threshold value for useful daylight illuminance. Default: 300.
    
    Returns:
        A 1-dimensional NumPy array with the useful daylight illuminance for each row in
        the input array.
    """
    check_array_dim(array, 2)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.shape[1]

    udi = np.apply_along_axis(
            udi_array1d, 1, array, total_occ=total_occ, min_t=min_t, max_t=max_t)

    return udi


def udi_array1d(array: np.ndarray, total_occ: int = None, min_t: float = 100,
                max_t: float = 3000) -> np.float64:
    """Calculate useful daylight illuminance for a 1D NumPy array.
    
    Args:
        array: A 1D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        min_t: Minimum threshold for useful daylight illuminance. Default: 100.
        max_t: Maximum threshold for useful daylight illuminance. Default: 3000.
        
    Returns:
        A NumPy float of the useful daylight illuminance.
    """
    check_array_dim(array, 1)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.size

    return np.float64(((array >= min_t) & (array <= max_t)).sum() / total_occ * 100)


def udi_lower_array2d(array: np.ndarray, total_occ: int = None, min_t: float = 100,
                      sun_down_occ_hours: int = 0) -> np.ndarray:
    """Calculate lower than useful daylight illuminance for a 2D NumPy array.
    
    Args:
        array: A 2D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        min_t: Minimum threshold for useful daylight illuminance. Default: 100.
    
    Returns:
        A 1-dimensional NumPy array with the lower than useful daylight illuminance for
        each row in the input array.
    """
    check_array_dim(array, 2)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.shape[1]

    udi = np.apply_along_axis(
            udi_lower_array1d, 1, array, total_occ=total_occ, min_t=min_t,
            sun_down_occ_hours=sun_down_occ_hours)

    return udi


def udi_lower_array1d(
        array: np.ndarray, total_occ: int = None, min_t: float = 100,
        sun_down_occ_hours: int = 0) -> np.float64:
    """Calculate lower than useful daylight illuminance for a 1D NumPy array.
    
    Args:
        array: A 1D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        min_t: Minimum threshold for useful daylight illuminance. Default: 100.
        
    Returns:
        A NumPy float of the lower than useful daylight illuminance.
    """
    check_array_dim(array, 1)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.size

    return np.float64(((array < min_t).sum() + sun_down_occ_hours) / total_occ * 100)


def udi_upper_array2d(
        array: np.ndarray, total_occ: int = None, max_t: float = 3000) -> np.ndarray:
    """Calculate higher than useful daylight illuminance for a 2D NumPy array.
    
    Args:
        array: A 2D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        max_t: Maximum threshold for useful daylight illuminance. Default: 3000.
    
    Returns:
        A 1-dimensional NumPy array with the higher than useful daylight illuminance for
        each row in the input array.
    """
    check_array_dim(array, 2)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.shape[1]

    udi = np.apply_along_axis(
            udi_upper_array1d, 1, array, total_occ=total_occ, max_t=max_t)

    return udi


def udi_upper_array1d(
        array: np.ndarray, total_occ: int = None, max_t: float = 3000) -> np.float64:
    """Calculate higher than useful daylight illuminance for a 1D NumPy array.
    
    Args:
        array: A 1D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        max_t: Maximum threshold for higher than useful daylight illuminance.
            Default: 3000.
        
    Returns:
        A NumPy float of the higher than useful daylight illuminance.
    """
    check_array_dim(array, 1)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.size

    return np.float64((array > max_t).sum() / total_occ * 100)
