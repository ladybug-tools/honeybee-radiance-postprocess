"""Functions for NumPy data type (dtype)."""
from typing import Tuple
import numpy as np


def smallest_integer_dtype(array: np.ndarray) -> np.signedinteger:
    """Return the smallest possible integer dtype.

    Args:
        array: NumPy array.
    
    Returns:
        A NumPy integer dtype.
    """
    if np.all(array >= np.iinfo(np.int8).min) and \
        np.all(array <= np.iinfo(np.int8).max):
        return np.int8
    elif np.all(array >= np.iinfo(np.int16).min) and \
        np.all(array <= np.iinfo(np.int16).max):
        return np.int16
    elif np.all(array >= np.iinfo(np.int32).min) and \
        np.all(array <= np.iinfo(np.int32).max):
        return np.int32
    elif np.all(array >= np.iinfo(np.int64).min) and \
        np.all(array <= np.iinfo(np.int64).max):
        return np.int64


def smallest_float_dtype(array: np.ndarray) -> np.floating:
    """Return the smallest possible float dtype.

    The allclose function is used to check if a certain floating-point precision
    can be used without losing accuracy.

    Args:
        array: NumPy array.
    
    Returns:
        A NumPy floating dtype.
    """
    if np.all((array >= np.finfo(np.float16).min) & \
                (array <= np.finfo(np.float16).max)):
        if np.allclose(array, array.astype(np.float16), rtol=1e-5, atol=1e-5):
            return np.float16
    if np.all((array >= np.finfo(np.float32).min) & \
                (array <= np.finfo(np.float32).max)):
        if np.allclose(array, array.astype(np.float32), rtol=1e-5, atol=1e-5):
            return np.float32
    if np.all((array >= np.finfo(np.float64).min) & \
                (array <= np.finfo(np.float64).max)):
        if np.allclose(array, array.astype(np.float64), rtol=1e-5, atol=1e-5):
            return np.float64


def smallest_dtype(array: np.ndarray) -> Tuple[np.signedinteger, np.floating]:
    """Return the smallest possible dtype.

    Args:
        array: NumPy array.
    
    Returns:
        A NumPy dtype.
    """
    if np.issubdtype(array, np.integer):
        return smallest_integer_dtype(array)
    elif np.issubdtype(array, np.floating):
        return smallest_float_dtype(array)
    else:
        raise TypeError(f'Expected integer or floating dtype. Got {array.dtype}')


def set_smallest_dtype(array: np.ndarray) -> np.ndarray:
    """Return a NumPy array with the smallest possible dtype.
    
    Args:
        array: NumPy array.
    
    Returns:
        A new NumPy array with a smaller dtype.
    """
    dtype = smallest_dtype(array)
    return array.astype(dtype)
