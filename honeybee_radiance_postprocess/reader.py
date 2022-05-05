"""Post-processing reader functions."""

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from .util import binary_mtx_dimension


def feather_to_array(
        filepath: str, memory_map: bool = True, use_threads: bool = True) -> np.ndarray:
    """Read a feather file as a NumPy array.

    This function will read the feather file as a PyArrow table and convert it to a NumPy
    array.
    
    Args:
        filepath: Path to a feather file.
        memory_map: Use memory mapping when opening file on disk.
        use_threads: Whether to parallelize reading using multiple threads.
    
    Returns:
        A NumPy array.
    """
    # read file to PyArrow table
    table = feather.read_table(filepath, memory_map=memory_map, use_threads=use_threads)
    np_arrays = [arr.to_numpy() for arr in table]
    array = np.array(np_arrays)
    
    return array


def feather_to_dataframe(filepath: str) -> pd.DataFrame:
    """Read a feather file as a Pandas dataframe.
    
    Args:
        filepath: Path to a feather file.
    
    Returns:
        A Pandas DataFrame.
    """
    # read file to Pandas DataFrame
    dataframe = feather.read_feather(filepath)
    
    return dataframe


def binary_to_array(
        binary_file: str, nrows: int = None, ncols: int = None, ncomp: int = None
        ) -> np.ndarray:
    """Read a Radiance binary file as a NumPy array.
    
    Args:
        binary_file: Path to binary Radiance file.
        nrows: Number of rows in the Radiance file.
        ncols: Number of columns in the Radiance file.
        ncomp: Number of components of each element in the Radiance file.
    
    Returns:
        A NumPy array.
    """

    with open(binary_file, 'rb') as reader:
        if (nrows or ncols or ncomp) is None:
            # get nrows, ncols and header line count
            nrows, ncols, ncomp, line_count = binary_mtx_dimension(binary_file)
            # skip first n lines from reader
            for i in range(line_count):
                reader.readline()
            
            if ncomp != 1:
                array = np.fromfile(reader, dtype=np.float32).reshape(nrows, ncols, ncomp)
            else:
                array = np.fromfile(reader, dtype=np.float32).reshape(nrows, ncols)

    return array
