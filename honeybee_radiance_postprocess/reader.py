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
        binary_file: str, nrows: int = None, ncols: int = None, ncomp: int = None,
        line_count: int = 0) -> np.ndarray:
    """Read a Radiance binary file as a NumPy array.
    
    Args:
        binary_file: Path to binary Radiance file.
        nrows: Number of rows in the Radiance file.
        ncols: Number of columns in the Radiance file.
        ncomp: Number of components of each element in the Radiance file.
        line_count: Number of lines to skip in the input file. Usually used to
            skip the header.
    
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

        array = np.fromfile(reader, dtype=np.float32)
        if ncomp != 1:
            array = array.reshape(nrows, ncols, ncomp)
        else:
            array = array.reshape(nrows, ncols)

    return array


def ascii_to_array(
        ascii_file: str, nrows: int = None, ncols: int = None, ncomp: int = None,
        line_count: int = 0) -> np.ndarray:
    """Read a Radiance ascii file as a NumPy array.
    
    Args:
        ascii_file: Path to ascii Radiance file.
        nrows: Number of rows in the Radiance file.
        ncols: Number of columns in the Radiance file.
        ncomp: Number of components of each element in the Radiance file.
        line_count: Number of lines to skip in the input file. Usually used to
            skip the header.
    
    Returns:
        A NumPy array.
    """
    with open(ascii_file, 'r') as reader:
        if (nrows or ncols or ncomp) is None:
            # get nrows, ncols and header line count
            # we can reuse binary_mtx_dimension though the input file is ascii
            nrows, ncols, ncomp, line_count = binary_mtx_dimension(ascii_file) 

        array = np.loadtxt(reader, dtype=np.float32, skiprows=line_count)
        if ncomp != 1:
            array = array.reshape(nrows, ncols, ncomp)
        else:
            array = array.reshape(nrows, ncols)

    return array
