"""Post-processing reader functions."""

import numpy as np
import pyarrow.feather as feather

from .util import binary_mtx_dimension


def feather_to_numpy(filepath, pandas_dataframe=False):
    """Read a feather file a NumPy array.
    
    Args:
        filepath: Path to a feather file.
        pandas_dataframe: A boolean to note if feather should read the file as Pandas
            DataFrame. The function will still return a NumPy array, i.e., only reading
            of the file is affected by this option. The Pandas DataFram will be converted
            to NumPy.
    
    Returns:
        A NumPy array.
    """
    if not pandas_dataframe:
        # read file to PyArrow table
        table = feather.read_table(filepath)
        np_arrays = [arr.to_numpy() for arr in table]
        data = np.array(np_arrays)
    else:
        # read file to Pandas DataFrame
        dataframe = feather.read_feather(filepath)
        data = dataframe.to_numpy().transpose()
    
    return data


def binary_to_numpy(binary_file, nrows=None, ncols=None, ncomp=None):
    """Read a binary Radiance file as a NumPy array.
    
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
                data = np.fromfile(reader, dtype=np.float32).reshape(nrows, ncols, ncomp)
            else:
                data = np.fromfile(reader, dtype=np.float32).reshape(nrows, ncols)

    return data
