"""Post-processing translator functions."""

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import os

from .reader import binary_to_array


def binary_to_feather(
        binary_file: str, filename: str = None, nrows: int = None, ncols: int = None,
        ncomp: int = None, output_folder: str = None) -> str:
    """Write a binary Radiance file to a feather file.
    
    The Radiance header will be read to determine NROWS, NCOLS, and NCOMP. If these are
    known one can also use them as input to this function.
    
    Args:
        binary_file: Path to binary Radiance file.
        filename: Name of the feather file.
        nrows: Number of rows in the Radiance file.
        ncols: Number of columns in the Radiance file.
        ncomp: Number of components of each element in the Radiance file.
    
    Returns:
        Path to the feather file.
    """

    array = binary_to_array(binary_file, nrows=nrows, ncols=ncols, ncomp=ncomp)
    pa_table = array_to_table(array)
    
    head, tail = os.path.split(binary_file)
    if not output_folder:
        output_folder = head
    if not filename:
        root, ext = os.path.splitext(tail)
        feather_file = os.path.join(output_folder, root + '.feather')
    else:
        feather_file = os.path.join(output_folder, filename + '.feather')
    # write Table to feather
    feather.write_feather(pa_table, feather_file)

    return feather_file


def array_to_feather(array: np.ndarray, filename: str, output_folder: str = '.',
        column_name: str = 'sensor') -> str:
    """Write a NumPy array to a feather file.

    The NumPy array will be converted to a PyArrow table which will be written to the
    feather file format.
    
    Args:
        array: A NumPy array.
        filename: Name of the feather file.
        output_folder: Optional output folder.
        column_name: A name to add before the column count. If the input is empty the
            column names will simply be the column count.
    
    Returns:
        Path to the feather file.
    """
    pa_table = array_to_table(array, column_name=column_name)

    feather_file = os.path.join(output_folder, filename + '.feather')
    # write Table to feather
    feather.write_feather(pa_table, feather_file)

    return feather_file


def array_to_table(array: np.ndarray, column_name: str = 'sensor') -> pa.Table:
    """Convert a NumPy array to a PyArrow table.
    
    This functions uses the from arrays method to create a PyArrow table from a NumPy
    array.
    TODO: Possibly add names as input to this function.

    Args:
        array: NumPy array.
        column_name: A name to add before the column count. If the input is empty the
            column names will simply be the column count.
    
    Returns:
        A PyArrow table.
    """
    if column_name:
        column_name = column_name + '_'
    pa_arrays = [pa.array(row) for row in array]
    names = ['%s%s' % (column_name, i) for i in range(len(pa_arrays))]
    table = pa.Table.from_arrays(pa_arrays, names=names)

    return table
