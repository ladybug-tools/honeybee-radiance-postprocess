"""Post-processing writer functions."""

import pyarrow.feather as feather
import os

from .util import numpy_to_pyarrow_table
from .reader import binary_to_numpy


def binary_to_feather(
        binary_file, filename=None, nrows=None, ncols=None, ncomp=None,
        output_folder=None):
    """Write a binary Radiance file to a feather file.
    
    The Radiance header will be read to determine NROWS, NCOLS, and NCOMP. IF these are
    known one can also use them as input to this function.
    
    Args:
        binary_file: Path to binary Radiance file.
        filename: Name of the feather file.
        nrows: Number of rows in the Radiance file.
        ncols: Number of columns in the Radiance file.
        ncomp: Number of components of each element in the Radiance file.
    
    Returns:
        Path to the feather file."""

    array = binary_to_numpy(binary_file, nrows=nrows, ncols=ncols, ncomp=ncomp)
    pa_table = numpy_to_pyarrow_table(array)
    
    head, tail = os.path.split(binary_file)
    if not output_folder:
        output_folder = head
    if not filename:
        root, ext = os.path.splitext(tail)
        filename = os.path.join(output_folder, root + '.feather')
    else:
        filename = os.path.join(output_folder, filename + '.feather')
    # write Table to feather
    feather.write_feather(pa_table, filename)

    return filename


def numpy_to_feather(array, filename, output_folder='.'):
    """Write a NumPy array to a feather file.

    The NumPy array will be convert to a PyArrow table which will be written to the
    feather file format.
    
    Args:
        array: A NumPy array.
        filename: Name of the feather file.
        output_folder: Optional output folder.
    
    Returns:
        Path to the feather file.
    """
    pa_table = numpy_to_pyarrow_table(array)

    file = os.path.join(output_folder, filename + '.feather')
    # write Table to feather
    feather.write_feather(pa_table, file)

    return file
