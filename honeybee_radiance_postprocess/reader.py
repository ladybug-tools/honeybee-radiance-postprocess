"""Post-processing reader functions."""
import numpy as np

from .util import binary_mtx_dimension


def binary_to_array(
        binary_file: str, nrows: int = None, ncols: int = None,
        ncomp: int = None, line_count: int = 0) -> np.ndarray:
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
        ascii_file: str, nrows: int = None, ncols: int = None,
        ncomp: int = None, line_count: int = 0) -> np.ndarray:
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
