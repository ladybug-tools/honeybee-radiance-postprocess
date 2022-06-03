"""Post-processing utility functions."""

import numpy as np
from typing import Tuple


def binary_mtx_dimension(filepath: str) -> Tuple[int, int, int, int]:
    """Return binary Radiance matrix dimensions if exist.

    This function returns NROWS, NCOLS, NCOMP and number of header lines including the
    white line after last header line.

    Args:
        filepath: Full path to Radiance file.

    Returns:
        nrows, ncols, ncomp, line_count
    """
    try:
        inf = open(filepath, 'rb', encoding='utf-8')
    except:
        # python 2
        inf = open(filepath, 'rb')
    try:
        first_line = next(inf).rstrip().decode('utf-8')
        if first_line[:10] != '#?RADIANCE':
            raise ValueError(
                'File with Radiance header must start with #?RADIANCE '
                'not {}.'.format(first_line)
                )
        header_lines = [first_line]
        nrows = ncols = ncomp = None
        for line in inf:
            line = line.rstrip().decode('utf-8')
            header_lines.append(line)
            if line[:6] == 'NROWS=':
                nrows = int(line.split('=')[-1])
            if line[:6] == 'NCOLS=':
                ncols = int(line.split('=')[-1])
            if line[:6] == 'NCOMP=':
                ncomp = int(line.split('=')[-1])
            if line[:7] == 'FORMAT=':
                break

        if not nrows or not ncols:
            raise ValueError(
                'NROWS or NCOLS was not found in the Radiance header. NROWS is %s and '
                'NCOLS is %s. The header must have both elements.' % (nrows, ncols)
                )
        return nrows, ncols, ncomp, len(header_lines) + 1
    finally:
        inf.close()


def check_array_dim(array: np.ndarray, dim: int):
    """Check NumPy array dimension.
    
    Args:
        array: A NumPy array.
        dim: The dimension to check against.
    """
    assert array.ndim == dim, \
        'Expected {}-dimensional array. Dimension of array is {}'.format(dim, array.ndim)


def occupancy_filter(array, mask):
    """Convert a list of ones and zeros to a NumPy masking array.
    
    Args:
        mask: List of ones and zeros.
    
    Returns:
        A NumPy array of booleans."""
    return array[mask.astype(bool)]
