"""Post-processing utility functions."""

import numpy as np
from typing import Tuple, Union


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


def filter_array(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Filter a NumPy array by a masking array. The array will be passed as is if the
    mask is None.
    
    Args:
        array: A NumPy array to filter.
        mask: A NumPy array of ones/zeros or True/False.
    
    Returns:
        A filtered NumPy array.
    """
    if mask is not None:
        return array[mask.astype(bool)]
    else:
        return array


def hoys_mask(
        sun_up_hours: list, hoys: list, timestep: int) -> np.ndarray:
    """Create a NumPy masking array from a list of hoys.

    Args:
        sun_up_hours: A list of integers for the sun-up hours.
        hoys: A list of 8760 * timestep values for the hoys to select. If an empty
            list is passed, None will be returned.
        timestep: Integer for the timestep of the analysis.
    
    Returns:
        A NumPy array of booleans.
    """
    if len(hoys) != 0:
        schedule = [False] * (8760 * timestep)
        for hr in hoys:
            schedule[int(hr * timestep)] = True
        su_pattern = [schedule[int(h * timestep)] for h in sun_up_hours]
        return np.array(su_pattern)
