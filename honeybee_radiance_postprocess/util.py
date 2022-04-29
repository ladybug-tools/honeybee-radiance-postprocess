"""Post-processing utility functions."""

import pyarrow as pa


def binary_mtx_dimension(filepath):
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


def numpy_to_pyarrow_table(array):
    """Convert a NumPy array to a PyArrow table.
    
    This functions uses the from arrays method to create a PyArrow table from a NumPy
    array.
    TODO: Possibly add names as input to this function.

    Args:
        array: NumPy array.
    
    Returns:
        A PyArrow table.
    """
    pa_arrays = [pa.array(row) for row in array]
    names = [str(i) for i in range(len(pa_arrays))]
    pa_table = pa.Table.from_arrays(pa_arrays, names=names)

    return pa_table
