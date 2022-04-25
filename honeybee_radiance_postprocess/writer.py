import numpy as np
import pyarrow
import pyarrow.feather as feather


def binary_to_feather(binary_file, filename, nrows=None, ncols=None):
    """A very interesting description."""

    with open(binary_file, 'rb') as reader:
        if (nrows and ncols) is None:
            # get nrows, ncols and header line count
            nrows, ncols, line_count = _parse_header(binary_file)
            # skip first n lines from reader
            for i in range(line_count):
                reader.readline()
        data = np.fromfile(reader, dtype=np.float32).reshape(nrows, ncols)
    
    arrays = [pyarrow.array(column) for column in data]
    names = [str(n) for n in range(nrows)]

    table = pyarrow.Table.from_arrays(arrays, names=names)
    
    # write Table to feather
    feather.write_feather(table, filename)

    return filename


def _parse_header(filepath):
    """Return binary Radiance file header if exist.

    This function returns all NROWS, NCOLS and number of header lines including the white
    line after last header line.

    Args:
        filepath: Full path to Radiance file.

    Returns:
        nrows, ncols, line_count
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
        for line in inf:
            line = line.rstrip().decode('utf-8')
            header_lines.append(line)
            if line[:6] == 'NROWS=':
                nrows = int(line.split('=')[-1])
            if line[:6] == 'NCOLS=':
                ncols = int(line.split('=')[-1])
            if line[:7] == 'FORMAT=':
                break
        return nrows, ncols, len(header_lines) + 1
    finally:
        inf.close()
