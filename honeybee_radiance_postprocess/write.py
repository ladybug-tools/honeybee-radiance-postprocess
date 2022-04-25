import numpy as np
import pyarrow
import pyarrow.feather as feather


def binary_to_feather(binary_file, filename, nrows=None, ncols=None):
    """A very interesting description."""
    if (nrows and ncols) is None:
        # read Radiance header and find nrows and ncols
        pass

    with open(binary_file, 'rb') as reader:
        data = np.fromfile(reader, dtype=np.float32).reshape(nrows, ncols)
    
    arrays = [pyarrow.array(column) for column in data]
    names = [str(n) for n in range(nrows)]

    table = pyarrow.Table.from_arrays(arrays, names=names)
    
    # write Table to feather
    feather.write_feather(table, filename)

    return filename
