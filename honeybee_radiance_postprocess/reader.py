import numpy as np
import pyarrow.feather as feather


def read_feather(filepath, pandas_dataframe=True):
    """A very interesting description."""
    if not pandas_dataframe:
        table = feather.read_table(filepath)
        np_arrays = [arr.to_numpy() for arr in table]
        data = np.array(np_arrays)
    else:
        table = feather.read_feather(filepath)
        data = table.to_numpy().transpose()
    
    return data

