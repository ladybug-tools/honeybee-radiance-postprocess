"""honeybee-radiance-postprocess library."""
import sys

IS_GPU = False

if sys.version_info >= (3, 0):
    try:
        import cupy as np
        np.arange(1)
        IS_GPU = True
        try:
            device_id = np.cuda.runtime.getDevice()
            device_props = np.cuda.runtime.getDeviceProperties(device_id)
            device_name = device_props['name'].decode()
        except Exception:
            device_name = 'Unknown GPU'

        msg = (
            'Using CuPy ({}) for GPU ({}) acceleration in '
            'honeybee-radiance-postprocess.'
        ).format(np.__version__, device_name)
        sys.stderr.write(msg)

    except ModuleNotFoundError:
        import numpy as np

    except Exception as e:
        import numpy as np
        msg = (
            'Failed to load CuPy successfully. '
            'Help: https://github.com/ladybug-tools/honeybee-radiance-postprocess/blob/master/CUPY.md. '
            'Falling back to NumPy ({}) in honeybee-radiance-postprocess: {}'
        ).format(np.__version__, e)
        sys.stderr.write(msg)

IS_CPU = not IS_GPU
