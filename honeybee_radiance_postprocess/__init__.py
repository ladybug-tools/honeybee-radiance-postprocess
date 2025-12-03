"""honeybee-radiance-postprocess library."""
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
    msg = (f'Using CuPy ({np.__version__}) for GPU ({device_name}) acceleration in '
           'honeybee-radiance-postprocess.')
    print(msg)
except ModuleNotFoundError:
    import numpy as np
    IS_GPU = False
except Exception as e:
    import numpy as np
    IS_GPU = False
    msg = (f'Failed to load CuPy successfully. '
           'Help: https://github.com/ladybug-tools/honeybee-radiance-postprocess/blob/master/CUPY.md. '
           f'Falling back to NumPy ({np.__version__}) in honeybee-radiance-postprocess: {e}')
    print(msg)

IS_CPU = not IS_GPU
