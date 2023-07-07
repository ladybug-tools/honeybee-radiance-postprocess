"""Helper functions."""
import numpy as np

from honeybee.model import Model


def model_grid_areas(model, grids_info):
    if isinstance(model, Model):
        hb_model = model
    else:
        hb_model = Model.from_file(model)

    full_ids = [grid_info['full_id'] for grid_info in grids_info]
    sensor_grids = hb_model.properties.radiance.sensor_grids
    grid_areas = []
    for s_grid in sensor_grids:
        if s_grid.identifier in full_ids:
            if s_grid.mesh is not None:
                grid_areas.append(s_grid.mesh.face_areas)
    grid_areas = [np.array(grid) for grid in grid_areas]
    if not grid_areas:
        grid_areas = [None] * len(full_ids)

    return grid_areas


def grid_summary(
        folder, extension, grids_info, grid_areas, name='grid_summary',
        grid_metrics=None
    ):
    """Calculate a grid summary for a single metric.

    Args:
        folder: A folder with results.
        extension: Extension of the files to collect data from.
        grids_info: Grid information as a dictionary.
        grid_areas:
        name: Optional filename of grid summary.
        grid_metrics: Additional customized metrics to calculate.
    """
    # set up the default data types
    dtype = [
        ('Sensor Grid', 'O'),
        ('Mean', np.float32),
        ('Minimum', np.float32),
        ('Maximum', np.float32),
        ('Uniformity Ratio', np.float32)
    ]

    if grid_metrics is not None:
        for gr_m in grid_metrics:
            gr_m_h = []
            for k, v in gr_m.items():
                gr_m_h.append(k + str(v))
            dtype.append((' '.join(gr_m_h), np.float32))

    header = [dt[0] for dt in dtype]

    arrays = []
    for grid_info, grid_area in zip(grids_info, grid_areas):
        full_id = grid_info['full_id']
        array = np.loadtxt(folder.joinpath(f'{full_id}.{extension}'))
        _mean = array.mean()
        _min = array.min()
        _max = array.max()
        _uniformity_ratio = _min / _mean * 100

        data = [full_id, _mean, _min, _max, _uniformity_ratio]

        grid_metrics_data = []
        if grid_metrics is not None:
            for gr_metric in grid_metrics:
                if len(gr_metric) == 1:
                    for k, v in gr_metric.items():
                        if k == 'anyOff' or k == 'allOff':
                            gr_metric_arrays = []
                            for vv in v:
                                for kk, threshold in vv.items():
                                    if kk == 'minimum':
                                        gr_metric_arrays.append(array > threshold)
                                    elif kk == 'exclusiveMinimum':
                                        gr_metric_arrays.append(array >= threshold)
                                    elif kk == 'maximum':
                                        gr_metric_arrays.append(array < threshold)
                                    elif kk == 'exclusiveMaximum':
                                        gr_metric_arrays.append(array <= threshold)
                            if k == 'anyOff':
                                gr_metric_bool = np.any(gr_metric_arrays, axis=0)
                            else:
                                gr_metric_bool = np.all(gr_metric_arrays, axis=0)
                            gr_metric_pct = \
                                _calculate_percentage(gr_metric_bool, grid_info, grid_area)
                        else:
                            threshold = v
                            if k == 'minimum':
                                gr_metric_bool = array > threshold
                            elif k == 'exclusiveMinimum':
                                gr_metric_bool = array >= threshold
                            elif k == 'maximum':
                                gr_metric_bool = array < threshold
                            elif k == 'exclusiveMaximum':
                                gr_metric_bool = array <= threshold
                            gr_metric_pct = \
                                _calculate_percentage(gr_metric_bool, grid_info, grid_area)
                elif len(gr_metric) == 2:
                    gr_metric_arrays = []
                    for k, threshold in gr_metric.items():
                        if k == 'minimum':
                            gr_metric_arrays.append(array > threshold)
                        elif k == 'exclusiveMinimum':
                            gr_metric_arrays.append(array >= threshold)
                        elif k == 'maximum':
                            gr_metric_arrays.append(array < threshold)
                        elif k == 'exclusiveMaximum':
                            gr_metric_arrays.append(array <= threshold)
                    gr_metric_bool = np.all(gr_metric_arrays, axis=0)
                    gr_metric_pct = \
                        _calculate_percentage(gr_metric_bool, grid_info, grid_area)
                grid_metrics_data.append(gr_metric_pct)

        data.extend(grid_metrics_data)

        arrays.append(tuple(data))

    # create structured array
    struct_array = np.array(arrays, dtype=dtype)

    # write header to file
    with open(folder.joinpath(f'{name}.csv'), 'w') as file:
        file.write(','.join(header))
    # write structured array to file
    fmt = ['%s' , '%.2f', '%.2f', '%.2f', '%.2f']
    if grid_metrics is not None:
        fmt.extend(['%.2f' for d in grid_metrics])
    with open(folder.joinpath(f'{name}.csv'), 'a') as file:
        file.write('\n')
        np.savetxt(file, struct_array, delimiter=',', fmt=fmt)

    return file


def _calculate_percentage(gr_metric_bool, grid_info, grid_area=None):
    """Calculate percentage of floor area where True.

    Args:
        gr_metric_bool: A NumPy array of booleans.
        grid_info: Grid information.
        grid_area: A NumPy array of area for each sensor. (Default: None).
    
    Returns:
        The percentage of floor area where gr_metric_bool is True.
    """
    if grid_area is not None:
        gr_metric_pct = \
            grid_area[gr_metric_bool].sum() / grid_area.sum() * 100
    else:
        gr_metric_pct = \
            gr_metric_bool.sum() / grid_info['count'] * 100
    return gr_metric_pct
