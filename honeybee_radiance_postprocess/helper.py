"""Helper functions."""
import json
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
        folder, extension, grid_areas=None, grids_info=None, name='grid_summary',
        grid_metrics=None
    ):
    """Calculate a grid summary for a single metric.

    Args:
        folder: A folder with results.
        extension: Extension of the files to collect data from.
        grid_areas: A list of area of each sensor.
        grids_info: Grid information as a dictionary.
        name: Optional filename of grid summary.
        grid_metrics: Additional customized metrics to calculate.
    """
    if grids_info is None:
        gi_file = folder.joinpath('grids_info.json')
        if not gi_file.exists():
            raise FileNotFoundError(
                f'The file grids_info.json was not found in the folder: {folder}.')
        with open(gi_file) as gi:
            grids_info = json.load(gi)

    if grid_areas is None:
        grid_areas = [None] * len(grids_info)

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

        if grid_metrics is not None:
            # get grid metrics
            grid_metrics_data = \
                _get_grid_metrics(array, grid_metrics, grid_info, grid_area)
            data.extend(grid_metrics_data)

        arrays.append(tuple(data))

    # create structured array
    struct_array = np.array(arrays, dtype=dtype)

    # write header to file
    with open(folder.joinpath(f'{name}.csv'), 'w') as grid_summary_file:
        grid_summary_file.write(','.join(header))
    # write structured array to grid_summary_file
    fmt = ['%s' , '%.2f', '%.2f', '%.2f', '%.2f']
    if grid_metrics is not None:
        fmt.extend(['%.2f' for _gr_m in grid_metrics])
    with open(folder.joinpath(f'{name}.csv'), 'a') as grid_summary_file:
        grid_summary_file.write('\n')
        np.savetxt(grid_summary_file, struct_array, delimiter=',', fmt=fmt)

    return grid_summary_file


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


def _numeric_type(array, gr_metric):
    if 'minimum' in gr_metric:
        gr_metric_bool = array > gr_metric['minimum']
    elif 'exclusiveMinimum' in gr_metric:
        gr_metric_bool = array >= gr_metric['minimum']
    elif 'maximum' in gr_metric:
        gr_metric_bool = array < gr_metric['maximum']
    elif 'exclusiveMaximum' in gr_metric:
        gr_metric_bool = array <= gr_metric['exclusiveMaximum']
    return gr_metric_bool


def _grid_summary_all_any(array, gr_metric, grid_info, grid_area, keyword):
    gr_metric_arrays = []
    for gr_m in gr_metric[keyword]:
        assert len(gr_m) == 1
        gr_metric_arrays.append(_numeric_type(array, gr_m))
    if keyword == 'allOf':
        gr_metric_bool = np.all(gr_metric_arrays, axis=0)
    else:
        gr_metric_bool = np.any(gr_metric_arrays, axis=0)
    gr_metric_pct = \
        _calculate_percentage(gr_metric_bool, grid_info, grid_area)
    return gr_metric_pct


def _get_grid_metrics(array, grid_metrics, grid_info, grid_area):
    grid_metrics_data = []
    for gr_metric in grid_metrics:
        if len(gr_metric) == 1:
            if 'allOf' in gr_metric:
                gr_metric_pct = \
                    _grid_summary_all_any(
                        array, gr_metric, grid_info, grid_area, 'allOf')
            elif 'anyOf' in gr_metric:
                gr_metric_pct = \
                    _grid_summary_all_any(
                        array, gr_metric, grid_info, grid_area, 'anyOf')
            else:
                gr_metric_bool = _numeric_type(array, gr_metric)
                gr_metric_pct = \
                    _calculate_percentage(gr_metric_bool, grid_info, grid_area)
        elif len(gr_metric) == 2:
            gr_metric_arrays = []
            for k, threshold in gr_metric.items():
                gr_metric_arrays.append(_numeric_type(array, {k: threshold}))
            gr_metric_bool = np.all(gr_metric_arrays, axis=0)
            gr_metric_pct = \
                _calculate_percentage(gr_metric_bool, grid_info, grid_area)
        grid_metrics_data.append(gr_metric_pct)
    return grid_metrics_data
