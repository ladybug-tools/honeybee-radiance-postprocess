"""Functions for LEED post-processing."""
from typing import Tuple, Union
from pathlib import Path
from collections import defaultdict
import json
import itertools
import numpy as np

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype.generic import GenericType
from ladybug.color import Colorset
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.fraction import Fraction
from ladybug.datatype.time import Time
from ladybug.legend import LegendParameters
from ladybug.header import Header
from honeybee.model import Model
from honeybee.units import conversion_factor_to_meters
from honeybee_radiance.writer import _filter_by_pattern
from honeybee_radiance.postprocess.annual import filter_schedule_by_hours
from .metrics import da_array2d, ase_array2d
from .annual import schedule_to_hoys, leed_occupancy_schedule
from .results import Results
from .util import filter_array, recursive_dict_merge


def _create_grid_summary(
    grid_name, sda_grid, ase_grid, pass_sda, pass_ase, total_floor,
    area_weighted=True):
    """Create a LEED summary for a single grid.

    Args:
        grid_name: Name of grid.
        sda_grid: Spatial Daylight Autonomy.
        ase_grid: Annual Sunlight Exposure.
        pass_sda: The percentage of the sensor points or floor area that
            passes sDA.
        pass_ase: The percentage of the sensor points or floor area that
            passes ASE.
        total_floor: The number of sensor points or floor area.
        area_weighted: Boolean to determine if the results are area
            weighted. Defaults to True.

    Returns:
        Tuple:
        -   summary_grid: Summary of each grid individually.
    """
    grid_summary = {
        grid_name: {}
    }
    if ase_grid > 10:
        ase_note = (
            'The Annual Sunlight Exposure is greater than 10% for space: '
            f'{grid_name}. Identify in writing how the space is designed to '
            'address glare.'
        )
        grid_summary[grid_name]['ase_note'] = ase_note

    if area_weighted:
        _grid_summary = {
            grid_name: {
                'ase': round(ase_grid, 2),
                'sda': round(sda_grid, 2),
                'floor_area_passing_ase': round(pass_ase, 2),
                'floor_area_passing_sda': round(pass_sda, 2),
                'total_floor_area': total_floor
            }
        }
    else:
        _grid_summary = {
            grid_name: {
                'ase': round(ase_grid, 2),
                'sda': round(sda_grid, 2),
                'sensor_count_passing_ase': int(round(pass_ase, 2)),
                'sensor_count_passing_sda': int(round(pass_sda, 2)),
                'total_sensor_count': total_floor
            }
        }

    recursive_dict_merge(grid_summary, _grid_summary)

    return grid_summary


def _leed_summary(
    pass_ase_grids: list, pass_sda_grids: list, grids_info: list,
    grid_areas: list) -> Tuple[dict, dict]:
    """_summary_

    Args:
        pass_ase_grids: A list where each sublist is a list of True/False that
            tells if each sensor point passes ASE.
        pass_sda_grids: A list where each sublist is a list of True/False that
            tells if each sensor point passes sDA.
        grids_info: A list of grid information.
        grid_areas: A list where each sublist is the area of each sensor point.
            The alternative is a list of None values for each grid information.

    Returns:
        Tuple:
        -   summary: Summary of of all grids combined.
        -   summary_grid: Summary of each grid individually.
    """
    summary = {}
    summary_grid = {}

    if all(grid_area is not None for grid_area in grid_areas):
        # weighted by mesh face area
        total_area = 0
        total_area_pass_ase = 0
        total_area_pass_sda = 0
        for (pass_ase, pass_sda, grid_area, grid_info) in \
            zip(pass_ase_grids, pass_sda_grids, grid_areas, grids_info):
            grid_name = grid_info['name']
            total_grid_area = grid_area.sum()
            area_pass_ase = grid_area[pass_ase].sum()
            area_pass_sda = grid_area[pass_sda].sum()
            ase_grid = (total_grid_area - area_pass_ase) / total_grid_area * 100
            sda_grid = area_pass_sda / total_grid_area * 100
            # grid summary
            grid_summary = \
                _create_grid_summary(
                    grid_name, sda_grid, ase_grid, area_pass_sda, area_pass_ase,
                    total_grid_area, area_weighted=True
                )

            recursive_dict_merge(summary_grid, grid_summary)

            total_area += total_grid_area
            total_area_pass_ase += area_pass_ase
            total_area_pass_sda += area_pass_sda

        summary['ase'] = round((total_area - total_area_pass_ase) / total_area * 100, 2)
        summary['sda'] = round(total_area_pass_sda / total_area * 100, 2)
        summary['floor_area_passing_ase'] = total_area_pass_ase
        summary['floor_area_passing_sda'] = total_area_pass_sda
        summary['total_floor_area'] = total_area
    else:
        # assume all sensor points cover the same area
        total_sensor_count = 0
        total_sensor_count_pass_ase = 0
        total_sensor_count_pass_sda = 0
        for (pass_ase, pass_sda, grid_info) in \
            zip(pass_ase_grids, pass_sda_grids, grids_info):
            grid_name = grid_info['name']
            grid_count = grid_info['count']
            sensor_count_pass_ase = pass_ase.sum()
            sensor_count_pass_sda = pass_sda.sum()
            ase_grid = (grid_count - sensor_count_pass_ase) / grid_count * 100
            sda_grid = sensor_count_pass_sda / grid_count * 100
            # grid summary
            grid_summary = \
                _create_grid_summary(
                    grid_name, sda_grid, ase_grid, sensor_count_pass_sda,
                    sensor_count_pass_ase, grid_count, area_weighted=False
                )

            recursive_dict_merge(summary_grid, grid_summary)

            total_sensor_count += grid_count
            total_sensor_count_pass_ase += sensor_count_pass_ase
            total_sensor_count_pass_sda += sensor_count_pass_sda

        summary['ase'] = round((total_sensor_count - total_sensor_count_pass_ase) /
            total_sensor_count * 100, 2
        )
        summary['sda'] = round(total_sensor_count_pass_sda / total_sensor_count * 100, 2)
        summary['sensor_count_passing_ase'] = int(total_sensor_count_pass_ase)
        summary['sensor_count_passing_sda'] = int(total_sensor_count_pass_sda)
        summary['total_sensor_count'] = total_sensor_count

    return summary, summary_grid


def _ase_hourly_percentage(
    results: Results, array: np.ndarray, grid_info: dict,
    direct_threshold: float = 1000, grid_area: Union[None, np.ndarray] = None
    ) -> np.ndarray:
    """Calculate the percentage of floor area that receives greater than 1000
    direct lux for each hour.

    Args:
        results: A Results object.
        array: A NumPy array of the grid to process.
        grid_info: Grid information of the grid to process..
        direct_threshold: Direct threshold.
        grid_area: Grid area as a NumPy array with a area value for each sensor
            point, or a None value if there is no area associated with the
            sensor point.

    Returns:
        A hourly data collection of the percentage of floor area that receives
        greater than 1000 direct lux.
    """
    if grid_area is not None:
        grid_area_2d = np.array([grid_area] * array.shape[1]).transpose()
        area_above = \
            np.where((array > direct_threshold), grid_area_2d, 0).sum(axis=0)
        percentage_above = area_above / grid_area.sum() * 100
    else:
        percentage_above = \
            (array > direct_threshold).sum(axis=0) / grid_info['count'] * 100

    occupancy_hoys = schedule_to_hoys(results.schedule, results.sun_up_hours)
    # map states to 8760 values
    percentage_above = results.values_to_annual(
        occupancy_hoys, percentage_above, results.timestep)
    header = Header(Fraction('Percentage above 1000 direct lux'), '%',
                    AnalysisPeriod(results.timestep),
                    metadata={'SensorGrid': grid_info['name']})
    data_collection = HourlyContinuousCollection(header, percentage_above.tolist())

    return data_collection

def shade_transmittance_per_light_path(
    light_paths: list, shade_transmittance: Union[float, dict],
    shd_trans_dict: dict) -> dict:
    """Filter shade_transmittance by light paths and add default multiplier.

    Args:
        light_paths: A list of light paths.
        shade_transmittance: A value to use as a multiplier in place of solar
            shading. This input can be either a single value that will be used
            for all aperture groups, or a dictionary where aperture groups are
            keys, and the value for each key is the shade transmittance. Values
            for shade transmittance must be 1 > value > 0.
        shd_trans_dict: A dictionary used to store shade transmittance value
            for each aperture group.

    Returns:
        A dictionary with filtered light paths.
    """
    shade_transmittances = {}
    if isinstance(shade_transmittance, dict):
        for light_path in light_paths:
            # default multiplier
            shade_transmittances[light_path] = [1]
            # add custom shade transmittance
            if light_path in shade_transmittance:
                shade_transmittances[light_path].append(
                    shade_transmittance[light_path])
                shd_trans_dict[light_path] = shade_transmittance[light_path]
            # add default shade transmittance (0.05)
            elif light_path != '__static_apertures__':
                shade_transmittances[light_path].append(0.05)
                shd_trans_dict[light_path] = 0.05
    else:
        shd_trans = float(shade_transmittance)
        for light_path in light_paths:
            # default multiplier
            shade_transmittances[light_path] = [1]
            # add custom shade transmittance
            if light_path != '__static_apertures__':
                shade_transmittances[light_path].append(shd_trans)
                shd_trans_dict[light_path] = shd_trans

    return shade_transmittances


def leed_states_schedule(
        results: Union[str, Results], grids_filter: str = '*',
        shade_transmittance: Union[float, dict] = 0.05
        ) -> Tuple[dict, dict]:
    """Calculate a schedule of each aperture group for LEED compliant sDA.

    This function calculates an annual shading schedule of each aperture
    group. Hour by hour it will select the least shaded aperture group
    configuration, so that no more than 2% of the sensors points receive
    direct illuminance of 1000 lux or more.

    Args:
        results: Path to results folder or a Results class object.
        grids_filter: The name of a grid or a pattern to filter the grids.
            Defaults to '*'.
        shade_transmittance: A value to use as a multiplier in place of solar
            shading. This input can be either a single value that will be used
            for all aperture groups, or a dictionary where aperture groups are
            keys, and the value for each key is the shade transmittance. Values
            for shade transmittance must be 1 > value > 0.
            Defaults to 0.05.

    Returns:
        Tuple: A tuple with a dictionary of the annual schedule and a
            dictionary of hours where no shading configuration comply with the
            2% rule.
    """
    if not isinstance(results, Results):
        results = Results(results)

    grids_info = results._filter_grids(grids_filter=grids_filter)
    schedule = leed_occupancy_schedule(as_list=True)
    occ_pattern = \
        filter_schedule_by_hours(results.sun_up_hours, schedule=schedule)[0]
    occ_mask = np.array(occ_pattern)

    states_schedule = defaultdict(list)
    fail_to_comply = {}
    shd_trans_dict = {}

    for grid_info in grids_info:
        grid_id = grid_info['full_id']
        grid_count = grid_info['count']
        light_paths = [lp[0] for lp in grid_info['light_path']]

        shade_transmittances = shade_transmittance_per_light_path(
            light_paths, shade_transmittance, shd_trans_dict)

        if len(light_paths) > 6:
            full_direct = []
            full_thresh = []
            shd_trans_array = []
            for light_path in light_paths:
                array = results._get_array(grid_info, light_path, res_type='direct')
                array = np.apply_along_axis(filter_array, 1, array, occ_mask)
                full_direct.append(array)
                full_thresh.append((array >= 1000).sum(axis=0))
                shd_trans_array.append(shade_transmittances[light_path][1])

            # sum the array element-wise
            full_direct_sum = sum(full_direct)

            # create base list of shading combinations (all set to 1)
            combinations = [
                {light_path: 1 for light_path in light_paths} \
                    for i in range(full_direct_sum.shape[1])
            ]

            # find the percentage of floor area >= 1000 lux
            direct_pct_above = (full_direct_sum >= 1000).sum(axis=0) / grid_count

            # find the indices where the percentage of floor area is > 2%
            above_2_indices = np.where(direct_pct_above > 0.02)[0]

            # get an array of only the relevant hours
            direct_sum = np.take(full_direct_sum, above_2_indices, axis=1)

            # get an array of only the relevant hours
            direct = np.take(full_direct, above_2_indices, axis=2)

            # get an array of only the relevant hours
            thresh = np.take(full_thresh, above_2_indices, axis=1)

            # sort and get indices. Negate the array to get descending order
            sort_thresh = np.argsort(-thresh, axis=0).transpose()

            _combinations = []
            _combinations.insert(0, (np.arange(full_direct_sum.shape[1]), combinations))

            if np.any(above_2_indices):
                for idx in range(len(full_direct)):
                    # take column
                    sort_indices = np.take(sort_thresh, idx, axis=1)

                    # map light path identifiers
                    light_path_ids = np.take(light_paths, sort_indices)

                    # get a list of shade transmittances
                    shd_trans_array = np.take(shd_trans_array, sort_indices)

                    # create combination for the subset
                    _subset_combination = [
                        {light_path: _shd_trans} for light_path, _shd_trans in \
                            zip(light_path_ids, shd_trans_array)
                    ]
                    _combinations.insert(0, (above_2_indices, _subset_combination))

                    # take the values from each array by indexing
                    direct_array = \
                        direct[sort_indices, :, range(len(sort_indices))].transpose()

                    # subtract the illuminance values
                    direct_sum = direct_sum - (direct_array * (1 - shd_trans_array))

                    # find the percentage of floor area >= 1000 lux
                    direct_pct_above = (direct_sum >= 1000).sum(axis=0) / grid_count

                    # find the indices where the percentage of floor area is > 2%
                    above_2_indices = np.where(direct_pct_above > 0.02)[0]

                    # break if there are no hours above 2%
                    if not np.any(above_2_indices):
                        break

                    # update variables for the next iteration
                    direct_sum = np.take(direct_sum, above_2_indices, axis=1)
                    direct = np.take(direct, above_2_indices, axis=2)
                    thresh = np.take(thresh, above_2_indices, axis=1)
                    sort_thresh = np.take(sort_thresh, above_2_indices, axis=0)
                    shd_trans_array = np.take(shd_trans_array, above_2_indices)

                if np.any(above_2_indices):
                    # take column
                    sort_indices = np.take(sort_thresh, idx, axis=1)

                    # map light path identifiers
                    light_path_ids = np.take(light_paths, sort_indices)

                    # get a list of shade transmittances
                    shd_trans_array = np.take(shd_trans_array, sort_indices)

                    # create combination for the subset
                    _subset_combination = [
                        {light_path: _shd_trans} for light_path, _shd_trans in \
                            zip(light_path_ids, shd_trans_array)
                    ]
                    _combinations.insert(0, (above_2_indices, _subset_combination))

                    # there are hours not complying with the 2% rule
                    previous_indices = []
                    previous_combination = []
                    grid_comply = []
                    # merge the combinations from the iterations of the subsets
                    for i, subset in enumerate(_combinations):
                        if i == 0:
                            previous_indices = subset[0]
                        else:
                            _indices = subset[0]
                            grid_comply = []
                            for _pr_idx in previous_indices:
                                grid_comply.append(_indices[_pr_idx])
                            previous_indices = grid_comply
                    # convert indices to sun up hours indices
                    filter_indices = np.where(occ_mask.astype(bool))[0]
                    grid_comply = [filter_indices[_gc] for _gc in grid_comply]
                    grid_comply = np.array(results.sun_up_hours)[grid_comply]
                    fail_to_comply[grid_id] = \
                        [int(hoy) for hoy in grid_comply]

                previous_indices = None
                previous_combination = None
                # merge the combinations from the iterations of the subsets
                for i, subset in enumerate(_combinations):
                    if i == 0:
                        previous_indices, previous_combination = subset
                    else:
                        _indices, _combination = subset
                        for _pr_idx, _pr_comb in \
                            zip(previous_indices, previous_combination):
                            for light_path, _shd_trans in _pr_comb.items():
                                _combination[_pr_idx][light_path] = _shd_trans
                        previous_indices = _indices
                        previous_combination = _combination

                combinations = _combination

            # merge the combinations of dicts
            for combination in combinations:
                for light_path, shd_trans in combination.items():
                    if light_path != '__static_apertures__':
                        states_schedule[light_path].append(shd_trans)

        else:
            keys, values = zip(*shade_transmittances.items())
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            array_list_combinations = []
            for combination in combinations:
                combination_arrays = []
                for light_path, shd_trans in combination.items():
                    array = results._get_array(
                        grid_info, light_path, res_type='direct')
                    if shd_trans == 1:
                        combination_arrays.append(array)
                    else:
                        combination_arrays.append(array * shd_trans)
                combination_array = sum(combination_arrays)
                combination_percentage = \
                    (combination_array >= 1000).sum(axis=0) / grid_count
                array_list_combinations.append(combination_percentage)
            array_combinations = np.array(array_list_combinations)
            array_combinations[array_combinations > 0.02] = np.NINF

            grid_comply = np.where(np.all(array_combinations==np.NINF, axis=0))[0]
            if grid_comply.size != 0:
                grid_comply = np.array(results.sun_up_hours)[grid_comply]
                fail_to_comply[grid_id] = \
                    [int(hoy) for hoy in grid_comply]

            array_combinations_filter = \
                np.apply_along_axis(filter_array, 1, array_combinations, occ_mask)
            max_indices = array_combinations_filter.argmax(axis=0)
            # select the combination for each hour
            combinations = [combinations[idx] for idx in max_indices]
            # merge the combinations of dicts
            for combination in combinations:
                for light_path, shd_trans in combination.items():
                    if light_path != '__static_apertures__':
                        states_schedule[light_path].append(shd_trans)

    occupancy_hoys = schedule_to_hoys(schedule, results.sun_up_hours)

    # map states to 8760 values
    for light_path, shd_trans in states_schedule.items():
        mapped_states = results.values_to_annual(
            occupancy_hoys, shd_trans, results.timestep)
        states_schedule[light_path] = mapped_states

    return states_schedule, fail_to_comply, shd_trans_dict


def leed_option_one(
        results: Union[str, Results], grids_filter: str = '*',
        shade_transmittance: Union[float, dict] = 0.05,
        states_schedule: dict = None, threshold: float = 300,
        direct_threshold: float = 1000, occ_hours: int = 250,
        target_time: float = 50, sub_folder: str = None):
    """Calculate credits for LEED v4.1 Daylight Option 1.

    Args:
        results: Path to results folder or a Results class object.
        grids_filter: The name of a grid or a pattern to filter the grids.
            Defaults to '*'.
        shade_transmittance: A value to use as a multiplier in place of solar
            shading. This input can be either a single value that will be used
            for all aperture groups, or a dictionary where aperture groups are
            keys, and the value for each key is the shade transmittance. Values
            for shade transmittance must be 1 > value > 0.
            Defaults to 0.05.
        states_schedule: A custom dictionary of shading states. In case this is
            left empty, the function will calculate a shading schedule by using
            the shade_transmittance input. If a states schedule is provided it
            will check that it is complying with the 2% rule. Defaults to None.
        threshold: Threshold value for daylight autonomy. Default: 300.
        direct_threshold: The threshold that determines if a sensor is overlit.
            Defaults to 1000.
        occ_hours: The number of occupied hours that cannot receive more than
            the direct_threshold. Defaults to 250.
        target_time: A minimum threshold of occupied time (eg. 50% of the
            time), above which a given sensor passes and contributes to the
            spatial daylight autonomy. Defaults to 50.
        sub_folder: Relative path for a subfolder to write the output. If None,
            the files will not be written. Defaults to None.

    Returns:
        Tuple:
        -   summary: Summary of all grids combined.
        -   summary_grid: Summary of each grid individually.
        -   da_grids: List of daylight autonomy values for each grid. Each item
                in the list is a NumPy array of DA values.
        -   hours_above: List of hours above 1000 direct illuminance (with
                default states) for each grid. Each item in the list is a NumPy
                array of hours above 1000 lux.
        -   states_schedule: A dictionary of annual shading schedules for each
                aperture group.
        -   grids_info: Grid information.
    """
    # use default leed occupancy schedule
    schedule = leed_occupancy_schedule(as_list=True)

    if not isinstance(results, Results):
        results = Results(results, schedule=schedule)
    else:
        # set schedule to default leed schedule
        results.schedule = schedule

    occ_mask = results.occ_mask
    total_occ = results.total_occ

    grids_info = results._filter_grids(grids_filter=grids_filter)

    if not states_schedule:
        states_schedule, fail_to_comply, shd_trans_dict = \
            leed_states_schedule(results, grids_filter=grids_filter,
                shade_transmittance=shade_transmittance)
    else:
        raise NotImplementedError(
            'Custom input for argument states_schedule is not yet implemented.'
            )

    # check to see if there is a HBJSON with sensor grid meshes for areas
    grid_areas, units_conversion = [], 1
    for base_file in Path(results.folder).parent.iterdir():
        if base_file.suffix in ('.hbjson', '.hbpkl'):
            hb_model = Model.from_file(base_file)
            units_conversion = conversion_factor_to_meters(hb_model.units)
            filt_grids = _filter_by_pattern(
                hb_model.properties.radiance.sensor_grids, filter=grids_filter)
            for s_grid in filt_grids:
                if s_grid.mesh is not None:
                    grid_areas.append(s_grid.mesh.face_areas)
            grid_areas = [np.array(grid) for grid in grid_areas]
    if not grid_areas:
        grid_areas = [None] * len(grids_info)

    # annual sunlight exposure
    ase_grids = []
    hours_above = []
    pass_ase_grids = []
    ase_hr_pct = []
    for (grid_info, grid_area) in zip(grids_info, grid_areas):
        light_paths = [lp[0] for lp in grid_info['light_path']]
        arrays = []
        # combine direct array for all light paths
        for light_path in light_paths:
            array = results._get_array(
                grid_info, light_path, res_type='direct')
            array_filter = np.apply_along_axis(
                filter_array, 1, array, occ_mask)
            arrays.append(array_filter)
        array = sum(arrays)
        # calculate ase per grid
        ase_grid, h_above = ase_array2d(
            array, occ_hours=occ_hours, direct_threshold=direct_threshold)

        # calculate the number of sensor points above 1000 lux for each hour
        ase_hr_pct.append(
            _ase_hourly_percentage(
                results, array, grid_info, direct_threshold=direct_threshold,
                grid_area=grid_area
            )
        )

        ase_grids.append(ase_grid)
        hours_above.append(h_above)
        pass_ase = h_above < occ_hours
        pass_ase_grids.append(pass_ase)

    # spatial daylight autonomy
    da_grids = []
    pass_sda_grids = []
    for grid_info in grids_info:
        light_paths = [lp[0] for lp in grid_info['light_path']]
        arrays = []
        # combine total array for all light paths
        for light_path in light_paths:
            array = results._get_array(grid_info, light_path, res_type='total')
            array_filter = np.apply_along_axis(
                filter_array, 1, array, occ_mask)
            if light_path != '__static_apertures__':
                sun_up_hours = np.array(results.sun_up_hours).astype(int)
                shd_trans_array = states_schedule[light_path][sun_up_hours]
                shd_trans_array = shd_trans_array[occ_mask.astype(bool)]
                arrays.append(array_filter * shd_trans_array)
            else:
                arrays.append(array_filter)
        array = sum(arrays)
        # calculate da per grid
        da_grid = da_array2d(array, total_occ=total_occ, threshold=threshold)
        da_grids.append(da_grid)
        # calculate sda per grid
        pass_sda = da_grid >= target_time
        pass_sda_grids.append(pass_sda)

    # create summaries for all grids and each grid individually
    summary, summary_grid = _leed_summary(
        pass_ase_grids, pass_sda_grids, grids_info, grid_areas=grid_areas)

    # credits
    if not fail_to_comply:
        if summary['sda'] >= 75:
            summary['credits'] = 3
        elif summary['sda'] >= 55:
            summary['credits'] = 2
        elif summary['sda'] >= 40:
            summary['credits'] = 1
        else:
            summary['credits'] = 0

        if all(grid_summary['sda'] >= 55 for grid_summary in summary_grid.values()):
            if summary['credits'] <= 2:
                summary['credits'] += 1
            else:
                summary['credits'] = 'Exemplary performance'
    else:
        summary['credits'] = 0
        fail_to_comply_rooms = ', '.join(list(fail_to_comply.keys()))
        note = (
            '0 credits have been awarded. The following sensor grids have at '
            'least one hour where 2% of the floor area receives direct '
            f'illuminance of 1000 lux or more: {fail_to_comply_rooms}.'
        )
        summary['note'] = note

    # convert to datacollection
    def to_datacollection(aperture_group: str, values: np.ndarray):
        # convert values to 0 and 1 (0 = no shading, 1 = shading)
        values[values == 1] = 0
        values[values == shd_trans_dict[aperture_group]] = 1
        header = Header(data_type=GenericType(aperture_group, ''), unit='',
                        analysis_period=AnalysisPeriod(),
                        metadata={'Shade Transmittance': shd_trans_dict[aperture_group]})
        hourly_data = HourlyContinuousCollection(header=header, values=values.tolist())
        return hourly_data.to_dict()

    states_schedule = {k:to_datacollection(k, v) for k, v in states_schedule.items()}

    if sub_folder:
        folder = Path(sub_folder)
        folder.mkdir(parents=True, exist_ok=True)

        summary_file = folder.joinpath('summary.json')
        summary_file.write_text(json.dumps(summary, indent=2))
        summary_grid_file = folder.joinpath('summary_grid.json')
        summary_grid_file.write_text(json.dumps(summary_grid, indent=2))
        states_schedule_file = folder.joinpath('states_schedule.json')
        states_schedule_file.write_text(json.dumps(states_schedule))
        grids_info_file = folder.joinpath('grids_info.json')
        grids_info_file.write_text(json.dumps(grids_info, indent=2))

        for (da, h_above, ase_hr_p, grid_info) in \
            zip(da_grids, hours_above, ase_hr_pct, grids_info):
            grid_id = grid_info['full_id']
            da_file = folder.joinpath('results', 'da', f'{grid_id}.da')
            da_file.parent.mkdir(parents=True, exist_ok=True)
            hours_above_file = folder.joinpath(
                'results', 'ase_hours_above', f'{grid_id}.res')
            hours_above_file.parent.mkdir(parents=True, exist_ok=True)
            ase_hr_p_file = folder.joinpath(
                'datacollections', 'ase_percentage_above', f'{grid_id}.json')
            ase_hr_p_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(da_file, da, fmt='%.2f')
            np.savetxt(hours_above_file, h_above, fmt='%.0f')
            ase_hr_p_file.write_text(json.dumps(ase_hr_p.to_dict()))

        da_grids_info_file = folder.joinpath(
            'results', 'da', 'grids_info.json')
        da_grids_info_file.write_text(json.dumps(grids_info, indent=2))
        ase_grids_info_file = folder.joinpath(
            'results', 'ase_hours_above', 'grids_info.json')
        ase_grids_info_file.write_text(json.dumps(grids_info, indent=2))
        ase_hr_pct_info_file = folder.joinpath(
            'datacollections', 'ase_percentage_above', 'grids_info.json')
        ase_hr_pct_info_file.write_text(json.dumps(grids_info, indent=2))

        states_schedule_err_file = \
            folder.joinpath('states_schedule_err.json')
        states_schedule_err_file.write_text(json.dumps(fail_to_comply))

    return (summary, summary_grid, da_grids, hours_above, states_schedule,
            fail_to_comply, grids_info)


def _leed_daylight_option_one_vis_metadata():
    """Return visualization metadata for leed daylight option one."""
    da_lpar = LegendParameters(min=0, max=100, colors=Colorset.annual_comfort())
    ase_hrs_lpar = LegendParameters(min=0, max=250, colors=Colorset.original())

    metric_info_dict = {
        'da': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'ase_hours_above': {
            'type': 'VisualizationMetaData',
            'data_type': Time('Hours above direct threshold').to_dict(),
            'unit': 'hr',
            'legend_parameters': ase_hrs_lpar.to_dict()
        }
    }

    return metric_info_dict
