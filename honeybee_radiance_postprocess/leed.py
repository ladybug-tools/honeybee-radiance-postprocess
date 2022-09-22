"""Functions for LEED post-processing."""
from typing import Tuple, Union
from pathlib import Path
from collections import defaultdict
import json
import itertools
import warnings
import numpy as np

from honeybee.model import Model
from honeybee.units import conversion_factor_to_meters
from honeybee_radiance.writer import _filter_by_pattern
from honeybee_radiance.postprocess.annual import filter_schedule_by_hours
from .metrics import da_array2d, ase_array2d
from .annual import schedule_to_hoys, leed_occupancy_schedule
from .results import Results
from .util import filter_array, recursive_dict_merge


def _leed_summary(
    pass_ase_grids: list, pass_sda_grids: list, grid_areas: list,
    grids_info: list) -> Tuple[dict, dict]:
    summary = {}
    summary_grid = {}

    if grid_areas:
        # weighted by mesh face area
        total_area = total_area_pass_ase = total_area_pass_sda = 0
        for (pass_ase, pass_sda, grid_area, grid_info) in \
            zip(pass_ase_grids, pass_sda_grids, grid_areas, grids_info):
            grid_id = grid_info['full_id']
            total_grid_area = grid_area.sum()
            area_pass_ase = grid_area[pass_ase].sum()
            area_pass_sda = grid_area[pass_sda].sum()
            # grid summary
            grid_summary = {
                grid_id: {
                    'ase': area_pass_ase / total_grid_area,
                    'sda': area_pass_sda / total_grid_area,
                    'floor_area_passing_ase': area_pass_ase,
                    'floor_area_passing_sda': area_pass_sda,
                    'total_floor_area': total_grid_area
                }
            }
            recursive_dict_merge(summary_grid, grid_summary)

            total_area +=total_grid_area
            total_area_pass_ase += area_pass_ase
            total_area_pass_sda += area_pass_sda

        summary['ase'] = total_area_pass_ase / total_area
        summary['sda'] = total_area_pass_sda / total_area
        summary['floor_area_passing_ase'] = total_area_pass_ase
        summary['floor_area_passing_sda'] = total_area_pass_sda
        summary['total_floor_area'] = total_area
    else:
        # assume all sensor points cover the same area
        total_sensor_count = total_sensor_count_pass_ase = \
            total_sensor_count_pass_sda = 0
        for (pass_ase, pass_sda, grid_info) in \
            zip(pass_ase_grids, pass_sda_grids, grids_info):
            grid_id = grid_info['full_id']
            grid_count = grid_info['count']
            sensor_count_pass_ase = pass_ase.sum()
            sensor_count_pass_sda = pass_sda.sum()
            # grid summary
            grid_summary = {
                grid_id: {
                    'ase': sensor_count_pass_ase / grid_count,
                    'sda': sensor_count_pass_sda / grid_count,
                    'sensor_count_passing_ase': int(sensor_count_pass_ase),
                    'sensor_count_pass_sda': int(sensor_count_pass_sda),
                    'total_sensor_count': grid_count
                }
            }
            recursive_dict_merge(summary_grid, grid_summary)

            total_sensor_count += grid_count
            total_sensor_count_pass_ase += sensor_count_pass_ase
            total_sensor_count_pass_sda += sensor_count_pass_sda

        summary['ase'] = total_sensor_count_pass_ase / total_sensor_count
        summary['sda'] = total_sensor_count_pass_sda / total_sensor_count
        summary['sensor_count_passing_ase'] = int(total_sensor_count_pass_ase)
        summary['sensor_count_passing_sda'] = int(total_sensor_count_pass_sda)
        summary['total_sensor_count'] = total_sensor_count

    return summary, summary_grid

def shade_transmittance_per_light_path(
    light_paths: list, shade_transmittance: Union[float, dict]) -> dict:
    """Filter shade_transmittance by light paths and add default multiplier.

    Args:
        light_paths: A list of light paths.
        shade_transmittance: A value to use as a multiplier in place of solar
            shading. This input can be either a single value that will be used
            for all aperture groups, or a dictionary where aperture groups are
            keys, and the value for each key is the shade transmittance. Values
            for shade transmittance must be 1 > value > 0.

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
            # add default shade transmittance (0.2)
            elif light_path != '__static_apertures__':
                shade_transmittances[light_path].append(0.2)
    else:
        shade_transmittance = float(shade_transmittance)
        for light_path in light_paths:
            # default multiplier
            shade_transmittances[light_path] = [1]
            # add custom shade transmittance
            if light_path != '__static_apertures__':
                shade_transmittances[light_path].append(shade_transmittance)

    return shade_transmittances


def leed_states_schedule(
        results: Union[str, Results], grids_filter: str = '*',
        shade_transmittance: Union[float, dict] = 0.2
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
            Defaults to 0.2.

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

    for grid_info in grids_info:
        grid_id = grid_info['full_id']
        grid_count = grid_info['count']
        light_paths = [lp[0] for lp in grid_info['light_path']]
        shade_transmittances = shade_transmittance_per_light_path(
            light_paths, shade_transmittance)
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
            fail_to_comply[grid_id] = \
                [results.datetimes[hoy] for hoy in grid_comply]

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

    return states_schedule, fail_to_comply


def leed_option_1(
        results: Union[str, Results], grids_filter: str = '*',
        shade_transmittance: Union[float, dict] = 0.2,
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
            Defaults to 0.2.
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
        states_schedule, fail_to_comply = leed_states_schedule(
            results, grids_filter=grids_filter,
            shade_transmittance=shade_transmittance
            )
    else:
        raise NotImplementedError(
            'Custom input for argument states_schedule is not yet implemented.'
            )

    if fail_to_comply:
        warning_msg = (
            'For some hours it is not possible to keep the percentage of '
            'sensors that receive more than 1000 direct illuminance at 2% or '
            'below. We should do something about this, but for now I just let '
            'it continue.'
            )
        warnings.warn(warning_msg)

    # check to see if there is a HBJSON with sensor grid meshes for areas
    grid_areas, units_conversion = [], 1
    for base_file in Path(results.folder).parent.iterdir():
        if base_file.suffix in ('.hbjson', '.hbpkl') :
            hb_model = Model.from_file(base_file)
            units_conversion = conversion_factor_to_meters(hb_model.units)
            filt_grids = _filter_by_pattern(
                hb_model.properties.radiance.sensor_grids, filter=grids_filter)
            for s_grid in filt_grids:
                if s_grid.mesh is not None:
                    grid_areas.append(s_grid.mesh.face_areas)
            grid_areas = [np.array(grid) for grid in grid_areas]

    # annual sunlight exposure
    ase_grids = []
    hours_above = []
    pass_ase_grids = []
    for grid_info in grids_info:
        grid_id = grid_info['full_id']
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
        if ase_grid > 0.10:
            warning_msg = (
                'The Annual Sunlight Exposure is greater than 10% for grid: '
                f'{grid_id}. Identify in writing how the space is designed to '
                'address glare.'
                )
            warnings.warn(warning_msg)
        ase_grids.append(ase_grid)
        hours_above.append(h_above)
        pass_ase = (h_above > occ_hours)
        pass_ase_grids.append(pass_ase)

    # spatial daylight autonomy
    da_grids = []
    sda_grids = []
    pass_sda_grids = []
    for grid_info in grids_info:
        grid_id = grid_info['full_id']
        light_paths = [lp[0] for lp in grid_info['light_path']]
        arrays = []
        # combine total array for all light paths
        for light_path in light_paths:
            array = results._get_array(grid_info, light_path, res_type='total')
            array_filter = np.apply_along_axis(
                filter_array, 1, array, occ_mask)
            sun_up_hours = np.array(results.sun_up_hours).astype(int)
            shade_transmittance = states_schedule[light_path][sun_up_hours]
            shade_transmittance = shade_transmittance[occ_mask.astype(bool)]
            arrays.append(array_filter * shade_transmittance)
        array = sum(arrays)
        # calculate da per grid
        da_grid = da_array2d(array, total_occ=total_occ, threshold=threshold)
        da_grids.append(da_grid)
        # calculate sda per grid
        pass_sda = (da_grid >= target_time)
        pass_sda_grids.append(pass_sda)

    # create summaries for all grids and each grid individually
    summary, summary_grid = _leed_summary(
        pass_ase_grids, pass_sda_grids, grid_areas, grids_info)

    # credits
    if summary['sda'] >= 0.75:
        summary['credits'] = 3
    elif summary['sda'] >= 0.55:
        summary['credits'] = 2
    elif summary['sda'] >= 0.40:
        summary['credits'] = 1
    else:
        summary['credits'] = 0

    if (np.array(sda_grids) >= 0.55).all():
        if summary['credits'] <= 2:
            summary['credits'] += 1
        else:
            summary['credits'] = 'Exemplary performance'

    states_schedule = {k:v.tolist() for k, v in states_schedule.items()}

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

        for da, h_above, grid_info in zip(da_grids, hours_above, grids_info):
            grid_id = grid_info['full_id']
            da_file = folder.joinpath('da', f'{grid_id}.da')
            da_file.parent.mkdir(parents=True, exist_ok=True)
            hours_above_file = folder.joinpath(
                'ase_hours_above', f'{grid_id}.hours')
            hours_above_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(da_file, da, fmt='%.2f')
            np.savetxt(hours_above_file, h_above, fmt='%.0f')

    return (summary, summary_grid, da_grids, hours_above, states_schedule,
            grids_info)
