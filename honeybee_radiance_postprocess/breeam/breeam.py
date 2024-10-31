"""Functions for BREEAM post-processing."""
from typing import Union
from pathlib import Path
import numpy as np

from honeybee.model import Model
from honeybee_radiance.writer import _filter_by_pattern

from ..annual import occupancy_schedule_8_to_6
from ..results.annual_daylight import AnnualDaylight


def breeam_daylight_assessment_4b(
        results: Union[str, AnnualDaylight],
        grids_filter: str = '*',
        sub_folder: str = None):
    """Calculate credits for BREEAM 4b.

    Args:
        results: Path to results folder or a Results class object.
        grids_filter: The name of a grid or a pattern to filter the grids.
            Defaults to '*'.
        sub_folder: Relative path for a subfolder to write the output. If None,
            the files will not be written. Defaults to None.

    Returns:
        Tuple:
        -   summary: Summary of all grids combined.
        -   summary_grid: Summary of each grid individually.
        -   da_grids: List of daylight autonomy values for each grid. Each item
                in the list is a NumPy array of DA values.
        -   states_schedule: A dictionary of annual shading schedules for each
                aperture group.
        -   grids_info: Grid information.
    """
    schedule = occupancy_schedule_8_to_6(as_list=True)

    if not isinstance(results, AnnualDaylight):
        results = AnnualDaylight(results, schedule=schedule)
    else:
        # set schedule to default leed schedule
        results.schedule = schedule

    grids_info = results._filter_grids(grids_filter=grids_filter)

    # check to see if there is a HBJSON with sensor grid meshes for areas
    grid_areas, units_conversion = {}, 1
    grid_program_types = {}
    for base_file in Path(results.folder).parent.iterdir():
        if base_file.suffix in ('.hbjson', '.hbpkl'):
            hb_model: Model = Model.from_file(base_file)
            filt_grids = _filter_by_pattern(
                hb_model.properties.radiance.sensor_grids, filter=grids_filter)
            for s_grid in filt_grids:
                if s_grid.mesh is not None:
                    grid_areas[s_grid.identifier] = np.array(s_grid.mesh.face_areas).sum()
                else:
                    grid_areas[s_grid.identifier] = None
                hb_room = hb_model.rooms_by_identifier([s_grid.room_identifier])[0]
                program_type_id = hb_room.properties.energy.program_type.identifier
                grid_program_types[s_grid.identifier] = program_type_id
    if not grid_areas:
        grid_areas = {grid_info['full_id']: None for grid_info in grids_info}

    type_summary = {}
    for grid_info in grids_info:
        program_type = grid_program_types[grid_info['full_id']]
        if program_type not in type_summary:
            type_summary[program_type] = {}
        type_summary[program_type][grid_info['full_id']] = []

        array = results._array_from_states(grid_info)
        # calculate average along axis 0 (average for each hour)
        avg_ill = array.mean(axis=0)

        metrics_list = program_type_metrics[program_type]
        for metrics in metrics_list:
            metrics_summary = {}
            metrics_summary['type'] = metrics['type']
            metrics_summary['area'] = grid_areas[grid_info['full_id']]
            # calculate number of hours where avg. illuminance > target illuminance
            target_ill = metrics['average_daylight_illuminance']['illuminance']
            hrs_abv = (avg_ill >= target_ill).sum()
            # check if value is >= target hours
            target_hrs = metrics['average_daylight_illuminance']['hours']
            avg_comply = hrs_abv >= target_hrs

            # calculate number of hours where illuminance > target illuminance
            if program_type == 'BREEAM::Prison_buildings::Cells_and_custody_cells':
                minimum_comply = True
            else:
                target_ill = metrics['minimum_daylight_illuminance']['illuminance']
                hrs_abv_target = (array >= target_ill).sum(axis=1)
                # get the minimum, i.e., worst lit point
                worst_lit_point = np.min(hrs_abv_target)
                # check if values is >= target hours
                target_hrs = metrics['minimum_daylight_illuminance']['hours']
                minimum_comply = worst_lit_point >= target_hrs

            metrics_summary['credits'] = metrics['credits']
            if avg_comply and minimum_comply:
                metrics_summary['comply'] = True
            else:
                metrics_summary['comply'] = False

            type_summary[program_type][grid_info['full_id']].append(metrics_summary)

    program_summary = {}
    for program_type, grid_summary in type_summary.items():
        program_summary[program_type] = {}
        program_summary[program_type]['credits'] = 0  # set 0 by default
        program_summary[program_type]['comply'] = False  # set False by default

        metrics_summary = {}
        for grid_id, metrics_list in grid_summary.items():
            for metric in metrics_list:
                if metric['credits'] not in metrics_summary:
                    metrics_summary[metric['credits']] = {}
                metrics_summary[metric['credits']]['type'] = metric['type']
                if 'total_area' not in metrics_summary[metric['credits']]:
                    metrics_summary[metric['credits']]['total_area'] = 0
                metrics_summary[metric['credits']]['total_area'] += metric['area']
                if 'area_comply' not in metrics_summary[metric['credits']]:
                    metrics_summary[metric['credits']]['area_comply'] = 0
                if metric['comply']:
                    metrics_summary[metric['credits']]['area_comply'] += metric['area']

        for credit, metric_summary in metrics_summary.items():
            area_comply_pct = metric_summary['area_comply'] / metric_summary['total_area'] * 100
            metric_summary['area_comply_%'] = area_comply_pct
            for metric in program_type_metrics[program_type]:
                if credit == metric['credits']:
                    if area_comply_pct >= metric['area']:
                        metric_summary['comply'] = True
                    else:
                        metric_summary['comply'] = False

        for credit, metric_summary in metrics_summary.items():
            if metric_summary['comply'] and credit > program_summary[program_type]['credits']:
                program_summary[program_type]['comply'] = True
                program_summary[program_type]['credits'] = credit
                program_summary[program_type]['total_area'] = metric_summary['total_area']
                program_summary[program_type]['area_comply'] = metric_summary['area_comply']
                program_summary[program_type]['area_comply_%'] = metric_summary['area_comply_%']
                program_summary[program_type]['type'] = metric_summary['type']
            else:
                program_summary[program_type]['total_area'] = metric_summary['total_area']
                program_summary[program_type]['area_comply'] = metric_summary['area_comply']
                program_summary[program_type]['area_comply_%'] = metric_summary['area_comply_%']
                program_summary[program_type]['type'] = metric_summary['type']

    building_type_summary = {}
    for program_type, summary in program_summary.items():
        if summary['type'] not in building_type_summary:
            building_type_summary[summary['type']] = []
        building_type_summary[summary['type']].append(summary)

    credit_summary = []
    for building_type, summary in building_type_summary.items():
        _building_type_summary = {}
        _building_type_summary['type'] = building_type
        if all([v['comply'] for v in summary]):
            _building_type_summary['comply'] = True
        else:
            _building_type_summary['comply'] = False
        _building_type_summary['total_area'] = sum([v['total_area'] for v in summary])

        credit_summary.append(_building_type_summary)

    if sub_folder:
        pass

    return credit_summary, program_summary
