"""Functions for BREEAM post-processing."""
from typing import Union
from pathlib import Path
import json
import numpy as np

from honeybee.model import Model
from honeybee_radiance.writer import _filter_by_pattern

from ..results.annual_daylight import AnnualDaylight


program_type_metrics = {
    'BREEAM::Education_buildings::Preschools': [
        {
            'type': 'Education buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2000
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2000
            }
        }
    ],
    'BREEAM::Education_buildings::Higher_education': [
        {
            'type': 'Education buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2000
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2000
            }
        },
        {
            'type': 'Education buildings',
            'credits': 1,
            'area': 60,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2000
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2000
            }
        }
    ],
    'BREEAM::Healthcare_buildings::Staff_and_public_areas': [
        {
            'type': 'Healthcare buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2650
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2650
            }
        },
        {
            'type': 'Healthcare buildings',
            'credits': 1,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2000
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2000
            }
        }
    ],
    'BREEAM::Healthcare_buildings::Patients_areas_and_consulting_rooms': [
        {
            'type': 'Healthcare buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2650
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2650
            }
        },
        {
            'type': 'Healthcare buildings',
            'credits': 1,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2000
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2000
            }
        }
    ],
    'BREEAM::Multi_residential_buildings::Kitchen': [
        {
            'type': 'Multi-residential buildings',
            'credits': 2,
            'area': 100,
            'average_daylight_illuminance': {
                'illuminance': 100,
                'hours': 3450
            },
            'minimum_daylight_illuminance': {
                'illuminance': 30,
                'hours': 3450
            }
        }
    ],
    'BREEAM::Multi_residential_buildings::Living_rooms_dining_rooms_studies': [
        {
            'type': 'Multi-residential buildings',
            'credits': 2,
            'area': 100,
            'average_daylight_illuminance': {
                'illuminance': 100,
                'hours': 3450
            },
            'minimum_daylight_illuminance': {
                'illuminance': 30,
                'hours': 3450
            }
        }
    ],
    'BREEAM::Multi_residential_buildings::Non_residential_or_communal_spaces': [
        {
            'type': 'Multi-residential buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 200,
                'hours': 2650
            },
            'minimum_daylight_illuminance': {
                'illuminance': 60,
                'hours': 2650
            }
        }
    ],
    'BREEAM::Retail_buildings::Sales_areas': [
        {
            'type': 'Retail buildings',
            'credits': 1,
            'area': 35,
            'average_daylight_illuminance': {
                'illuminance': 200,
                'hours': 2650
            },
            'minimum_daylight_illuminance': {
                'illuminance': 200,
                'hours': 2650
            }
        }
    ],
    'BREEAM::Retail_buildings::Other_occupied_areas': [
        {
            'type': 'Retail buildings',
            'credits': 1,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 200,
                'hours': 2650
            },
            'minimum_daylight_illuminance': {
                'illuminance': 60,
                'hours': 2650
            }
        }
    ],
    'BREEAM::Prison_buildings::Cells_and_custody_cells': [
        {
            'type': 'Prison buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 100,
                'hours': 3150
            },
            'minimum_daylight_illuminance': None
        }
    ],
    'BREEAM::Prison_buildings::Internal_association_or_atrium': [
        {
            'type': 'Prison buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2650
            },
            'minimum_daylight_illuminance': {
                'illuminance': 210,
                'hours': 2650
            }
        }
    ],
    'BREEAM::Prison_buildings::Patient_care_spaces': [
        {
            'type': 'Prison buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2650
            },
            'minimum_daylight_illuminance': {
                'illuminance': 210,
                'hours': 2650
            }
        }
    ],
    'BREEAM::Prison_buildings::Teaching_lecture_and_seminar_spaces': [
        {
            'type': 'Prison buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2000
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2000
            }
        }
    ],
    'BREEAM::Office_buildings::Occupied_spaces': [
        {
            'type': 'Office buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2000
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2000
            }
        }
    ],
    'BREEAM::Crèche_buildings::Occupied_spaces': [
        {
            'type': 'Crèche buildings',
            'credits': 2,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2000
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2000
            }
        }
    ],
    'BREEAM::Other_buildings::Occupied_spaces': [
        {
            'type': 'Other buildings',
            'credits': 1,
            'area': 80,
            'average_daylight_illuminance': {
                'illuminance': 300,
                'hours': 2000
            },
            'minimum_daylight_illuminance': {
                'illuminance': 90,
                'hours': 2000
            }
        }
    ]
}


def breeam_daylight_assessment_4b(
        results: Union[str, AnnualDaylight], model: Union[str, Path, Model] = None,
        grids_filter: str = '*', sub_folder: str = None):
    """Calculate credits for BREEAM 4b.

    Args:
        results: Path to results folder or a Results class object.
        model: A model as a path or a HB Model object. If None, the function
            will look for a model in the parent of the results folder. If a model
            does not exist in this directory the function will raise an error.
            Defaults to None.
        grids_filter: The name of a grid or a pattern to filter the grids.
            Defaults to '*'.
        sub_folder: Relative path for a subfolder to write the output. If None,
            the files will not be written. Defaults to None.

    Returns:
        Tuple:
        -   credit_summary: Summary of each building type.
        -   program_summary: Summary of program type / space type.
    """
    if not isinstance(results, AnnualDaylight):
        results = AnnualDaylight(results)

    grids_info = results._filter_grids(grids_filter=grids_filter)

    # check to see if there is a HBJSON with sensor grid meshes for areas
    grid_areas = {}
    grid_program_types = {}
    if model is None:
        found_file = False
        for base_file in Path(results.folder).parent.iterdir():
            if base_file.suffix in ('.hbjson', '.hbpkl'):
                hb_model: Model = Model.from_file(base_file)
                found_file = True
                break
        if not found_file:
            raise FileNotFoundError(
                'Found no hbjson or hbpkl file in parent of results folder.')
    else:
        if isinstance(model, Model):
            hb_model = model
        else:
            hb_model = Model.from_file(model)

    filt_grids = _filter_by_pattern(
        hb_model.properties.radiance.sensor_grids, filter=grids_filter)
    for s_grid in filt_grids:
        if s_grid.mesh is not None:
            grid_areas[s_grid.identifier] = np.array(s_grid.mesh.face_areas).sum()
        else:
            grid_areas[s_grid.identifier] = None
        hb_room = hb_model.rooms_by_identifier([s_grid.room_identifier])[0]
        try:
            program_type_id = hb_room.properties.energy.program_type.identifier
        except AttributeError as e:
            raise ImportError('honeybee_energy library must be installed to use '
                              'breeam_daylight_assessment method. {}'.format(e))
        if program_type_id not in program_type_metrics:
            program_type_id = 'BREEAM::Office_buildings::Occupied_spaces'
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

    program_summary = []
    for program_type, grid_summary in type_summary.items():
        program_type_summary = {}
        program_type_summary['program_type'] = program_type
        program_type_summary['credits'] = 0  # set 0 by default
        program_type_summary['comply'] = False  # set False by default

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
            if metric_summary['comply'] and credit > program_type_summary['credits']:
                program_type_summary['comply'] = True
                program_type_summary['credits'] = credit
                program_type_summary['total_area'] = metric_summary['total_area']
                program_type_summary['area_comply'] = metric_summary['area_comply']
                program_type_summary['area_comply_%'] = metric_summary['area_comply_%']
                program_type_summary['type'] = metric_summary['type']
            else:
                program_type_summary['total_area'] = metric_summary['total_area']
                program_type_summary['area_comply'] = metric_summary['area_comply']
                program_type_summary['area_comply_%'] = metric_summary['area_comply_%']
                program_type_summary['type'] = metric_summary['type']

        program_summary.append(program_type_summary)

    building_type_summary = {}
    for _program_summary in program_summary:
        if _program_summary['type'] not in building_type_summary:
            building_type_summary[_program_summary['type']] = []
        building_type_summary[_program_summary['type']].append(_program_summary)

    credit_summary = []
    for building_type, summary in building_type_summary.items():
        _building_type_summary = {}
        _building_type_summary['type'] = building_type
        if all([v['comply'] for v in summary]):
            _building_type_summary['comply'] = True
            _building_type_summary['credits'] = min([v['credits'] for v in summary])
        else:
            _building_type_summary['comply'] = False
            _building_type_summary['credits'] = 0
        _building_type_summary['total_area'] = sum([v['total_area'] for v in summary])

        credit_summary.append(_building_type_summary)

    if sub_folder:
        folder = Path(sub_folder)
        folder.mkdir(parents=True, exist_ok=True)

        credit_summary_file = folder.joinpath('summary.json')
        credit_summary_file.write_text(json.dumps(credit_summary, indent=2))
        program_summary_file = folder.joinpath('program_summary.json')
        program_summary_file.write_text(json.dumps(program_summary, indent=2))

    return credit_summary, program_summary
