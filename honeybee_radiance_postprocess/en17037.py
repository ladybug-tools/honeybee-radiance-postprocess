"""Functions for EN 17037 post-processing."""
from typing import Union
from pathlib import Path
import json
import numpy as np

from .results import Results
from .metrics import da_array2d
from .util import filter_array


def en17037_to_files(
        array: np.ndarray, metrics_folder: Path, grid_id: str,
        total_occ: int = None) -> list:
    """Compute annual EN 17037 metrics for a NumPy array and write the results
    to a folder.

    This function generates 6 different files for daylight autonomy based on
    the varying level of recommendation in EN 17037.

    Args:
        array: A 2D NumPy array.
        metrics_folder: An output folder where the results will be written to.
            The folder will be created if it does not exist.
        grid_id: A grid id which will be used to name the output files.
        total_occ: Integer indicating the number of occupied hours. If not
            given any input the number of occupied hours will be found by the
            array shape.

    Returns:
        list -- List of paths of daylight autonomy folders.
    """
    recommendations = {
        'minimum_illuminance': {
            'minimum': 100,
            'medium': 300,
            'high': 500
        },
        'target_illuminance': {
            'minimum': 300,
            'medium': 500,
            'high': 750
        }
    }

    da_folders = []
    for target_type, thresholds in recommendations.items():
        type_folder = metrics_folder.joinpath(target_type)

        for level, threshold in thresholds.items():
            level_folder = type_folder.joinpath(level)

            # da
            da_file = level_folder.joinpath('da', f'{grid_id}.da')
            if not da_file.parent.is_dir():
                da_file.parent.mkdir(parents=True)
            da = da_array2d(array, total_occ=total_occ, threshold=threshold)
            np.savetxt(da_file, da, fmt='%.2f')

            # sda
            space_target = 50 if target_type == 'target_illuminance' else 95
            sda_file = level_folder.joinpath('sda', f'{grid_id}.sda')
            if not sda_file.parent.is_dir():
                sda_file.parent.mkdir(parents=True)
            sda = (da >= space_target).mean() * 100
            with open(sda_file, 'w') as sdaf:
                sdaf.write(str(round(sda, 2)))

            da_folders.append(da_file.parent)

    return da_folders


def en17037_to_folder(
        results: Union[str, Results], schedule: list, states: dict = None,
        grids_filter: str = '*', sub_folder: str = 'en17037') -> Path:
    """Compute annual EN 17037 metrics in a folder and write them in a subfolder.

    The results is an output folder of annual daylight recipe.

    Args:
        results: Results folder.
        schedule: An annual schedule for 8760 hours of the year as a list of
            values. This should be a daylight hours schedule.
        grids_filter: A pattern to filter the grids. By default all the grids
            will be processed.
        states: A dictionary of states. Defaults to None.
        sub_folder: An optional relative path for subfolder to copy results
            files. Default: en17037.

    Returns:
        str -- Path to results folder.
    """
    if not isinstance(results, Results):
        results = Results(results, schedule=schedule)
    else:
        results.schedule = schedule

    total_occ = results.total_occ
    occ_mask = results.occ_mask

    grids_info = results._filter_grids(grids_filter=grids_filter)

    if total_occ != 4380:
        raise ValueError(
            f'There are {total_occ} occupied hours in the schedule. According '
            'to EN 17037 the schedule must consist of the daylight hours '
            'which is defined as the half of the year with the largest '
            'quantity of daylight')

    metrics_folder = Path(results.folder).parent.joinpath(sub_folder)

    for grid_info in grids_info:
        array = results._array_from_states(
            grid_info, states=states, res_type='total')
        if np.any(array):
            array = np.apply_along_axis(
                filter_array, 1, array, occ_mask)
        da_folders = en17037_to_files(
            array, metrics_folder, grid_info['full_id'], total_occ)

    # copy grids_info.json to all results folders
    for da_folder in da_folders:
        grids_info_file = Path(da_folder, 'grids_info.json')
        with open(grids_info_file, 'w') as outf:
            json.dump(grids_info, outf, indent=2)

    # create info for available results. This file will be used by honeybee-vtk
    # for results visualization
    config_file = Path(metrics_folder, 'config.json')
    cfg = _annual_daylight_en17037_config()
    with open(config_file, 'w') as outf:
        json.dump(cfg, outf)

    return metrics_folder


def _annual_daylight_en17037_config():
    """Return vtk-config for annual daylight EN 17037. """
    cfg = {
        "data": [
            {
                "identifier": "Daylight Autonomy - target 300 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "target_illuminance/minimum/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
            {
                "identifier": "Daylight Autonomy - target 500 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "target_illuminance/medium/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
            {
                "identifier": "Daylight Autonomy - target 750 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "target_illuminance/high/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
            {
                "identifier": "Daylight Autonomy - minimum 100 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "minimum_illuminance/minimum/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
            {
                "identifier": "Daylight Autonomy - minimum 300 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "minimum_illuminance/medium/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
            {
                "identifier": "Daylight Autonomy - minimum 500 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "minimum_illuminance/high/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
        ]
    }

    return cfg
