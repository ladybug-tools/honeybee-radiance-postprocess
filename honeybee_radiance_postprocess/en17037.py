"""Functions for EN 17037 post-processing."""
from typing import Union
from pathlib import Path
import json

from ladybug.color import Colorset
from ladybug.datatype.fraction import Fraction
from ladybug.legend import LegendParameters

from . import np
from .results.annual_daylight import AnnualDaylight
from .dynamic import DynamicSchedule
from .metrics import da_array2d
from .util import filter_array


EN17037_RECOMMENDATIONS = {
    'minimum_illuminance': {
        'minimum': 100,
        'medium': 300,
        'high': 500,
    },
    'target_illuminance': {
        'minimum': 300,
        'medium': 500,
        'high': 750,
    },
}

EN17037_COMPLIANCE_VALUE = {
    'minimum': 1,
    'medium': 2,
    'high': 3,
}

EN17037_SPACE_TARGET = {
    'minimum_illuminance': 95,
    'target_illuminance': 50,
}

EN17037_CRITERION_LABELS = {
    ('minimum_illuminance', 'minimum'): 'Minimum Illuminance 100',
    ('minimum_illuminance', 'medium'): 'Minimum Illuminance 300',
    ('minimum_illuminance', 'high'): 'Minimum Illuminance 500',
    ('target_illuminance', 'minimum'): 'Target Illuminance 300',
    ('target_illuminance', 'medium'): 'Target Illuminance 500',
    ('target_illuminance', 'high'): 'Target Illuminance 750',
}


def en17037_compute(array: np.ndarray, grid_info: dict) -> dict:
    """Compute EN 17037 metrics for a 2D NumPy array.

    Args:
        array: A 2D NumPy array (sensors x occupied hours).
        grid_info: A grid information dictionary that must contain at least
            full_id and count.

    Returns:
        dict -- Nested result dictionary with the following structure::

            {
                'grid_id': str,
                'grid_count': int,
                'target_types': {
                    '<target_type>': {
                        'compliance_level': np.ndarray,
                        'levels': {
                            '<level>': {
                                'threshold': int,
                                'da': np.ndarray,
                                'sda': float,
                                'passes': bool,
                            },
                        },
                    },
                },
            }
    """
    grid_id = grid_info['full_id']
    grid_count = grid_info['count']

    result = {
        'grid_id': grid_id,
        'grid_count': grid_count,
        'target_types': {},
    }

    for target_type, thresholds in EN17037_RECOMMENDATIONS.items():
        space_target = EN17037_SPACE_TARGET[target_type]
        compliance_level = None
        levels = {}

        for level, threshold in thresholds.items():
            da = da_array2d(array, total_occ=4380, threshold=threshold)
            sda = float((da >= 50).mean() * 100)
            passes = sda >= space_target

            if passes:
                compliance_level = np.full(
                    (grid_count,), EN17037_COMPLIANCE_VALUE[level], dtype=int)

            levels[level] = {
                'threshold': threshold,
                'da': da,
                'sda': sda,
                'passes': passes,
            }

        if compliance_level is None:
            compliance_level = np.zeros(grid_count, dtype=int)

        result['target_types'][target_type] = {
            'compliance_level': compliance_level,
            'levels': levels,
        }

    return result


def en17037_to_files(
        array: np.ndarray, metrics_folder: Path, grid_info: dict) -> dict:
    """Compute EN 17037 metrics and write the results to metrics_folder.

    Args:
        array: A 2D NumPy array.
        metrics_folder: Output folder. Created if it does not exist.
        grid_info: A grid information dictionary.

    Returns:
        dict -- The result dictionary.
    """
    metrics_folder = Path(metrics_folder)
    results = en17037_compute(array, grid_info)

    grid_id = results['grid_id']

    da_folder = metrics_folder / 'da'
    sda_folder = metrics_folder / 'sda'
    compliance_folder = metrics_folder / 'compliance_level'

    for target_type, target_data in results['target_types'].items():
        for level, level_data in target_data['levels'].items():
            threshold = level_data['threshold']
            folder_name = f'{target_type}_{threshold}'

            da_level_folder = da_folder / folder_name
            da_level_folder.mkdir(parents=True, exist_ok=True)
            da_file = da_level_folder / f'{grid_id}.da'
            np.savetxt(da_file, level_data['da'], fmt='%.2f')

            sda_level_folder = sda_folder / folder_name
            sda_level_folder.mkdir(parents=True, exist_ok=True)
            sda_file = sda_level_folder / f'{grid_id}.sda'
            sda_file.write_text(str(round(level_data['sda'], 2)))

        compliance_level_folder = compliance_folder / target_type
        compliance_level_folder.mkdir(parents=True, exist_ok=True)
        compliance_level_file = compliance_level_folder / f'{grid_id}.pf'
        np.savetxt(compliance_level_file, target_data['compliance_level'], fmt='%i')

    return results


def en17037_to_folder(
        results: Union[str, AnnualDaylight], schedule: list,
        states: DynamicSchedule = None, grids_filter: str = '*',
        sub_folder: str = 'en17037') -> Path:
    """Compute annual EN 17037 metrics in a folder and write them in a subfolder.

    The results is an output folder of annual daylight recipe.

    Args:
        results: Results folder.
        schedule: An annual schedule for 8760 hours of the year as a list of
            values. This should be a daylight hours schedule.
        states: A dictionary of states. Defaults to None.
        grids_filter: A pattern to filter the grids. By default all the grids
            will be processed.
        sub_folder: An optional relative path for the subfolder where results
            are written. Default: en17037.

    Returns:
        Path -- Path to the results folder.
    """
    if not isinstance(results, AnnualDaylight):
        results = AnnualDaylight(results, schedule=schedule)
    else:
        results.schedule = schedule

    total_occ = results.total_occ
    occ_mask = results.occ_mask
    grids_info = results._filter_grids(grids_filter=grids_filter)

    sub_folder = Path(sub_folder)

    if total_occ != 4380:
        raise ValueError(
            f'There are {total_occ} occupied hours in the schedule. According '
            'to EN 17037 the schedule must consist of the daylight hours '
            'which is defined as the half of the year with the largest '
            'quantity of daylight'
        )

    all_grid_results: list[dict] = []
    all_output_folders: list[Path] = []

    for grid_info in grids_info:
        array = results._array_from_states(
            grid_info, states=states, res_type='total', zero_array=True)
        if np.any(array):
            array = np.apply_along_axis(filter_array, 1, array, occ_mask)

        grid_results = en17037_to_files(array, sub_folder, grid_info)
        all_grid_results.append((grid_info, grid_results))

        for target_type, target_data in grid_results['target_types'].items():
            for level, level_data in target_data['levels'].items():
                threshold = level_data['threshold']
                folder_name = f'{target_type}_{threshold}'
                all_output_folders.append(sub_folder / 'da' / folder_name)
                all_output_folders.append(sub_folder / 'sda' / folder_name)
            all_output_folders.append(
                sub_folder / 'compliance_level' / target_type)

    seen = set()
    for folder in all_output_folders:
        if folder in seen:
            continue
        seen.add(folder)
        grids_info_file = folder / 'grids_info.json'
        with open(grids_info_file, 'w') as outf:
            json.dump(grids_info, outf, indent=2)

    metric_info_dict = _annual_daylight_en17037_vis_metadata()
    da_folder = sub_folder / 'da'
    for metric, data in metric_info_dict.items():
        file_path = da_folder / metric / 'vis_metadata.json'
        with open(file_path, 'w') as fp:
            json.dump(data, fp, indent=4)

    _write_en17037_summary(all_grid_results, sub_folder)

    return sub_folder


def _build_grid_summary(grid_info: dict, grid_results: dict) -> dict:
    """Build the summary dict for a single grid.

    Args:
        grid_info: The grid information dictionary for this grid.
        grid_results: The result dict returned by en17037_compute for
            the same grid.

    Returns:
        dict -- Summary with the grid's display_name and a boolean
        passes entry for every EN 17037 criterion.
    """
    data = {
        'Sensor Grid': grid_info.get('display_name', grid_results['grid_id'])
    }
    for target_type, target_data in grid_results['target_types'].items():
        for level, level_data in target_data['levels'].items():
            label = EN17037_CRITERION_LABELS[(target_type, level)]
            data[label] = {
                'sDA': round(level_data['sda'], 2),
                'passes': level_data['passes'],
            }


    return data


def _write_en17037_summary(
        all_grid_results: list, sub_folder: Path) -> Path:
    """Write summary.json to sub_folder.

    The file contains one object per grid with the grid's display_name
    and a pass/fail boolean for each EN 17037 criterion.

    Args:
        all_grid_results: A list of (grid_info, grid_results) tuples as
            collected by en17037_to_folder.
        sub_folder: The root output folder.

    Returns:
        Path -- Path to the written summary.json file.
    """
    summary = [
        _build_grid_summary(grid_info, grid_results)
        for grid_info, grid_results in all_grid_results
    ]

    summary_file = sub_folder / 'summary.json'
    sub_folder.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as fp:
        json.dump(summary, fp, indent=2)

    return summary_file


def _annual_daylight_en17037_vis_metadata():
    """Return visualization metadata for annual daylight."""
    da_lpar = LegendParameters(min=0, max=100, colors=Colorset.annual_comfort())

    metric_info_dict = {
        'minimum_illuminance_100': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - minimum 100 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'minimum_illuminance_300': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - minimum 300 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'minimum_illuminance_500': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - minimum 500 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'target_illuminance_300': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - target 300 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'target_illuminance_500': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - target 500 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'target_illuminance_750': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - target 750 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        }
    }

    return metric_info_dict


def _annual_daylight_en17037_config():
    """Return vtk-config for annual daylight EN 17037."""
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
