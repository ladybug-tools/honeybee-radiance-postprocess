"""Functions for EN 17037 post-processing."""
from typing import Union
from pathlib import Path
import json

from ladybug.color import Colorset
from ladybug.datatype.fraction import Fraction
from ladybug.legend import LegendParameters
from honeybee.model import Model

from . import np
from .results.annual_daylight import AnnualDaylight
from .dynamic import DynamicSchedule
from .metrics import da_array2d
from .util import filter_array


EN17037_RECOMMENDATIONS = {
    'Minimum Illuminance': {
        'minimum': 100,
        'medium': 300,
        'high': 500,
    },
    'Target Illuminance': {
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
    'Minimum Illuminance': 95,
    'Target Illuminance': 50,
}

EN17037_CRITERION_LABELS = {
    ('Minimum Illuminance', 'minimum'): 'Minimum Illuminance 100',
    ('Minimum Illuminance', 'medium'): 'Minimum Illuminance 300',
    ('Minimum Illuminance', 'high'): 'Minimum Illuminance 500',
    ('Target Illuminance', 'minimum'): 'Target Illuminance 300',
    ('Target Illuminance', 'medium'): 'Target Illuminance 500',
    ('Target Illuminance', 'high'): 'Target Illuminance 750',
}


def en17037_compute(array: np.ndarray, grid_info: dict, area_array: np.ndarray = None) -> dict:
    """Compute EN 17037 metrics for a 2D NumPy array.

    Args:
        array: A 2D NumPy array (sensors x occupied hours).
        grid_info: A grid information dictionary that must contain at least
            full_id and count.
        area_array: An optional 1D NumPy array containing the face areas 
            corresponding to each sensor point.

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

            pass_mask = da >= 50
            if area_array is not None and len(area_array) == grid_count:
                total_area = area_array.sum()
                sda = float((area_array[pass_mask].sum() / total_area) * 100) if total_area > 0 else 0.0
            else:
                sda = float(pass_mask.mean() * 100)

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
        array: np.ndarray, metrics_folder: Path, grid_info: dict,
        area_array: np.ndarray = None) -> dict:
    """Compute EN 17037 metrics and write the results to metrics_folder.

    Args:
        array: A 2D NumPy array.
        metrics_folder: Output folder. Created if it does not exist.
        grid_info: A grid information dictionary.
        area_array: A 2D NumPy array representing the area of each sensor.

    Returns:
        dict -- The result dictionary.
    """
    metrics_folder = Path(metrics_folder)

    results = en17037_compute(array, grid_info, area_array=area_array)

    grid_id = results['grid_id']

    da_folder = metrics_folder / 'da'
    sda_folder = metrics_folder / 'sda'
    compliance_folder = metrics_folder / 'compliance_level'

    for target_type, target_data in results['target_types'].items():
        for level, level_data in target_data['levels'].items():
            threshold = level_data['threshold']
            folder_name = f'{target_type} {threshold}'

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

    grid_mesh_dict = {}
    for base_file in Path(results.folder).parent.iterdir():
        if base_file.suffix in ('.hbjson', '.hbpkl'):
            hb_model = Model.from_file(base_file)
            for s_grid in hb_model.properties.radiance.sensor_grids:
                if s_grid.mesh is not None:
                    grid_mesh_dict[s_grid.identifier] = np.array(s_grid.mesh.face_areas)
            break

    all_grid_results = []
    all_output_folders: list[Path] = []

    for grid_info in grids_info:
        area_array = grid_mesh_dict.get(grid_info['full_id'], None)

        array = results._array_from_states(
            grid_info, states=states, res_type='total', zero_array=True)
        if np.any(array):
            array = np.apply_along_axis(filter_array, 1, array, occ_mask)

        grid_results = en17037_to_files(array, sub_folder, grid_info, area_array=area_array)

        all_grid_results.append((grid_info, grid_results, area_array))

        for target_type, target_data in grid_results['target_types'].items():
            for level, level_data in target_data['levels'].items():
                threshold = level_data['threshold']
                folder_name = f'{target_type} {threshold}'
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

    # Writes individual grid summary
    _write_en17037_summary_grid(all_grid_results, sub_folder)

    # Writes the combined weighted summary
    _write_en17037_summary(all_grid_results, sub_folder)

    return sub_folder


def _build_grid_summary(
        grid_info: dict, grid_results: dict, area_array: np.ndarray = None) -> dict:
    """Build the summary dict for a single grid.

    Args:
        grid_info: The grid information dictionary for this grid.
        grid_results: The result dict returned by en17037_compute for
            the same grid.
        area_array: A 2D NumPy array representing the area of each sensor.

    Returns:
        dict -- Summary with the grid's display_name and a boolean
        passes entry for every EN 17037 criterion.
    """
    if area_array is not None:
        total_weight = float(area_array.sum())
        weight_label = 'Total Floor Area'
    else:
        total_weight = grid_info['count']
        weight_label = 'Total Sensors'

    data = {
        'Sensor Grid': grid_info.get('display_name', grid_results['grid_id']),
        weight_label: round(total_weight, 2) if area_array is not None else total_weight
    }
    for target_type, target_data in grid_results['target_types'].items():
        for level, level_data in target_data['levels'].items():
            label = EN17037_CRITERION_LABELS[(target_type, level)]
            data[label] = {
                'sDA': round(level_data['sda'], 2),
                'passes': level_data['passes'],
            }


    return data


def _write_en17037_summary_grid(
        all_grid_results: list, sub_folder: Path) -> Path:
    """Write summary_grid.json to sub_folder.

    The file contains one object per grid with the grid's display_name
    and a pass/fail boolean for each EN 17037 criterion.

    Args:
        all_grid_results: A list of (grid_info, grid_results, area_array) tuples as
            collected by en17037_to_folder.
        sub_folder: The root output folder.

    Returns:
        Path -- Path to the written summary_grid.json file.
    """
    summary = [
        _build_grid_summary(grid_info, grid_results, area_array)
        for grid_info, grid_results, area_array in all_grid_results
    ]

    summary_file = sub_folder / 'summary_grid.json'
    sub_folder.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as fp:
        json.dump(summary, fp, indent=2)

    return summary_file


def _write_en17037_summary(
        all_grid_results: list, sub_folder: Path) -> Path:
    """Write summary.json to sub_folder with weighted sDA and area breakdowns."""
    # check if all grids contain valid mesh face area arrays
    has_mesh = all(area_array is not None for _, _, area_array in all_grid_results)

    if has_mesh:
        total_weight = sum(float(area_array.sum()) for _, _, area_array in all_grid_results)
        weight_label = 'Total Floor Area'
    else:
        total_weight = sum(grid_info['count'] for grid_info, _, _ in all_grid_results)
        weight_label = 'Total Sensors'

    summary = {
        weight_label: round(total_weight, 2) if has_mesh else total_weight
    }

    # extract structural categories from the first available grid result
    _, first_grid_results, _ = all_grid_results[0]

    for target_type, target_data in first_grid_results['target_types'].items():
        space_target = EN17037_SPACE_TARGET[target_type]

        # compute overall weighted sDA criteria
        for level in target_data['levels'].keys():
            label = EN17037_CRITERION_LABELS[(target_type, level)]

            total_weighted_sda = 0.0
            for grid_info, grid_results, area_array in all_grid_results:
                weight = float(area_array.sum()) if has_mesh else grid_info['count']
                sda = grid_results['target_types'][target_type]['levels'][level]['sda']
                total_weighted_sda += sda * weight

            weighted_sda = total_weighted_sda / total_weight if total_weight > 0 else 0.0
            passes = weighted_sda >= space_target

            summary[label] = {
                'sDA': round(weighted_sda, 2),
                'passes': passes
            }

        # calculate exclusive area breakdown based on individual sensor point DA
        high_weight = 0.0
        medium_weight = 0.0
        minimum_weight = 0.0
        fail_weight = 0.0

        for grid_info, grid_results, area_array in all_grid_results:
            levels = grid_results['target_types'][target_type]['levels']
            da_high = levels['high']['da']
            da_medium = levels['medium']['da']
            da_minimum = levels['minimum']['da']

            # a sensor satisfies the threshold tier if its DA is >= 50%
            pass_high = da_high >= 50
            pass_medium = da_medium >= 50
            pass_minimum = da_minimum >= 50

            # group into exclusive, non-overlapping bins
            mask_high = pass_high
            mask_medium = pass_medium & ~pass_high
            mask_minimum = pass_minimum & ~pass_medium
            mask_fail = ~pass_minimum

            if has_mesh:
                high_weight += float(area_array[mask_high].sum())
                medium_weight += float(area_array[mask_medium].sum())
                minimum_weight += float(area_array[mask_minimum].sum())
                fail_weight += float(area_array[mask_fail].sum())
            else:
                high_weight += int(np.sum(mask_high))
                medium_weight += int(np.sum(mask_medium))
                minimum_weight += int(np.sum(mask_minimum))
                fail_weight += int(np.sum(mask_fail))

        # extract threshold lux integers for dynamic labeling
        thresh_high = EN17037_RECOMMENDATIONS[target_type]['high']
        thresh_med = EN17037_RECOMMENDATIONS[target_type]['medium']
        thresh_min = EN17037_RECOMMENDATIONS[target_type]['minimum']

        breakdown_suffix = "Area Percentage" if has_mesh else "Sensor Percentage"
        breakdown_key = f"{target_type} {breakdown_suffix}"

        if total_weight > 0:
            summary[breakdown_key] = {
                f'High (>= {thresh_high} lux)': round((high_weight / total_weight) * 100, 2),
                f'Medium ({thresh_med} - {thresh_high} lux)': round((medium_weight / total_weight) * 100, 2),
                f'Minimum ({thresh_min} - {thresh_med} lux)': round((minimum_weight / total_weight) * 100, 2),
                f'Fail (< {thresh_min} lux)': round((fail_weight / total_weight) * 100, 2)
            }
        else:
            summary[breakdown_key] = {
                'High': 0.0, 'Medium': 0.0, 'Minimum': 0.0, 'Fail': 0.0
            }

    summary_file = sub_folder / 'summary.json'
    sub_folder.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as fp:
        json.dump(summary, fp, indent=2)

    return summary_file


def _annual_daylight_en17037_vis_metadata():
    """Return visualization metadata for annual daylight."""
    da_lpar = LegendParameters(min=0, max=100, colors=Colorset.annual_comfort())

    metric_info_dict = {
        'Minimum Illuminance 100': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - minimum 100 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'Minimum Illuminance 300': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - minimum 300 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'Minimum Illuminance 500': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - minimum 500 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'Target Illuminance 300': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - target 300 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'Target Illuminance 500': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - target 500 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'Target Illuminance 750': {
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
