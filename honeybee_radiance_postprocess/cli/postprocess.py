"""honeybee radiance daylight postprocessing commands."""
from pathlib import Path
import sys
import os
import logging
import json
import click
import numpy as np

from ..dynamic import DynamicSchedule
from honeybee_radiance_postprocess.results import Results
from honeybee_radiance_postprocess.metrics import da_array2d, cda_array2d, \
    udi_array2d, udi_lower_array2d, udi_upper_array2d
from honeybee_radiance_postprocess.reader import binary_to_array
from honeybee_radiance.postprocess.annual import filter_schedule_by_hours, \
    generate_default_schedule

from ..en17037 import en17037_to_folder
from ..util import filter_array
from .two_phase import two_phase
from .leed import leed
from ..helper import model_grid_areas, grid_summary

_logger = logging.getLogger(__name__)


@click.group(help='Commands to post-process Radiance results.')
def post_process():
    pass

post_process.add_command(two_phase)
post_process.add_command(leed)

@post_process.command('annual-daylight')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--schedule', '-sch', help='Path to an annual schedule file. Values should be 0-1 '
    'separated by new line. If not provided an 8-5 annual schedule will be created.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--threshold', '-t', help='Threshold illuminance level for daylight autonomy.',
    default=300, type=int, show_default=True
)
@click.option(
    '--lower-threshold', '-lt',
    help='Minimum threshold for useful daylight illuminance.', default=100, type=int,
    show_default=True
)
@click.option(
    '--upper-threshold', '-ut',
    help='Maximum threshold for useful daylight illuminance.', default=3000, type=int,
    show_default=True
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states are not '
    'provided the default states will be used for any aperture groups.', default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def annual_metrics(
    folder, schedule, threshold, lower_threshold, upper_threshold, states,
    grids_filter, sub_folder
):
    """Compute annual metrics in a folder and write them in a subfolder.

    \b
    This command generates 5 files for each input grid.
        da/{grid-name}.da -> Daylight Autonomy
        cda/{grid-name}.cda -> Continuos Daylight Autonomy
        udi/{grid-name}.udi -> Useful Daylight Illuminance
        udi_lower/{grid-name}_upper.udi -> Upper Useful Daylight Illuminance
        udi_upper/{grid-name}_lower.udi -> Lower Useful Daylight Illuminance

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual
            daylight recipe. Folder should include grids_info.json and
            sun-up-hours.txt.
    """
    # optional input - only check if the file exist otherwise ignore
    if schedule and os.path.isfile(schedule):
        with open(schedule) as hourly_schedule:
            schedule = [int(float(v)) for v in hourly_schedule]
    else:
        schedule = None

    if states:
        states = DynamicSchedule.from_json(states)

    try:
        results = Results(folder, schedule=schedule)
        results.annual_metrics_to_folder(
            sub_folder, threshold=threshold, min_t=lower_threshold,
            max_t=upper_threshold, states=states, grids_filter=grids_filter
        )
    except Exception:
        _logger.exception('Failed to calculate annual metrics.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('annual-daylight-en17037')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.argument(
    'schedule',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states are not '
    'provided the default states will be used for any aperture groups.', default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub_folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='en17037'
)
def annual_en17037_metrics(
    folder, schedule, states, grids_filter, sub_folder
):
    """Compute annual EN 17037 metrics in a folder and write them in a subfolder.

    \b
    This command generates multiple files for each input grid. Files for target
    illuminance and minimum illuminance will be calculated for three levels of
    recommendation: minimum, medium, high.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual
            daylight recipe. Folder should include grids_info.json and
            sun-up-hours.txt.
        schedule: Path to an annual schedule file. Values should be 0-1
            separated by new line. This should be a daylight hours schedule.
    """
    with open(schedule) as hourly_schedule:
        schedule = [int(float(v)) for v in hourly_schedule]

    if states:
        states = DynamicSchedule.from_json(states)

    try:
        en17037_to_folder(
            folder, schedule, states=states, grids_filter=grids_filter,
            sub_folder=sub_folder)
    except Exception:
        _logger.exception('Failed to calculate annual EN 17037 metrics.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('average-values')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--hoys-file', '-h', help='Path to an HOYs file. Values must be separated by '
    'new line. If not provided the data will not be filtered by HOYs.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def average_values(
    folder, hoys_file, states, grids_filter, total, sub_folder
):
    """Get average values for each sensor over a given period.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    try:
        if hoys_file:
            with open(hoys_file) as hoys:
                hoys = [float(h) for h in hoys.readlines()]
        else:
            hoys = []

        if states:
            states = DynamicSchedule.from_json(states)

        res_type = 'total' if total is True else 'direct'

        results = Results(folder)
        results.average_values_to_folder(
            sub_folder, hoys=hoys, states=states, grids_filter=grids_filter,
            res_type=res_type)
    except Exception:
        _logger.exception('Failed to calculate average values.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('median-values')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--hoys-file', '-h', help='Path to an HOYs file. Values must be separated by '
    'new line. If not provided the data will not be filtered by HOYs.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def median_values(
    folder, hoys_file, states, grids_filter, total, sub_folder
):
    """Get median values for each sensor over a given period.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    try:
        if hoys_file:
            with open(hoys_file) as hoys:
                hoys = [float(h) for h in hoys.readlines()]
        else:
            hoys = []

        if states:
            states = DynamicSchedule.from_json(states)

        res_type = 'total' if total is True else 'direct'

        results = Results(folder)
        results.median_values_to_folder(
            sub_folder, hoys=hoys, states=states, grids_filter=grids_filter,
            res_type=res_type)
    except Exception:
        _logger.exception('Failed to calculate median values.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('cumulative-values')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--hoys-file', '-h', help='Path to an HOYs file. Values must be separated by '
    'new line. If not provided the data will not be filtered by HOYs.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def cumulative_values(
    folder, hoys_file, states, grids_filter, total, sub_folder
):
    """Get cumulative values for each sensor over a given period.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    try:
        if hoys_file:
            with open(hoys_file) as hoys:
                hoys = [float(h) for h in hoys.readlines()]
        else:
            hoys = []

        if states:
            states = DynamicSchedule.from_json(states)

        res_type = 'total' if total is True else 'direct'

        results = Results(folder)
        results.cumulative_values_to_folder(
            sub_folder, hoys=hoys, states=states, grids_filter=grids_filter,
            res_type=res_type)
    except Exception:
        _logger.exception('Failed to calculate cumulative values.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('peak-values')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--hoys-file', '-h', help='Path to an HOYs file. Values must be separated by '
    'new line. If not provided the data will not be filtered by HOYs.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--coincident/--non-coincident', is_flag=True, default=False, show_default=True,
    help='Boolean to indicate whether output values represent the the peak value for '
    'each sensor throughout the entire analysis (False) or they represent the highest '
    'overall value across each sensor grid at a particular timestep (True).'
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def peak_values(
    folder, hoys_file, states, grids_filter, total, coincident, sub_folder
):
    """Get peak values for each sensor over a given period.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    try:
        if hoys_file:
            with open(hoys_file) as hoys:
                hoys = [float(h) for h in hoys.readlines()]
        else:
            hoys = []

        if states:
            states = DynamicSchedule.from_json(states)

        res_type = 'total' if total is True else 'direct'

        results = Results(folder)
        results.peak_values_to_folder(
            sub_folder, hoys=hoys, states=states, grids_filter=grids_filter,
            coincident=coincident, res_type=res_type)
    except Exception:
        _logger.exception('Failed to calculate peak values.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('annual-to-data')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sensor-index', '-si', help='A JSON file with a dictionary of sensor indices '
    'for each grid. If not provided all sensors will be used.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--output-file', '-f', help='Optional file to output the JSON strings of '
    'the data collections. By default, it will be printed to stdout',
    type=click.File('w'), default='-', show_default=True
)
def annual_to_data(
    folder, states, grids_filter, sensor_index, total, output_file
):
    """Get annual data collections as JSON files.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    if states:
        states = DynamicSchedule.from_json(states)

    if sensor_index:
        with open(sensor_index) as json_file:
            sensor_index = json.load(json_file)

    res_type = 'total' if total is True else 'direct'

    try:
        results = Results(folder)
        data_cs, grids_info, sensor_index = results.annual_data(
            states=states, grids_filter=grids_filter,
            sensor_index=sensor_index, res_type=res_type)
        data_colls = [[data.to_dict() for data in data_list] for data_list in data_cs]
        output_file.write(json.dumps(data_colls))
    except Exception:
        _logger.exception('Failed to create data collections.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('annual-sunlight-exposure')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--schedule', '-sch', help='Path to an annual schedule file. Values should be 0-1 '
    'separated by new line. If not provided an 8-5 annual schedule will be created.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--direct-threshold', '-dt', help='The threshold that determines if a '
    'sensor is overlit.',
    default=1000, type=float, show_default=True
)
@click.option(
    '--occ_hours', '-oh', help='The number of occupied hours that cannot '
    'receive more than the direct_threshold.', default=250, type=int,
    show_default=True
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states are not '
    'provided the default states will be used for any aperture groups.', default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def annual_sunlight_exposure(
    folder, schedule, direct_threshold, occ_hours, states, grids_filter,
    sub_folder
):
    """Compute annual sunlight exposure in a folder and write them in a subfolder.

    \b
    This command generates 2 files for each input grid.
        ase/{grid-name}.ase -> Annual Sunlight Exposure
        hours_above/{grid-name}.hours -> Number of overlit hours for each sensor

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual
            daylight recipe. Folder should include grids_info.json and
            sun-up-hours.txt.
    """
    # optional input - only check if the file exist otherwise ignore
    if schedule and os.path.isfile(schedule):
        with open(schedule) as hourly_schedule:
            schedule = [int(float(v)) for v in hourly_schedule]
    else:
        schedule = None

    if states:
        states = DynamicSchedule.from_json(states)

    try:
        results = Results(folder, schedule=schedule)
        results.annual_sunlight_exposure_to_folder(
            sub_folder, direct_threshold=direct_threshold, occ_hours=occ_hours,
            states=states, grids_filter=grids_filter
        )
    except Exception:
        _logger.exception('Failed to calculate annual sunlight exposure.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('annual-daylight-file')
@click.argument(
    'file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'sun-up-hours',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--schedule', '-sch', help='Path to an annual schedule file. Values should be 0-1 '
    'separated by new line. If not provided an 8-5 annual schedule will be created.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--threshold', '-t', help='Threshold illuminance level for daylight autonomy.',
    default=300, type=int, show_default=True
)
@click.option(
    '--lower-threshold', '-lt',
    help='Minimum threshold for useful daylight illuminance.', default=100, type=int,
    show_default=True
)
@click.option(
    '--upper-threshold', '-ut',
    help='Maximum threshold for useful daylight illuminance.', default=3000, type=int,
    show_default=True
)
@click.option(
    '--grid-name', '-gn', help='Optional name of each metric file.',
    default=None, show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def annual_metrics_file(
    file, sun_up_hours, schedule, threshold, lower_threshold, upper_threshold,
    grid_name, sub_folder
):
    """Compute annual metrics for a single file and write the metrics in a
    subfolder.

    \b
    This command generates 5 files for each input grid.
        da/{grid-name}.da -> Daylight Autonomy
        cda/{grid-name}.cda -> Continuos Daylight Autonomy
        udi/{grid-name}.udi -> Useful Daylight Illuminance
        udi_lower/{grid-name}_upper.udi -> Upper Useful Daylight Illuminance
        udi_upper/{grid-name}_lower.udi -> Lower Useful Daylight Illuminance

    \b
    Args:
        file: Annual illuminance file. This can be either a NumPy file or a
            binary Radiance file.
    """
    file = Path(file)
    # load file to array
    try:
        array = np.load(file)
    except Exception:
        array = binary_to_array(file)

    # read sun up hours to list of integers
    with open(sun_up_hours) as _sun_up_hours:
        sun_up_hours = [int(float(v)) for v in _sun_up_hours]
    # optional input - only check if the file exist otherwise ignore
    if schedule and os.path.isfile(schedule):
        with open(schedule) as hourly_schedule:
            schedule = [int(float(v)) for v in hourly_schedule]
    else:
        schedule = generate_default_schedule()

    if grid_name is None:
        grid_name = file.stem

    occ_pattern, total_hours, sun_down_occ_hours = \
        filter_schedule_by_hours(sun_up_hours, schedule=schedule)
    occ_mask = np.array(occ_pattern)
    array_filter = np.apply_along_axis(
        filter_array, 1, array, mask=occ_mask)
    try:
        da = da_array2d(array_filter, total_occ=total_hours, threshold=threshold)
        cda = cda_array2d(array_filter, total_occ=total_hours, threshold=threshold)
        udi = udi_array2d(
            array_filter, total_occ=total_hours, min_t=lower_threshold,
            max_t=upper_threshold)
        udi_lower = udi_lower_array2d(
            array_filter, total_occ=total_hours, min_t=lower_threshold,
            sun_down_occ_hours=sun_down_occ_hours)
        udi_upper = udi_upper_array2d(
            array_filter, total_occ=total_hours, max_t=upper_threshold)

        sub_folder = Path(sub_folder)
        pattern = {
            'da': da, 'cda': cda, 'udi_lower': udi_lower, 'udi': udi,
            'udi_upper': udi_upper
        }
        for metric, data in pattern.items():
            metric_folder = sub_folder.joinpath(metric)
            extension = metric.split('_')[0]
            output_file = metric_folder.joinpath(f'{grid_name}.{extension}')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, data, fmt='%.2f')
    except Exception:
        _logger.exception('Failed to calculate annual metrics.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('grid-summary')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--model', '-m', help='An optional HBJSON model file. This will be used to '
    'find the area of the grids. The area is used when calculating percentages '
    'of floor area.',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-info', '-gi', help='An optional JSON file with grid information. '
    'If no file is provided the command will look for a file in the folder.',
    default=None, show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--name', '-n', help='Optional filename of grid summary.',
    type=str, default='grid_summary', show_default=True
)
@click.option(
    '--grid-metrics', '-gm', help='An optional JSON file with additional '
    'custom metrics to calculate.', default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--sub-folder/--main-folder', is_flag=True, default=True,
    help='If sub-folder is selected it will look for any sub-folders in the '
    'folder argument. If main-folder is seleted it will look result files '
    'in the folder argument.'
)
def grid_summary_metric(
    folder, model, grids_info, name, grid_metrics, sub_folder
):
    """Calculate a grid summary.

    \b
    Args:
        folder: A folder with results.
    """
    try:
        # create Path object
        folder = Path(folder)

        # get grids information
        if grids_info:
            with open(grids_info) as gi:
                grids_info = json.load(gi)

        # get grid metrics
        if grid_metrics and Path(grid_metrics).is_file():
            with open(grid_metrics) as gm:
                grid_metrics = json.load(gm)
        else:
            grid_metrics = None

        # check to see if there is a HBJSON with sensor grid meshes for areas
        if grids_info and model:
            grid_areas = model_grid_areas(model, grids_info)
        else:
            grid_areas = None

        grid_summary(
            folder, grid_areas=grid_areas, grids_info=grids_info, name=name,
            grid_metrics=grid_metrics, sub_folder=sub_folder)

    except Exception:
        _logger.exception('Failed to calculate grid summary.')
        sys.exit(1)
    else:
        sys.exit(0)
