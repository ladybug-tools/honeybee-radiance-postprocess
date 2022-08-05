"""honeybee radiance daylight postprocessing commands."""
import json
import click
import sys
import os
import logging

from honeybee_radiance_postprocess.results import Results

_logger = logging.getLogger(__name__)


# we will import this from inside honeybee-radiance and expose it from honeybee-radiance
# cli
@click.group(help='Commands to post-process Radiance results.')
def post_process():
    pass


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
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    # optional input - only check if the file exist otherwise ignore
    if schedule and os.path.isfile(schedule):
        with open(schedule) as hourly_schedule:
            schedule = [int(float(v)) for v in hourly_schedule]
    else:
        schedule = None

    if states:
        with open(states) as json_file:
            states = json.load(json_file)

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
            with open(states) as json_file:
                states = json.load(json_file)

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
            with open(states) as json_file:
                states = json.load(json_file)

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
            with open(states) as json_file:
                states = json.load(json_file)

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
        with open(states) as json_file:
            states = json.load(json_file)

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
