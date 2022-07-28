"""honeybee radiance daylight postprocessing commands."""
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
    '--states', '-st', help='A dictionary of states.', default=None, type=dict,
    show_default=True
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub_folder', '-sf', help='Optional relative path for subfolder to write output '
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
        daylight recipe. Folder should include grids_info.json and sun-up-hours.txt.
        The command uses the list in grids_info.json to find the result files for each
        sensor grid.
    """
    # optional input - only check if the file exist otherwise ignore
    if schedule and os.path.isfile(schedule):
        with open(schedule) as hourly_schedule:
            schedule = [int(float(v)) for v in hourly_schedule]
    else:
        schedule = None
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
    '--hoys-file', '-h', help='Path to an HOYs file. Values must be separated by new line. '
    'If not provided the data will not be filtered by HOYs.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A dictionary of states.', default=None, type=dict,
    show_default=True
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--res-type', '-rt', help='Which type of results to process', default='total',
    show_default=True
)
def average_values(
    folder, hoys_file, states, grids_filter, res_type
):
    """Get average values for each sensor over a given period.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual
        daylight recipe. Folder should include grids_info.json and sun-up-hours.txt.
        The command uses the list in grids_info.json to find the result files for each
        sensor grid.
    """
    if hoys_file:
        with open(hoys_file) as hoys:
            hoys = [int(h) for h in hoys.readlines()]
    else:
        hoys = []

    try:
        results = Results(folder, )
        average_values = results.average_values(
            hoys=hoys, states=states, grids_filter=grids_filter,
            res_type=res_type)
        for grid in average_values:
            print(grid[0:5])
    except Exception:
        _logger.exception('Failed to calculate average values.')
        sys.exit(1)
    else:
        sys.exit(0)
