"""honeybee-radiance-postprocess WELL commands."""
import sys
import logging
import click

from ..well.well import well_annual_daylight

_logger = logging.getLogger(__name__)


@click.group(help='Commands for WELL post-processing of Radiance results.')
def well():
    pass


@well.command('well-annual-daylight')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.argument(
    'daylight-hours',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Relative path for subfolder to write output '
    'files.', default='well_summary', show_default=True
)
def well_daylight(
    folder, daylight_hours, grids_filter, sub_folder
):
    """Calculate credits for WELL L06.

    Use the shade-transmittance option to set a shade transmittance values for
    aperture groups. The shade-transmittance-file option takes precedence over
    the shade-transmittance, however, if any aperture groups are missing in the
    JSON file given to the shade-transmittance-file option, the value from
    shade-transmittance will be used for those aperture groups.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. The daylight simulation must include aperture groups.
    """
    with open(daylight_hours) as hourly_schedule:
        daylight_hours = [int(float(v)) for v in hourly_schedule]

    try:
        well_annual_daylight(
            folder, daylight_hours, grids_filter=grids_filter, sub_folder=sub_folder
        )
    except Exception:
        _logger.exception('Failed to generate WELL summary.')
        sys.exit(1)
    else:
        sys.exit(0)
