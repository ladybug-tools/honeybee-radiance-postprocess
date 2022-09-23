"""honeybee radiance daylight leed postprocessing commands."""
import json
import sys
import logging
import os
import click

from ..leed import leed_option_1
from ..results import Results

_logger = logging.getLogger(__name__)


@click.group(help='Commands for LEED post-processing of Radiance results.')
def leed():
    pass


@leed.command('daylight-option-1')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--shade-transmittance', '-st', help='A value to use as a multiplier in place of '
    'solar shading. Value for shade transmittance must be 1 > value > 0.',
    default=0.2, show_default=True, type=click.FLOAT
)
@click.option(
    '--shade-transmittance-file', '-stf', help='A JSON file with a dictionary '
    'where aperture groups are keys, and the value for each key is the shade '
    'transmittance. Values for shade transmittance must be 1 > value > 0. '
    'If any aperture groups are missing in the JSON file, its shade transmittance '
    'value will be set to the value of the shade transmittance option (0.2 by '
    'default).', default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--sub-folder', '-sf', help='Relative path for subfolder to write output '
    'files.', default='leed_summary'
)
def daylight_option_1(
    folder, shade_transmittance, shade_transmittance_file, grids_filter,
    sub_folder
):
    """Calculate credits for LEED v4.1 Daylight Option 1.

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
    if shade_transmittance_file and os.path.isfile(shade_transmittance_file):
        with open(shade_transmittance_file) as json_file:
            shd_trans = json.load(json_file)
        results = Results(folder)
        # check if aperture groups are missing in json file
        for light_path in results.light_paths:
            if (not light_path in shd_trans and
                light_path != '__static_apertures__'):
                shd_trans[light_path] = shade_transmittance
        shade_transmittance = shd_trans
    try:
        leed_option_1(
            folder, grids_filter=grids_filter,
            shade_transmittance=shade_transmittance, sub_folder=sub_folder
        )
    except Exception:
        _logger.exception('Failed to generate LEED summary.')
        sys.exit(1)
    else:
        sys.exit(0)
