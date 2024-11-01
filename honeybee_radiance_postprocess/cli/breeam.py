"""Commands for BREEAM post-processing."""
import sys
import logging
from pathlib import Path
import click

from honeybee_radiance_postprocess.breeam.breeam import breeam_daylight_assessment_4b

_logger = logging.getLogger(__name__)


@click.group(help='Commands for BREEAM post-processing of Radiance results.')
def breeam():
    pass


@breeam.command('breeam-4b')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option('--model-file', '-m', help='A Honeybee Model file that was used '
    'in the simulation.', type=click.Path(
    exists=False, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--sub-folder', '-sf', help='Relative path for subfolder to write output '
    'files.', default='breeam_summary', type=click.Path(
    exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path)
)
def breeam_4b(
    folder, model_file, sub_folder
):
    """Calculate metrics for BREEAM.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe.
        model-file: A Honeybee Model file that was used in the simulation.
    """
    try:
        breeam_daylight_assessment_4b(folder, model=model_file, sub_folder=sub_folder)
    except Exception:
        _logger.exception('Failed to calculate BREEAM metrics.')
        sys.exit(1)
    else:
        sys.exit(0)
