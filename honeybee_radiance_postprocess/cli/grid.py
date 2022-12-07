"""honeybee radiance postprocess grid commands."""
import click
import sys
import os
import logging
import json
import numpy as np
from pathlib import Path

_logger = logging.getLogger(__name__)


@click.group(help='Commands for generating and modifying sensor grids.')
def grid():
    pass


@grid.command('merge-folder')
@click.argument(
    'input-folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.argument(
    'output-folder',
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True))
@click.argument('extension', type=str)
@click.option(
    '--dist-info', '-di',
    help='An optional input for distribution information to put the grids back together '
    '. Alternatively, the command will look for a _redist_info.json file inside the '
    'folder.', type=click.Path(file_okay=True, dir_okay=False, resolve_path=True)
)
def merge_grid_folder(input_folder, output_folder, extension, dist_info):
    """Restructure files in a distributed folder.

    \b
    Args:
        input_folder: Path to input folder.
        output_folder: Path to the new restructured folder
        extension: Extension of the files to collect data from. It will be ``pts`` for
            sensor files. Another common extension is ``ill`` for the results of daylight
            studies.
    """
    try:
        # handle optional case for Functions input
        if dist_info and not Path(dist_info).is_file():
            dist_info = None
        restore_original_distribution(input_folder, output_folder, extension, dist_info)
    except Exception:
        _logger.exception('Failed to restructure data from folder.')
        sys.exit(1)
    else:
        sys.exit(0)


def restore_original_distribution(
        input_folder, output_folder, extension='npy', dist_info=None):
    """Restructure files to the original distribution based on the distribution info.

    Args:
        input_folder: Path to input folder.
        output_folder: Path to the new restructured folder
        extension: Extension of the files to collect data from. Default is ``npy`` for
            NumPy files. Another common extension is ``ill`` for the results of daylight
            studies.
        dist_info: Path to dist_info.json file. If None, the function will try to load
            ``_redist_info.json`` file from inside the input_folder. (Default: None).
    """
    if not dist_info:
        _redist_info_file = Path(input_folder, '_redist_info.json')
    else:
        _redist_info_file = Path(dist_info)

    assert _redist_info_file.is_file(), 'Failed to find %s' % _redist_info_file

    with open(_redist_info_file) as inf:
        data = json.load(inf)

    # create output folder
    output_folder = Path(output_folder)
    if not output_folder.is_dir():
        output_folder.mkdir()

    for f in data:
        output_file = Path(output_folder, f['identifier'])
        # ensure the new folder is created. in case the identifier has a subfolder
        parent_folder = output_file.parent
        if not parent_folder.is_dir():
            parent_folder.mkdir()

        out_arrays = []
        for src_info in f['dist_info']:
            src_file = Path(
                input_folder, '%s.%s' % (src_info['identifier'], extension)
            )
            st = src_info['st_ln']
            end = src_info['end_ln']

            array = np.load(src_file)
            slice_array = array[st:end+1,:]

            out_arrays.append(slice_array)

        out_array = np.concatenate(out_arrays)
        # save numpy array, .npy extension is added automatically
        np.save(output_file, out_array)
