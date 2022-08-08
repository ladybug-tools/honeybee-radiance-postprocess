"""Commands to translate objects."""
import sys
import logging
import numpy as np
import click
from pathlib import Path

from ..reader import binary_to_array

_logger = logging.getLogger(__name__)


@click.group(help='Commands to translate objects.')
def translate():
    pass


@translate.command('npy-to-txt')
@click.argument(
    'npy-file', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--name', '-n', help='Output file name.', default='output', show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--extension', '-ext', help='Output file extension', default='.txt', show_default=True
)
@click.option(
    '--output-format', '-fmt', help='Output format for each element in the array',
    default='%.7e', show_default=True
)
def npy_to_txt(npy_file, name, output_folder, extension, output_format):
    """Convert a npy file to text file.

    This command reads a NumPy array from a npy file and saves it as readable file. The
    converted file is tab separated.

    \b
    Args:
        npy-file: Path to npy file.
    """
    try:
        array = np.load(npy_file)
        output = Path(output_folder, name + extension)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output, array, fmt=output_format, delimiter='\t')

    except Exception:
        _logger.exception('Converting npy file to text file failed.')
        sys.exit(1)
    else:
        sys.exit(0)


@translate.command('txt-to-npy')
@click.argument(
    'txt-file', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--name', '-n', help='Output file name.', default='output', show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def txt_to_npy(txt_file, name, output_folder):
    """Convert a text file to npy file.

    This command reads a space or tab separated text file saves it as a NumPy file. As
    an example the input file could be the annual illuminance values.

    \b
    Args:
        txt-file: Path to text file.
    """
    try:
        array = np.loadtxt(txt_file, dtype=np.float32)
        output = Path(output_folder, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, array)

    except Exception:
        _logger.exception('Converting text file to npy file failed.')
        sys.exit(1)
    else:
        sys.exit(0)


@translate.command('binary-to-npy')
@click.argument(
    'mtx-file', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--conversion', help='Conversion as a string. This option is useful to post-process '
    'the results from 3 RGB components into one as part of this command.'
)
@click.option(
    '--name', '-n', help='Output file name.', default='output', show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def binary_to_npy(mtx_file, conversion, name, output_folder):
    """Convert a binary Radiance file to a npy file.

    This command reads a binary Radiance matrix file and saves it as a NumPy file.

    \b
    Args:
        mtx-file: Path to binary Radiance file.
    """
    try:
        array = binary_to_array(mtx_file)
        if conversion:
            conversion = list(map(float, conversion.split(' ')))
            conversion = np.array(conversion, dtype=np.float32)
            array = np.dot(array, conversion)
        output = Path(output_folder, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, array)

    except Exception:
        _logger.exception('Converting binary Radiance file to npy file failed.')
        sys.exit(1)
    else:
        sys.exit(0)
