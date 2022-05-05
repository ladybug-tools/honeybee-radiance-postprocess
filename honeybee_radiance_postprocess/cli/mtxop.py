"""Commands to work with Radiance matrix using NumPy."""
import click
import sys
import logging
import numpy as np

from ..translator import array_to_feather, binary_to_array

_logger = logging.getLogger(__name__)


@click.group(help='Commands to work with Radiance matrix using NumPy.')
def mtxop():
    pass


@mtxop.command('operate-two')
@click.argument(
    'first-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'second-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--operator', type=click.Choice(['+', '-', '/', '*']), default='+',
    help='Operation between the two matrices.'
)
@click.option(
    '--conversion', help='Conversion as a string. This option is useful to post-process '
    'the results from 3 RGB components into one as part of this command.'
)
@click.option(
    '--name', '-n', help='Output file name.', default='output', show_default=True
)
@click.option(
    '--array/--table', is_flag=True, help='A flag to select which format to save the '
    'output in. The default is a NumPy array. The alternative is a PyArrow table saved '
    'in a feather file.', default=True
)
def two_matrix_operations(
    first_mtx, second_mtx, operator, conversion, name, array
        ):
    """Operations between two Radiance matrices.

    The input matrices must be Radiance binary matrices. The operations will be performed
    elementwise for the matrices.

    \b
    Args:
        first-mtx: Path to fist matrix.
        second-mtx: Path to second matrix.
    """
    try:
        first = binary_to_array(first_mtx)
        second = binary_to_array(second_mtx)

        data = eval('first %s second' % operator)

        if conversion:
            conversion = list(map(float, conversion.split(' ')))
            conversion = np.array(conversion, dtype=np.float32)
            data = np.dot(data, conversion)

        if array:
            np.save(name, data)
        else:
            array_to_feather(data, name)

    except Exception:
        _logger.exception('Operation on two Radiance matrix failed.')
        sys.exit(1)
    else:
        sys.exit(0)


@mtxop.command('operate-three')
@click.argument(
    'first-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'second-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'third-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--operator-one', type=click.Choice(['+', '-', '/', '*']), default='+',
    help='Operation between the two matrices.'
)
@click.option(
    '--operator-two', type=click.Choice(['+', '-', '/', '*']), default='+',
    help='Operation between the two matrices.'
)
@click.option(
    '--conversion', help='Conversion as a string. This option is useful to post-process '
    'the results from 3 RGB components into one as part of this command.'
)
@click.option(
    '--name', '-n', help='Output file name.', default='output', show_default=True
)
@click.option(
    '--array/--table', is_flag=True, help='A flag to select which format to save the '
    'output in. The default is a NumPy array. The alternative is a PyArrow table saved '
    'in a feather file.', default=True
)
def three_matrix_operations(
    first_mtx, second_mtx, third_mtx, operator_one, operator_two, conversion, name, array
        ):
    """Operations between three Radiance matrices.

    The input matrices must be Radiance binary matrices. The operations will be performed
    elementwise for the matrices.

    \b
    Args:
        first-mtx: Path to fist matrix.
        second-mtx: Path to second matrix.
        third-mtx: Path to third matrix.
    """
    try:
        first = binary_to_array(first_mtx)
        second = binary_to_array(second_mtx)
        third = binary_to_array(third_mtx)

        data = eval('first %s second %s third' % (operator_one, operator_two))

        if conversion:
            conversion = list(map(float, conversion.split(' ')))
            conversion = np.array(conversion, dtype=np.float32)
            data = np.dot(data, conversion)

        if array:
            np.save(name, data)
        else:
            array_to_feather(data, name)

    except Exception:
        _logger.exception('Operation on three Radiance matrix failed.')
        sys.exit(1)
    else:
        sys.exit(0)
