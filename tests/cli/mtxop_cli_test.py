"""Test cli mtxop module."""
import os
import numpy as np

from click.testing import CliRunner

from ladybug.futil import nukedir

from honeybee_radiance_postprocess.cli.mtxop import (two_matrix_operations,
    three_matrix_operations)


def test_two_matrix_operations():
    runner = CliRunner()
    first_mtx = './tests/assets/binary/sky.ill'
    second_mtx = './tests/assets/binary/sky_dir.ill'
    operator = '-'
    name = 'results'
    output_folder = './tests/assets/temp'
    output_file = './tests/assets/temp/results.npy'
    cmd_args = [
        first_mtx, second_mtx, '--operator', operator, '--name', name,
        '--output-folder', output_folder
    ]

    result = runner.invoke(two_matrix_operations, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(output_file)
    nukedir(output_folder)


def test_three_matrix_operations():
    runner = CliRunner()
    first_mtx = './tests/assets/binary/sky.ill'
    second_mtx = './tests/assets/binary/sky_dir.ill'
    third_mtx = './tests/assets/binary/sun.ill'
    operator_one = '-'
    operator_two = '+'
    name = 'results'
    output_folder = './tests/assets/temp'
    output_file = './tests/assets/temp/results.npy'
    cmd_args = [
        first_mtx, second_mtx, third_mtx,
        '--operator-one', operator_one, '--operator-two', operator_two,
        '--name', name, '--output-folder', output_folder
    ]

    result = runner.invoke(three_matrix_operations, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(output_file)
    nukedir(output_folder)


def test_three_matrix_operations_conversion():
    runner = CliRunner()
    first_mtx = './tests/assets/binary/sky.ill'
    second_mtx = './tests/assets/binary/sky_dir.ill'
    third_mtx = './tests/assets/binary/sun.ill'
    operator_one = '-'
    operator_two = '+'
    name = 'results'
    output_folder = './tests/assets/temp'
    conversion = '47.4 119.9 11.6'
    output_file = './tests/assets/temp/results.npy'
    cmd_args = [
        first_mtx, second_mtx, third_mtx,
        '--operator-one', operator_one, '--operator-two', operator_two,
        '--conversion', conversion, '--name', name,
        '--output-folder', output_folder
    ]

    result = runner.invoke(three_matrix_operations, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(output_file)
    array = np.load(output_file)
    assert array.shape == (208, 4393)
    nukedir(output_folder)


def test_three_matrix_operations_conversion_ascii():
    runner = CliRunner()
    first_mtx = './tests/assets/ascii/sky.ill'
    second_mtx = './tests/assets/ascii/sky_dir.ill'
    third_mtx = './tests/assets/ascii/sun.ill'
    operator_one = '-'
    operator_two = '+'
    name = 'results'
    output_folder = './tests/assets/temp'
    conversion = '47.4 119.9 11.6'
    output_file = './tests/assets/temp/results.npy'
    cmd_args = [
        first_mtx, second_mtx, third_mtx,
        '--operator-one', operator_one, '--operator-two', operator_two,
        '--conversion', conversion, '--ascii', '--name', name,
        '--output-folder', output_folder
    ]

    result = runner.invoke(three_matrix_operations, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(output_file)
    array = np.load(output_file)
    assert array.shape == (208, 4393)
    nukedir(output_folder)
