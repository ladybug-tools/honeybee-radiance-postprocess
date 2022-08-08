"""Test cli translate module."""
import os
import numpy as np

from click.testing import CliRunner

from ladybug.futil import nukedir

from honeybee_radiance_postprocess.cli.translate import (npy_to_txt,
    txt_to_npy, binary_to_npy)


def test_npy_to_txt():
    runner = CliRunner()
    npy_file = './tests/assets/npy/Test_1.npy'
    name = 'Test_1'
    output_folder = './tests/assets/temp'
    output_file = './tests/assets/temp/Test_1.txt'
    cmd_args = [npy_file, '--name', name, '--output-folder', output_folder]

    result = runner.invoke(npy_to_txt, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(output_file)
    nukedir(output_folder)


def test_txt_to_npy():
    runner = CliRunner()
    txt_file = './tests/assets/txt/Test_1.txt'
    name = 'Test_1'
    output_folder = './tests/assets/temp'
    output_file = './tests/assets/temp/Test_1.npy'
    cmd_args = [txt_file, '--name', name, '--output-folder', output_folder]

    result = runner.invoke(txt_to_npy, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(output_file)
    nukedir(output_folder)


def test_binary_to_npy():
    runner = CliRunner()
    binary_file = './tests/assets/binary/sky.ill'
    name = 'sky'
    output_folder = './tests/assets/temp'
    output_file = './tests/assets/temp/sky.npy'
    cmd_args = [binary_file, '--name', name, '--output-folder', output_folder]

    result = runner.invoke(binary_to_npy, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(output_file)
    array = np.load(output_file)
    assert array.shape == (208, 4393, 3)
    nukedir(output_folder)


def test_binary_to_npy_conversion():
    runner = CliRunner()
    binary_file = './tests/assets/binary/sky.ill'
    name = 'sky'
    output_folder = './tests/assets/temp'
    output_file = './tests/assets/temp/sky.npy'
    conversion = '47.4 119.9 11.6'
    cmd_args = [
        binary_file, '--conversion', conversion, '--name', name,
        '--output-folder', output_folder
    ]

    result = runner.invoke(binary_to_npy, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(output_file)
    array = np.load(output_file)
    assert array.shape == (208, 4393)
    nukedir(output_folder)
