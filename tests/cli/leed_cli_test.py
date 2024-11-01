"""Test cli leed module."""
import os

from click.testing import CliRunner

from ladybug.futil import nukedir
from honeybee_radiance_postprocess.cli.leed import daylight_option_one


def test_daylight_option_one():
    runner = CliRunner()
    results = './tests/assets/leed/results'
    shade_transmittance = 0.02
    sub_folder = './tests/assets/temp/leed_summary'
    cmd_args = [
        results, '--shade-transmittance', shade_transmittance, '--sub-folder',
        sub_folder
    ]

    result = runner.invoke(daylight_option_one, cmd_args)
    assert result.exit_code == 0
    assert os.path.isdir(sub_folder)
    nukedir(sub_folder, rmdir=True)


def test_daylight_option_one_shade_transmittance_file():
    runner = CliRunner()
    results = './tests/assets/leed/results'
    shade_transmittance_file = './tests/assets/leed/shd.json'
    sub_folder = './tests/assets/temp/leed_summary'
    cmd_args = [
        results, '--shade-transmittance-file', shade_transmittance_file,
        '--sub-folder', sub_folder
    ]

    result = runner.invoke(daylight_option_one, cmd_args)
    assert result.exit_code == 0
    assert os.path.isdir(sub_folder)
    nukedir(sub_folder, rmdir=True)
