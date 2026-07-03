"""Test cli well module."""
import os

from click.testing import CliRunner

from ladybug.futil import nukedir
from honeybee_radiance_postprocess.cli.well import well_daylight


def test_well_daylight():
    runner = CliRunner()
    results = './tests/assets/well/results'
    daylight_hours = './tests/assets/well/daylight_hours.csv'
    sub_folder = './tests/assets/temp/well_summary'
    cmd_args = [
        results, daylight_hours, '--sub-folder', sub_folder
    ]

    result = runner.invoke(well_daylight, cmd_args)
    assert result.exit_code == 0
    assert os.path.isdir(sub_folder)
    nukedir(sub_folder, rmdir=True)
