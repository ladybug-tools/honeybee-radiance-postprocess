"""Test cli mtxop module."""
import os
import numpy as np

from click.testing import CliRunner

from ladybug.futil import nukedir

from honeybee_radiance_postprocess.cli.postprocess import annual_metrics_file


def test_annual_metrics_file():
    runner = CliRunner()
    file = './tests/assets/postprocess/annual_metrics_file/total.npy'
    sun_up_hours = './tests/assets/postprocess/annual_metrics_file/sun-up-hours.txt'
    schedule = './tests/assets/postprocess/annual_metrics_file/schedule.csv'
    output_folder = './tests/assets/temp/metrics'
    output_file = os.path.join(output_folder, 'da', 'grid.da')
    cmd_args = [
        file, sun_up_hours, '--schedule', schedule, '--sub-folder',
        output_folder, '--grid-name', 'grid'
    ]

    result = runner.invoke(annual_metrics_file, cmd_args)
    assert result.exit_code == 0
    assert os.path.isdir(output_folder)
    assert os.path.isfile(output_file)
    nukedir(output_folder, rmdir=True)
