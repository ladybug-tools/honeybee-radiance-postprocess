"""Test cli schedule module."""
import os
from click.testing import CliRunner

from ladybug.futil import nukedir
from honeybee_radiance_postprocess.cli.schedule import control_schedules


def test_control_schedules():
    runner = CliRunner()
    folder = './tests/assets/results_folders/results_sample'
    sub_folder = './tests/assets/temp/schedules'
    cmd_args = [folder, '--sub-folder', sub_folder]

    schedule_files = [
        os.path.join(sub_folder, 'control_schedules/TestRoom_1.txt'),
        os.path.join(sub_folder, 'control_schedules/TestRoom_2.txt')
    ]
    result = runner.invoke(control_schedules, cmd_args)
    assert result.exit_code == 0
    for schedule_file in schedule_files:
        assert os.path.isfile(schedule_file)
    nukedir(sub_folder, rmdir=True)
