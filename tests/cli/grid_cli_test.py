"""Test cli grid module."""
import os

from click.testing import CliRunner

from ladybug.futil import nukedir
from honeybee_radiance_postprocess.cli.grid import merge_grid_folder, \
    merge_metrics_folder


def test_merge_grid_folder():
    runner = CliRunner()
    input_folder = './tests/assets/grid/merge/input_folder'
    output_folder = './tests/assets/temp/merge_grid'
    dist_info = './tests/assets/grid/merge/dist_info.json'
    merge_file = './tests/assets/temp/merge_grid/TestRoom_1.npy'
    cmd_args = [input_folder, output_folder, 'ill', '--dist-info', dist_info]

    result = runner.invoke(merge_grid_folder, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(merge_file)
    nukedir(output_folder, rmdir=True)


def test_merge_metrics_folder():
    runner = CliRunner()
    input_folder = './tests/assets/grid/merge_metrics/input_folder'
    output_folder = './tests/assets/temp/merge_metrics'
    dist_info = './tests/assets/grid/merge_metrics/dist_info.json'
    grids_info = './tests/assets/grid/merge_metrics/grids_info.json'
    merge_file = './tests/assets/temp/merge_metrics/da/TestRoom_1.da'
    cmd_args = [input_folder, output_folder, '--dist-info', dist_info,
                '--grids-info', grids_info]

    result = runner.invoke(merge_metrics_folder, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(merge_file)
    nukedir(output_folder, rmdir=True)
