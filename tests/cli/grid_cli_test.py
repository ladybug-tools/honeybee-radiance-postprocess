"""Test cli grid module."""
import os

from click.testing import CliRunner

from ladybug.futil import nukedir

from honeybee_radiance_postprocess.cli.grid import merge_grid_folder


def test_merge_grid_folder():
    runner = CliRunner()
    input_folder = './tests/assets/grid/merge/input_folder'
    output_folder = './tests/assets/grid/merge/output_folder'
    dist_info = './tests/assets/grid/merge/dist_info.json'
    merge_file = './tests/assets/grid/merge/output_folder/TestRoom_1.npy'
    cmd_args = [input_folder, output_folder, 'ill', '--dist-info', dist_info]

    result = runner.invoke(merge_grid_folder, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(merge_file)
    nukedir(output_folder)
