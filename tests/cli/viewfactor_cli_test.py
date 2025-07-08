"""Test cli viewfactor module."""
import os

from click.testing import CliRunner

from ladybug.futil import nukedir

from honeybee_radiance_postprocess.cli.viewfactor import rcontrib_command_with_view_postprocess


def test_rcontrib_command_with_view_postprocess():
    runner = CliRunner()
    octree = './tests/assets/viewfactor/scene.oct'
    grid = './tests/assets/viewfactor/grid.pts'
    modifiers = './tests/assets/viewfactor/scene.mod'
    output_folder = './tests/assets/temp'
    output_file = './tests/assets/temp/view_factor.npy'
    cmd_args = [octree, grid, modifiers, '--ray-count', '6', '--folder', output_folder, '--name', 'view_factor']

    result = runner.invoke(rcontrib_command_with_view_postprocess, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(output_file)
    nukedir(output_folder)
