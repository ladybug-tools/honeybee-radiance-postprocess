"""Test cli postprocess module."""
from pathlib import Path

from click.testing import CliRunner

from ladybug.futil import nukedir

from honeybee_radiance_postprocess.cli.two_phase import rgb_to_illuminance, \
    rgb_to_illuminance_file, add_remove_sky_matrix


def test_annual_metrics():
    runner = CliRunner()
    total_mtx = './tests/assets/binary/sky.ill'
    direct_mtx = './tests/assets/binary/sky_dir.ill'
    direct_sunlight_mtx = './tests/assets/binary/sun.ill'
    output_folder = Path('./tests/assets/temp')
    total_file = output_folder.joinpath('total.npy')
    direct_file = output_folder.joinpath('direct.npy')
    cmd_args = [
        total_mtx, direct_mtx, direct_sunlight_mtx, '--output-folder', output_folder
    ]
    result = runner.invoke(rgb_to_illuminance, cmd_args)
    assert result.exit_code == 0
    assert total_file.exists()
    assert direct_file.exists()
    nukedir(output_folder, rmdir=False)
