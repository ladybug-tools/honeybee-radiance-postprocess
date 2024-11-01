"""Test cli breeam module."""
import os

from click.testing import CliRunner

from ladybug.futil import nukedir
from honeybee_radiance_postprocess.cli.breeam import breeam_4b


def test_breeam_4b():
    runner = CliRunner()
    results = './tests/assets/breeam/breeam_4b/results'
    sub_folder = './tests/assets/temp/breeam_summary'
    cmd_args = [
        results, '--sub-folder',
        sub_folder
    ]

    result = runner.invoke(breeam_4b, cmd_args)
    assert result.exit_code == 0
    assert os.path.isdir(sub_folder)
    nukedir(sub_folder, rmdir=True)
