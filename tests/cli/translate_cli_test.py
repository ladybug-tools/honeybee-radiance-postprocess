"""Test cli tranlate module."""
import os

from click.testing import CliRunner

from ladybug.futil import nukedir

from honeybee_radiance_postprocess.cli.translate import npy_to_txt


def test_npy_to_txt():
    runner = CliRunner()
    npy_file = './tests/assets/npy/Test_1.npy'
    output_folder = './tests/assets/temp'
    output = os.path.join('./tests/assets/temp/Test_1')
    cmd_args = [npy_file, '--name', output, '--extension', '.txt']

    result = runner.invoke(npy_to_txt, cmd_args)
    assert result.exit_code == 0
    assert os.path.isfile(output + '.txt')
    nukedir(output_folder)
