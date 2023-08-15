"""Test cli postprocess module."""
from pathlib import Path

from click.testing import CliRunner

from ladybug.futil import nukedir

from honeybee_radiance_postprocess.dynamic import DynamicSchedule, \
    ApertureGroupSchedule
from honeybee_radiance_postprocess.cli.postprocess import annual_metrics, \
    annual_metrics_file, peak_values


def test_annual_metrics():
    runner = CliRunner()
    folder = './tests/assets/results_folders/results_sample'
    subfolder = Path('./tests/assets/temp/metrics')
    cmd_args = [
        folder, '--sub-folder', subfolder
    ]
    result = runner.invoke(annual_metrics, cmd_args)
    assert result.exit_code == 0
    assert subfolder.is_dir()
    nukedir(subfolder, rmdir=True)


def test_annual_metrics_with_dyn_sch():
    runner = CliRunner()
    folder = './tests/assets/results_folders/results_sample'
    subfolder = Path('./tests/assets/temp/metrics')
    dyn_sch = DynamicSchedule()
    ag_sch = ApertureGroupSchedule('Room1_South', [0] * 4380 + [1] * 4380)
    dyn_sch.add_aperture_group_schedule(ag_sch)
    dyn_sch_file = Path(dyn_sch.to_json('./tests/assets/temp'))
    cmd_args = [
        folder, '--states', dyn_sch_file, '--sub-folder', subfolder
    ]
    result = runner.invoke(annual_metrics, cmd_args)
    assert result.exit_code == 0
    assert subfolder.is_dir()
    dyn_sch_file.unlink()
    nukedir(subfolder, rmdir=True)


def test_annual_metrics_file():
    runner = CliRunner()
    file = './tests/assets/postprocess/annual_metrics_file/total.npy'
    sun_up_hours = './tests/assets/postprocess/annual_metrics_file/sun-up-hours.txt'
    schedule = './tests/assets/postprocess/annual_metrics_file/schedule.csv'
    output_folder = Path('./tests/assets/temp/metrics')
    output_file = Path(output_folder, 'da', 'grid.da')
    cmd_args = [
        file, sun_up_hours, '--schedule', schedule, '--sub-folder',
        output_folder, '--grid-name', 'grid'
    ]

    result = runner.invoke(annual_metrics_file, cmd_args)
    assert result.exit_code == 0
    assert output_folder.is_dir()
    assert output_file.is_file()
    nukedir(output_folder, rmdir=True)


def test_peak_values():
    runner = CliRunner()
    folder = './tests/assets/results_folders/results_sample'
    hoys_file = './tests/assets/util/hoys_august.txt'
    output_folder = Path('./tests/assets/temp/metrics')
    cmd_args = [
        folder, '--hoys-file', hoys_file, '--sub-folder', output_folder,
    ]

    result = runner.invoke(peak_values, cmd_args)
    assert result.exit_code == 0
    assert output_folder.joinpath('peak_values').is_dir()
    nukedir(output_folder, rmdir=True)


def test_peak_values_coincident():
    runner = CliRunner()
    folder = './tests/assets/results_folders/results_sample'
    hoys_file = './tests/assets/util/hoys_august.txt'
    output_folder = Path('./tests/assets/temp/metrics')
    max_hoys_file = Path('./tests/assets/temp/metrics/peak_values/max_hoys.txt')
    cmd_args = [
        folder, '--hoys-file', hoys_file, '--sub-folder', output_folder,
        '--coincident'
    ]

    result = runner.invoke(peak_values, cmd_args)
    assert result.exit_code == 0
    assert output_folder.joinpath('peak_values').is_dir()
    assert max_hoys_file.is_file()
    nukedir(output_folder, rmdir=True)
