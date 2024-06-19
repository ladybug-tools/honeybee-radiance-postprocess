"""Test Results class."""
from pathlib import Path
import numpy as np

from ladybug.futil import nukedir
from honeybee_radiance_postprocess.results.results import Results
from honeybee_radiance_postprocess.dynamic import DynamicSchedule
from honeybee_radiance_postprocess.results.annual_irradiance import AnnualIrradiance


def test_results_sample_creation():
    folder = './tests/assets/results_folders/results_irradiance'
    results = AnnualIrradiance(folder)
    assert len(results.grids_info) == 2
    assert len(results.sun_up_hours) == 4393


def test_results_sample_annual_metrics():
    folder = './tests/assets/results_folders/results_irradiance'
    results = AnnualIrradiance(folder)
    average, peak, cumulative, grids_info = results.annual_metrics()


def test_results_sample_annual_metrics_to_folder():
    folder = Path('./tests/assets/results_folders/results_irradiance')
    target_folder = Path('./tests/assets/temp')
    results = AnnualIrradiance(folder)
    results.annual_metrics_to_folder(target_folder)
    avg_folder = Path(target_folder, 'average_irradiance')
    peak_folder = Path(target_folder, 'peak_irradiance')
    cml_folder = Path(target_folder, 'cumulative_radiation')
    assert avg_folder.is_dir()
    assert peak_folder.is_dir()
    assert cml_folder.is_dir()
    nukedir(target_folder, True)


def test_results_sample_average_values_to_folder():
    folder = Path('./tests/assets/results_folders/results_irradiance')
    target_folder = Path('./tests/assets/temp')
    results = AnnualIrradiance(folder)
    results.average_values_to_folder(target_folder)
    average_values_folder = Path(target_folder, 'average_values')
    assert average_values_folder.is_dir()
    nukedir(target_folder, True)


def test_results_sample_median_values_to_folder():
    folder = Path('./tests/assets/results_folders/results_irradiance')
    target_folder = Path('./tests/assets/temp')
    results = AnnualIrradiance(folder)
    results.median_values_to_folder(target_folder)
    median_values_folder = Path(target_folder, 'median_values')
    assert median_values_folder.is_dir()
    nukedir(target_folder, True)


def test_results_sample_cumulative_values_to_folder():
    folder = Path('./tests/assets/results_folders/results_irradiance')
    target_folder = Path('./tests/assets/temp')
    results = AnnualIrradiance(folder)
    results.cumulative_values_to_folder(target_folder)
    cumulative_values_folder = Path(target_folder, 'cumulative_values')
    assert cumulative_values_folder.is_dir()
    nukedir(target_folder, True)


def test_results_sample_peak_values_to_folder():
    folder = Path('./tests/assets/results_folders/results_irradiance')
    target_folder = Path('./tests/assets/temp')
    results = AnnualIrradiance(folder)
    results.peak_values_to_folder(target_folder)
    peak_values_folder = Path(target_folder, 'peak_values')
    assert peak_values_folder.is_dir()
    nukedir(target_folder, True)


def test_results_sample_annual_data():
    folder = Path('./tests/assets/results_folders/results_irradiance')
    results = AnnualIrradiance(folder)
    data_collections, grids_info, sensor_index = results.annual_data()
    # length of data collections should be equal to grid count
    for count, grid_info in enumerate(grids_info):
        assert grid_info['count'] == len(data_collections[count])


def test_results_sample_annual_data_to_folder():
    folder = Path('./tests/assets/results_folders/results_irradiance')
    target_folder = Path('./tests/assets/temp')
    results = AnnualIrradiance(folder)
    grids_info = results.grids_info
    sensor_index = {}
    files = []
    for grid_info in grids_info:
        grid_id = grid_info['full_id']
        idx = 0
        sensor_index[grid_id] = [idx]
        files.append(Path(target_folder, 'datacollections', f'{grid_id}_{idx}.json'))
    results.annual_data_to_folder(target_folder, sensor_index=sensor_index)
    for file in files:
        assert file.is_file()
    nukedir(target_folder, True)


def test_results_sample_annual_data_to_folder_all_sensors():
    folder = Path('./tests/assets/results_folders/results_irradiance')
    target_folder = Path('./tests/assets/temp')
    results = AnnualIrradiance(folder)
    grids_info = results.grids_info
    sensor_count = sum([grid_info['count'] for grid_info in grids_info])
    grids_info = results.grids_info
    results.annual_data_to_folder(target_folder)
    datacollections_folder = Path(target_folder, 'datacollections')
    file_count = 0
    for path in datacollections_folder.iterdir():
        if path.is_file():
            file_count += 1
    assert sensor_count == file_count
    nukedir(target_folder, True)


def test_results_sun_up_hours_to_annual():
    folder = Path('./tests/assets/results_folders/results_irradiance')
    results = AnnualIrradiance(folder)
    values = np.random.rand(len(results.sun_up_hours))
    annual_array = Results.values_to_annual(
        results.sun_up_hours, values, results.timestep)
    assert annual_array.size == 8760


def test_results_sun_up_hours_to_annual_t3():
    folder = Path('./tests/assets/results_folders/results_irradiance_timestep_3')
    results = AnnualIrradiance(folder)
    values = np.random.rand(len(results.sun_up_hours))
    annual_array = Results.values_to_annual(
        results.sun_up_hours, values, results.timestep)
    assert annual_array.size == 8760 * 3


def test_results_sun_up_hours_to_annual_t4():
    folder = Path('./tests/assets/results_folders/results_irradiance_timestep_4')
    results = AnnualIrradiance(folder)
    values = np.random.rand(len(results.sun_up_hours))
    annual_array = Results.values_to_annual(
        results.sun_up_hours, values, results.timestep)
    assert annual_array.size == 8760 * 4
