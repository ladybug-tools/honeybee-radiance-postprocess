"""Test Results class."""
from pathlib import Path
import numpy as np

from ladybug.futil import nukedir
from honeybee_radiance_postprocess.results.results import Results
from honeybee_radiance_postprocess.dynamic import DynamicSchedule
from honeybee_radiance_postprocess.results.annual_daylight import AnnualDaylight


def test_results_sample_creation():
    folder = './tests/assets/results_folders/results_sample'
    results = AnnualDaylight(folder)
    assert len(results.grids_info) == 2
    assert len(results.sun_up_hours) == 4393


def test_results_sample_annual_metrics():
    # annual_metrics will test all five metrics as well
    folder = './tests/assets/results_folders/results_sample'
    results = AnnualDaylight(folder)
    da, cda, udi, udi_lower, udi_upper, grids_info = results.annual_metrics()


def test_results_sample_annual_metrics_static_states():
    folder = './tests/assets/results_folders/results_sample'
    results = AnnualDaylight(folder)
    light_paths = results.light_paths
    states = {}
    for light_path in light_paths:
        states[light_path] = {
            'identifier': light_path,
            'schedule': [0] * 8760
            }
    states = DynamicSchedule.from_dict(states)
    for identifier, ap_gr_sch in states.dynamic_schedule.items():
        assert ap_gr_sch.is_static == True
    da, cda, udi, udi_lower, udi_upper, grids_info = results.annual_metrics(states=states)
    # length should be two because there are two grids
    elements = [da, cda, udi, udi_lower, udi_upper, grids_info]
    assert all(len(elem) == 2 for elem in elements)


def test_results_sample_annual_metrics_dynamic_states():
    folder = './tests/assets/results_folders/results_sample'
    results = AnnualDaylight(folder)
    light_paths = results.light_paths
    states = {}
    for light_path in light_paths:
        states[light_path] = {
            'identifier': light_path,
            'schedule': [0] * 4380 + [-1] * 4380
            }
    states = DynamicSchedule.from_dict(states)
    for identifier, ap_gr_sch in states.dynamic_schedule.items():
        assert ap_gr_sch.is_static == False
    da, cda, udi, udi_lower, udi_upper, grids_info = results.annual_metrics(states=states)
    # length should be two because there are two grids
    elements = [da, cda, udi, udi_lower, udi_upper, grids_info]
    assert all(len(elem) == 2 for elem in elements)


def test_results_sample_annual_metrics_dynamic_states_off():
    # set state to -1 for all light paths
    folder = './tests/assets/results_folders/results_sample'
    results = AnnualDaylight(folder)
    light_paths = results.light_paths
    states = {}
    for light_path in light_paths:
        states[light_path] = {
            'identifier': light_path,
            'schedule': [-1] * 8760
            }
    states = DynamicSchedule.from_dict(states)
    for identifier, ap_gr_sch in states.dynamic_schedule.items():
        assert ap_gr_sch.is_static == True
    da, cda, udi, udi_lower, udi_upper, grids_info = results.annual_metrics(states=states)
    # length should be two because there are two grids
    elements = [da, cda, udi, udi_lower, udi_upper, grids_info]
    assert all(len(elem) == 2 for elem in elements)
    # all results should be zero
    assert all(np.all(array == 0) for elem in elements[:-1] for array in elem)


def test_results_sample_annual_metrics_to_folder():
    folder = Path('./tests/assets/results_folders/results_sample')
    target_folder = Path('./tests/assets/temp')
    results = AnnualDaylight(folder)
    results.annual_metrics_to_folder(target_folder)
    da_folder = Path(target_folder, 'da')
    cda_folder = Path(target_folder, 'cda')
    udi_folder = Path(target_folder, 'udi')
    udi_lower_folder = Path(target_folder, 'udi_lower')
    udi_upper_folder = Path(target_folder, 'udi_upper')
    assert da_folder.is_dir()
    assert cda_folder.is_dir()
    assert udi_folder.is_dir()
    assert udi_lower_folder.is_dir()
    assert udi_upper_folder.is_dir()
    nukedir(target_folder, True)


def test_results_sample_average_values_to_folder():
    folder = Path('./tests/assets/results_folders/results_sample')
    target_folder = Path('./tests/assets/temp')
    results = AnnualDaylight(folder)
    results.average_values_to_folder(target_folder)
    average_values_folder = Path(target_folder, 'average_values')
    assert average_values_folder.is_dir()
    nukedir(target_folder, True)


def test_results_sample_median_values_to_folder():
    folder = Path('./tests/assets/results_folders/results_sample')
    target_folder = Path('./tests/assets/temp')
    results = AnnualDaylight(folder)
    results.median_values_to_folder(target_folder)
    median_values_folder = Path(target_folder, 'median_values')
    assert median_values_folder.is_dir()
    nukedir(target_folder, True)


def test_results_sample_cumulative_values_to_folder():
    folder = Path('./tests/assets/results_folders/results_sample')
    target_folder = Path('./tests/assets/temp')
    results = AnnualDaylight(folder)
    results.cumulative_values_to_folder(target_folder)
    cumulative_values_folder = Path(target_folder, 'cumulative_values')
    assert cumulative_values_folder.is_dir()
    nukedir(target_folder, True)


def test_results_sample_peak_values_to_folder():
    folder = Path('./tests/assets/results_folders/results_sample')
    target_folder = Path('./tests/assets/temp')
    results = AnnualDaylight(folder)
    results.peak_values_to_folder(target_folder)
    peak_values_folder = Path(target_folder, 'peak_values')
    assert peak_values_folder.is_dir()
    nukedir(target_folder, True)


def test_results_sample_annual_data():
    folder = Path('./tests/assets/results_folders/results_sample')
    results = AnnualDaylight(folder)
    data_collections, grids_info, sensor_index = results.annual_data()
    # length of data collections should be equal to grid count
    for count, grid_info in enumerate(grids_info):
        assert grid_info['count'] == len(data_collections[count])


def test_results_sample_annual_data_to_folder():
    folder = Path('./tests/assets/results_folders/results_sample')
    target_folder = Path('./tests/assets/temp')
    results = AnnualDaylight(folder)
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
    folder = Path('./tests/assets/results_folders/results_sample')
    target_folder = Path('./tests/assets/temp')
    results = AnnualDaylight(folder)
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
    folder = Path('./tests/assets/results_folders/results_sample')
    results = AnnualDaylight(folder)
    values = np.random.rand(len(results.sun_up_hours))
    annual_array = Results.values_to_annual(
        results.sun_up_hours, values, results.timestep)
    assert annual_array.size == 8760
