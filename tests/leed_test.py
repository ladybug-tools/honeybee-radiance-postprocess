"""Test Results class."""
from pathlib import Path
import json

from ladybug.futil import nukedir
from honeybee_radiance_postprocess.results.annual_daylight import AnnualDaylight
from honeybee_radiance_postprocess.leed.leed import leed_option_one


def test_results_leed_option_one_shade_transmittance_file():
    folder = './tests/assets/leed/results'
    results = AnnualDaylight(folder)
    shd_file = './tests/assets/leed/shd.json'
    with open(shd_file) as json_file:
        shade_transmittance = json.load(json_file)
    target_folder = Path('./tests/assets/temp/leed_summary')
    leed_option_one(results, shade_transmittance=shade_transmittance, sub_folder=target_folder)
    assert target_folder.is_dir()
    nukedir(target_folder, True)


def test_results_leed_option_one_shade_transmittance():
    folder = './tests/assets/leed/results'
    results = AnnualDaylight(folder)
    target_folder = Path('./tests/assets/temp/leed_summary')
    leed_option_one(results, shade_transmittance=0.02, sub_folder=target_folder)
    assert target_folder.is_dir()
    nukedir(target_folder, True)
