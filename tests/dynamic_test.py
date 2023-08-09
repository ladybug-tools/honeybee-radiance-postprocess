"""Test Results class."""
import os

from ladybug.futil import nukedir
from honeybee_radiance_postprocess.dynamic import ApertureGroupSchedule, \
    DynamicSchedule


def test_aperture_group_schedule():
    schedule = [0] * 8760
    ap_schedule = ApertureGroupSchedule('ApertureGroup_1', schedule=schedule)


def test_results_dynamic_schedule():
    schedule = [0] * 8760
    ap_schedule = ApertureGroupSchedule('ApertureGroup_1', schedule=schedule)
    dyn_sch = DynamicSchedule()
    dyn_sch.add_aperture_group_schedule(ap_schedule)
    ap_schedule = ApertureGroupSchedule('ApertureGroup_2', schedule=schedule)
    dyn_sch.add_aperture_group_schedule(ap_schedule)
    if not os.path.isdir('./tests/assets/temp/'):
        os.mkdir('./tests/assets/temp/')
    dyn_sch.to_json('./tests/assets/temp/', 'dyn_sch')
    output_file = os.path.join('./tests/assets/temp/', 'dyn_sch.json')
    assert os.path.isfile(output_file)
    nukedir('./tests/assets/temp/', False)
