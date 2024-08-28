"""Commands to work with comfort maps."""
import sys
import logging
import numpy as np
from pathlib import Path
import click
import json
import shutil
import os

from ladybug.header import Header
from ladybug.datatype.temperature import AirTemperature, \
    MeanRadiantTemperature, RadiantTemperature
from ladybug.datatype.temperaturedelta import RadiantTemperatureDelta
from ladybug.datatype.fraction import RelativeHumidity

from honeybee_radiance_postprocess.cli.grid import restore_original_distribution


_logger = logging.getLogger(__name__)


@click.group(help='Commands to work with comfort maps.')
def comfort_map():
    pass


@comfort_map.command('restructure-env-conditions')
@click.argument(
    'folder', type=click.Path(exists=True, dir_okay=True, resolve_path=True)
)
@click.argument(
    'dest-folder', type=click.Path(exists=False, dir_okay=True, resolve_path=True)
)
@click.argument(
    'sub-path', type=click.STRING
)
def restructure_env_conditions(folder, dest_folder, sub_path):
    """Read an npy file and convert every row to a data collection.

    This command reads a NumPy array from a npy file and sends it to stdout.

    \b
    Args:
        folder: Folder with environmental conditions (initial results).
        dest_folder: Destination folder for writing the results.
        sub_path: Sub path for the metric (mrt, air_temperature, longwave_mrt,
            shortwave_mrt, rel_humidity)-
    """
    try:
        source_folder = Path(folder, sub_path)
        if sub_path == 'mrt':
            source_folders = [Path(folder, 'longwave_mrt'),
                                Path(folder, 'shortwave_mrt')]
            dest_folders = [Path(folder, 'final', 'longwave_mrt'),
                            Path(folder, 'final', 'shortwave_mrt')]
        else:
            assert Path(source_folder).is_dir(), \
                'Metric "{}" does not exist for this comfort study.'.format(sub_path)
            source_folders, dest_folders = [Path(source_folder)], [Path(dest_folder)]

        # restructure the results to align with the sensor grids
        dist_info = Path(folder, '_redist_info.json')
        for src_f, dst_f in zip(source_folders, dest_folders):
            if not Path(dst_f).exists():
                Path.mkdir(dst_f, parents=True)
                restore_original_distribution(src_f, dst_f, extension='csv',
                                            dist_info=dist_info, output_extension='csv',
                                            as_text=True, fmt='%.12f', delimiter='comma')
                grid_info_src = Path(folder, 'grids_info.json')
                grid_info_dst = Path(dst_f, 'grids_info.json')
                shutil.copyfile(grid_info_src, grid_info_dst)
            data_header = create_result_header(folder, os.path.split(dst_f)[-1])
            result_info_path = Path(dst_f, 'results_info.json')
            with open(result_info_path, 'w') as fp:
                json.dump(data_header.to_dict(), fp, indent=4)
        # if MRT was requested, sum together the longwave and shortwave
        if sub_path == 'mrt':
            sum_matrices(dest_folders[0], dest_folders[1], dest_folder)
            data_header = create_result_header(folder, sub_path)
            result_info_path = os.path.join(dest_folder, 'results_info.json')
            with open(result_info_path, 'w') as fp:
                json.dump(data_header.to_dict(), fp, indent=4)
    except Exception:
        _logger.exception('Failed to restructure environmental conditions.')
        sys.exit(1)
    else:
        sys.exit(0)


def create_result_header(env_conds, sub_path):
    """Create a DataCollection Header for a given metric."""
    with open(Path(env_conds, 'results_info.json')) as json_file:
        base_head = Header.from_dict(json.load(json_file))
    if sub_path == 'mrt':
        return Header(MeanRadiantTemperature(), 'C', base_head.analysis_period)
    elif sub_path == 'air_temperature':
        return Header(AirTemperature(), 'C', base_head.analysis_period)
    elif sub_path == 'longwave_mrt':
        return Header(RadiantTemperature(), 'C', base_head.analysis_period)
    elif sub_path == 'shortwave_mrt':
        return Header(RadiantTemperatureDelta(), 'dC', base_head.analysis_period)
    elif sub_path == 'rel_humidity':
        return Header(RelativeHumidity(), '%', base_head.analysis_period)


def sum_matrices(mtxs_1, mtxs_2, dest_dir):
    """Sum together matrices of two folders."""
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    for mtx_file in os.listdir(mtxs_1):
        if mtx_file.endswith('.csv'):
            mtx_file1 = os.path.join(mtxs_1, mtx_file)
            mtx_file2 = os.path.join(mtxs_2, mtx_file)
            matrix_1 = np.loadtxt(mtx_file1, dtype=np.float32, delimiter=',')
            matrix_2 = np.loadtxt(mtx_file2, dtype=np.float32, delimiter=',')
            data = matrix_1 + matrix_2
            csv_path = os.path.join(dest_dir, mtx_file)
            np.savetxt(csv_path, data, fmt='%.12f', delimiter=',')
        elif mtx_file == 'grids_info.json':
            shutil.copyfile(
                os.path.join(mtxs_1, mtx_file),
                os.path.join(dest_dir, mtx_file)
            )
