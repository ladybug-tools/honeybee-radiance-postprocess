"""Commands for ABNT NBR post-processing."""
import json
import sys
import logging
from pathlib import Path
import click
import numpy as np

from honeybee.model import Model
from ladybug_geometry.geometry3d.face import Face3D

from ..vis_metadata import _abnt_nbr_15575_daylight_levels_vis_metadata


_logger = logging.getLogger(__name__)


@click.group(help='Commands for ABNT NBR (Brazil) post-processing of Radiance results.')
def abnt():
    pass


@abnt.command('abnt-nbr-15575')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.argument('model-file', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--sub-folder', '-sf', help='Relative path for subfolder to write output '
    'files.', default='abnt_nbr_15575', type=click.Path(
    exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path)
)
def abnt_nbr_15575(
    folder, model_file, sub_folder
):
    """Calculate metrics for ABNT NBR 15575.

    \b
    Args:
        folder: Simulation folder for a ABNT NBR 15575 simulation. It should
            contain four sub-folder of complete point-in-time illuminance
            simulations labeled "4_930AM", "4_330PM", "10_930AM", and "10_330PM".
            These sub-folder should each have results folder that include a
            grids_info.json and .res files with illuminance values for each
            sensor.
        model-file: A Honeybee Model file that was used in the simulations. This
            file is used to extract the center points of the sensor grids. It is
            a requirement that the sensor grids have Meshes.
    """
    def find_surrounding_points(x, y, x_coords, y_coords):
        x1 = np.max(x_coords[x_coords <= x]) if np.any(x_coords <= x) else x_coords[0]
        x2 = np.min(x_coords[x_coords > x]) if np.any(x_coords > x) else x_coords[-1]
        y1 = np.max(y_coords[y_coords <= y]) if np.any(y_coords <= y) else y_coords[0]
        y2 = np.min(y_coords[y_coords > y]) if np.any(y_coords > y) else y_coords[-1]
        return x1, x2, y1, y2

    def get_value(x, y, x_coords, y_coords, values):
        tolerance = 0.001
        index = np.where((np.abs(x_coords - x) <= tolerance) & (np.abs(y_coords - y) <= tolerance))
        return values[index][0]

    def perform_interpolation(x, y, x_coords, y_coords, pit_values):
        x1, x2, y1, y2 = find_surrounding_points(x, y, x_coords, y_coords)

        # extract the illuminance values at the surrounding points
        f_Q11 = get_value(x1, y1, x_coords, y_coords, pit_values) # bottom-left
        f_Q21 = get_value(x2, y1, x_coords, y_coords, pit_values) # bottom-right
        f_Q12 = get_value(x1, y2, x_coords, y_coords, pit_values) # top-left
        f_Q22 = get_value(x2, y2, x_coords, y_coords, pit_values) # top-right

        # edge cases
        if x == x1 and y == y1:
            f_xy = f_Q11
        elif x == x2 and y == y1:
            f_xy = f_Q21
        elif x == x1 and y == y2:
            f_xy = f_Q12
        elif x == x2 and y == y2:
            f_xy = f_Q22
        elif x1 == x2:
            # linear interpolation in y direction
            f_xy = f_Q11 + (f_Q12 - f_Q11) * (y - y1) / (y2 - y1)
        elif y1 == y2:
            # linear interpolation in x direction
            f_xy = f_Q11 + (f_Q21 - f_Q11) * (x - x1) / (x2 - x1)
        else:
            # perform bilinear interpolation
            f_xy = (f_Q11 * (x2 - x) * (y2 - y) +
                    f_Q21 * (x - x1) * (y2 - y) +
                    f_Q12 * (x2 - x) * (y - y1) +
                    f_Q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))

        return f_xy

    try:
        folder = Path(folder)
        hb_model = Model.from_file(model_file)
        sensor_grids = hb_model.properties.radiance.sensor_grids
        sg_full_identifier = {sg.full_identifier: sg for sg in sensor_grids}

        if not sub_folder.exists():
            sub_folder.mkdir(parents=True, exist_ok=True)
        illuminance_levels_folder = sub_folder.joinpath('illuminance_levels')
        if not illuminance_levels_folder.exists():
            illuminance_levels_folder.mkdir(parents=True, exist_ok=True)

        summary_file = sub_folder.joinpath('abnt_nbr_15575.json')
        summary_rooms_file = sub_folder.joinpath('abnt_nbr_15575_rooms.json')
        folder_names = ['4_930AM', '4_330PM', '10_930AM', '10_330PM']

        metric_info_dict = _abnt_nbr_15575_daylight_levels_vis_metadata()
        summary_output = {}
        summary_rooms_output = {}
        pof_sensor_grids = {}
        for _subfolder in folder_names:
            res_folder = folder.joinpath(_subfolder, 'results')
            with open(res_folder.joinpath('grids_info.json')) as data_f:
                grids_info = json.load(data_f)
            sub_output = []
            for grid_info in grids_info:
                pit_values = \
                    np.loadtxt(res_folder.joinpath(f'{grid_info["full_id"]}.res'))
                sensor_grid = sg_full_identifier[grid_info['full_id']]
                sensor_points = np.array(
                    [[sensor.pos[0], sensor.pos[1]] for sensor in sensor_grid.sensors])

                x_coords = sensor_points[:, 0]
                y_coords = sensor_points[:, 1]

                pof_sensor_grid = \
                    pof_sensor_grids.get(grid_info['full_id'], None)
                # if pof is not calculated for this grid
                if pof_sensor_grid is None:
                    faces_3d = [Face3D(face_vertices) for face_vertices in sensor_grid.mesh.face_vertices]
                    face_3d_union = Face3D.join_coplanar_faces(faces_3d, 0.05)
                    assert len(face_3d_union) == 1
                    if face_3d_union[0].is_convex:
                        centroid = face_3d_union[0].centroid
                        pof_sensor_grids[grid_info['full_id']] = (centroid.x, centroid.y)
                    else:
                        pof = face_3d_union[0].pole_of_inaccessibility(0.01)
                        pof_sensor_grids[grid_info['full_id']] = (pof.x, pof.y)

                x, y = pof_sensor_grids[grid_info['full_id']]
                f_xy = perform_interpolation(x, y, x_coords, y_coords, pit_values)

                if f_xy >= 120:
                    level = 'Superior'
                elif f_xy >= 90:
                    level = 'Intermediário'
                elif f_xy >= 60: # add check for ground floor (48 lux)
                    level = 'Mínimo'
                else:
                    level = 'Não atende'

                room_summary = \
                    summary_rooms_output.get(grid_info['full_id'], None)
                if room_summary is None:
                    summary_rooms_output[grid_info['full_id']] = {
                        'nível': level,
                        'iluminância': f_xy,
                        'grids_info': grid_info
                    }
                else:
                    if f_xy < room_summary['iluminância']:
                        room_summary['nível'] = level
                        room_summary['iluminância'] = f_xy

                sub_output.append(
                    {
                        'nível': level,
                        'iluminância': f_xy,
                        'grids_info': grid_info
                    }
                )

                conditions = [pit_values >= 120, pit_values >= 90, pit_values >= 60, pit_values < 60]
                conditions_values = [3, 2, 1, 0]
                illuminance_level = np.select(conditions, conditions_values)

                ill_level_file = illuminance_levels_folder.joinpath(_subfolder, f'{grid_info["full_id"]}.res')
                ill_level_file.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(ill_level_file, illuminance_level, fmt='%d')

                grids_info_file = illuminance_levels_folder.joinpath(_subfolder, 'grids_info.json')
                grids_info_file.write_text(json.dumps(grids_info, indent=2))

                vis_data = metric_info_dict[_subfolder]
                vis_metadata_file = illuminance_levels_folder.joinpath(_subfolder, 'vis_metadata.json')
                vis_metadata_file.write_text(json.dumps(vis_data, indent=4))

            summary_output[_subfolder] = sub_output

            grids_info_file = folder.joinpath(_subfolder, 'grids_info.json')
            grids_info_file.write_text(json.dumps(grids_info, indent=2))

        with summary_file.open(mode='w', encoding='utf-8') as output_file:
            json.dump(summary_output, output_file, indent=4, ensure_ascii=False)

        with summary_rooms_file.open(mode='w', encoding='utf-8') as output_file:
            json.dump(summary_rooms_output, output_file, indent=4, ensure_ascii=False)

    except Exception:
        _logger.exception('Failed to calculate ABNT NBR 15575 metrics.')
        sys.exit(1)
    else:
        sys.exit(0)
