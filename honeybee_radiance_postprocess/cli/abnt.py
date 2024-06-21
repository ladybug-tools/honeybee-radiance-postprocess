"""Commands for ABNT NBR post-processing."""
import json
import sys
import logging
from pathlib import Path
import click
import numpy as np

from honeybee.model import Model
from honeybee.room import Room
from ladybug_geometry.geometry3d.face import Face3D
from ladybug_geometry.geometry3d.pointvector import Vector3D

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
    '--ground-level', '-gl', help='A value to define the height of the ground '
    'level. This will make sure that rooms below this height will not be '
    'counted as ground level rooms.',
    default=0, show_default=True, type=click.FLOAT
)
@click.option(
    '--room-center/--grid-center', '-rc/-gc', help='Flag to note whether the '
    'evaluation of the center is at the room center or the grid center.',
    default=True, show_default=True)
@click.option(
    '--sub-folder', '-sf', help='Relative path for subfolder to write output '
    'files.', default='abnt_nbr_15575', type=click.Path(
    exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path)
)
def abnt_nbr_15575(
    folder, model_file, ground_level, room_center, sub_folder
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
        hb_model: Model = Model.from_file(model_file)
        grouped_rooms, floor_heights = Room.group_by_floor_height(hb_model.rooms)

        # pick the first group >= to ground level
        for gr, fh in zip(grouped_rooms, floor_heights):
            if fh >= ground_level:
                ground_level_rooms = gr
                break

        sensor_grids = hb_model.properties.radiance.sensor_grids
        sg_full_identifier = {sg.full_identifier: sg for sg in sensor_grids}

        if not sub_folder.exists():
            sub_folder.mkdir(parents=True, exist_ok=True)
        illuminance_levels_folder = sub_folder.joinpath('illuminance_levels')
        if not illuminance_levels_folder.exists():
            illuminance_levels_folder.mkdir(parents=True, exist_ok=True)

        summary_rooms_csv = sub_folder.joinpath('abnt_nbr_15575_rooms.csv')
        folder_names = ['4_930AM', '4_330PM', '10_930AM', '10_330PM']
        pit_mapper = {
            '4_930AM': 'Abril 09:30',
            '4_330PM': 'Abril 15:30',
            '10_930AM': 'Outubro 09:30',
            '10_330PM': 'Outubro 15:30'
        }

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

                room = hb_model.rooms_by_identifier([sensor_grid.room_identifier])[0]

                pof_sensor_grid = \
                    pof_sensor_grids.get(grid_info['full_id'], None)
                # if pof is not calculated for this grid
                if pof_sensor_grid is None:
                    if room_center:
                        floor_face = Face3D.join_coplanar_faces(
                            room.horizontal_floor_boundaries(), 0.05
                        )[0]
                        if floor_face.is_convex:
                            pof_sensor_grids[grid_info['full_id']] = \
                                floor_face.centroid + Vector3D(0, 0, 0.75)
                        else:
                            pof_sensor_grids[grid_info['full_id']] = \
                                floor_face.pole_of_inaccessibility(0.01) + Vector3D(0, 0, 0.75)
                    else:
                        faces_3d = [Face3D(face_vertices) for face_vertices in sensor_grid.mesh.face_vertices]
                        face_3d_union = Face3D.join_coplanar_faces(faces_3d, 0.05)
                        assert len(face_3d_union) == 1
                        if face_3d_union[0].is_convex:
                            centroid = face_3d_union[0].centroid
                            pof_sensor_grids[grid_info['full_id']] = centroid
                        else:
                            pof = face_3d_union[0].pole_of_inaccessibility(0.01)
                            pof_sensor_grids[grid_info['full_id']] = pof

                x = pof_sensor_grids[grid_info['full_id']].x
                y = pof_sensor_grids[grid_info['full_id']].y
                f_xy = perform_interpolation(x, y, x_coords, y_coords, pit_values)

                if room in ground_level_rooms:
                    minimo = 48
                else:
                    minimo = 60

                if f_xy >= 120:
                    level = 'Superior'
                elif f_xy >= 90:
                    level = 'Intermediário'
                elif f_xy >= minimo: # add check for ground floor (48 lux)
                    level = 'Mínimo'
                else:
                    level = 'Não atende'

                room_summary = \
                    summary_rooms_output.get(grid_info['full_id'], None)
                if room_summary is None:
                    summary_rooms_output[grid_info['full_id']] = {
                        'nível': level,
                        'iluminância': f_xy,
                        'grids_info': grid_info,
                        pit_mapper[_subfolder]: f_xy,
                    }
                else:
                    if f_xy < room_summary['iluminância']:
                        room_summary['nível'] = level
                        room_summary['iluminância'] = f_xy
                    room_summary[pit_mapper[_subfolder]] = f_xy

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

        # set up the default data types
        dtype = [
            ('Sensor Grid', 'O'),
            ('Sensor Grid ID', 'O'),
            ('Abril 09:30', np.float32),
            ('Abril 15:30', np.float32),
            ('Outubro 09:30', np.float32),
            ('Outubro 15:30', np.float32),
            ('Atendimento', 'O')
        ]

        # set up format
        fmt = ['%s', '%s', '%.2f', '%.2f', '%.2f', '%.2f', '%s']

        arrays = []
        for room_summary in summary_rooms_output.values():
            data = []
            data.append(room_summary['grids_info']['name'])
            data.append(room_summary['grids_info']['full_id'])
            data.append(room_summary['Abril 09:30'])
            data.append(room_summary['Abril 15:30'])
            data.append(room_summary['Outubro 09:30'])
            data.append(room_summary['Outubro 15:30'])
            data.append(room_summary['nível'])
            arrays.append(tuple(data))

        # create structured array
        struct_array = np.array(arrays, dtype=dtype)

        header = [dt[0] for dt in dtype]
        # write header to summary_rooms_csv
        with summary_rooms_csv.open(mode='w', encoding='utf-8') as output_file:
            output_file.write(','.join(header))
            output_file.write('\n') # add newline after header

        # write structured array to summary_rooms_csv
        with summary_rooms_csv.open(mode='a', encoding='utf-8') as output_file:
            np.savetxt(output_file, struct_array, delimiter=',', fmt=fmt)

        center_points_file = sub_folder.joinpath('center_points.json')
        data = [pof.to_dict() for pof in pof_sensor_grids.values()]
        center_points_file.write_text(json.dumps(data, indent=4))

    except Exception:
        _logger.exception('Failed to calculate ABNT NBR 15575 metrics.')
        sys.exit(1)
    else:
        sys.exit(0)
