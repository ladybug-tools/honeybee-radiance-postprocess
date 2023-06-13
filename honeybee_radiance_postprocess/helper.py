"""Helper functions."""
import numpy as np

from honeybee.model import Model


def model_grid_areas(model, grids_info):
    if isinstance(model, Model):
        hb_model = model
    else:
        hb_model = Model.from_file(model)

    full_ids = [grid_info['full_id'] for grid_info in grids_info]
    sensor_grids = hb_model.properties.radiance.sensor_grids
    grid_areas = []
    for s_grid in sensor_grids:
        if s_grid.identifier in full_ids:
            if s_grid.mesh is not None:
                grid_areas.append(s_grid.mesh.face_areas)
    grid_areas = [np.array(grid) for grid in grid_areas]
    if not grid_areas:
        grid_areas = [None] * len(full_ids)

    return grid_areas
