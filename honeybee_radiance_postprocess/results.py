import pathlib
import json
import os
import numpy as np

from honeybee_radiance.postprocess.annual import (_process_input_folder,
    filter_schedule_by_hours, generate_default_schedule)
from honeybee_radiance_postprocess.metrics import (da_array2d, cda_array2d, udi_array2d,
    udi_lower_array2d, udi_upper_array2d)
from honeybee_radiance_postprocess.annualdaylight import occupancy_filter


class _ResultsFolder(object):
    __slots__ = ('_folder', '_grids_info', '_sun_up_hours', '_light_path')

    def __init__(self, folder):
        self._folder = pathlib.Path(folder).as_posix()
        self._grids_info, self._sun_up_hours = _process_input_folder(self.folder, '*')
        self._light_path = self._load_light_path()

    @property
    def folder(self):
        return self._folder

    @property
    def grids_info(self):
        return self._grids_info

    @grids_info.setter
    def grids_info(self, grids_info):
        self._grids_info = grids_info
    
    @property
    def sun_up_hours(self):
        return self._sun_up_hours

    @property
    def light_path(self):
        return self._light_path
    
    def _load_light_path(self):
        grids_info = self.grids_info
        lp = []
        for grid_info in grids_info:
            light_paths = grid_info['light_path']
            for light_path in light_paths:
                light_path = light_path[0]
                if light_path in lp:
                    continue
                if light_path == 'static_apertures':
                    lp.insert(0, light_path)
                else:
                    lp.append(light_path)
        return lp


class Results(_ResultsFolder):

    __slots__ = ('_schedule', '_occ_pattern', '_total_occ', '_sun_down_occ_hours', '_occupancy_mask')
    
    def __init__(self, folder, schedule=None):
        _ResultsFolder.__init__(self, folder)
        self.schedule = schedule
    
    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        self._schedule = schedule or generate_default_schedule()
        self._update_schedule(self.sun_up_hours, self.schedule)

    @property
    def occ_pattern(self):
        return self._occ_pattern

    @property
    def total_occ(self):
        return self._total_occ

    @property
    def sun_down_occ_hours(self):
        return self._sun_down_occ_hours
    
    @property
    def occupancy_mask(self):
        return self._occupancy_mask

    def daylight_autonomy(
        self, threshold=300, states=None, grids_filter='*', folder='metrics',
        sub_folder='da', file_extension='da', exists=True):

        if grids_filter != '*':
            grids_info, _ = _process_input_folder(self.folder, grids_filter)
        else:
            grids_info = self.grids_info
       
        light_path_states = self.get_light_path_states(states=states)

        self.calculate_function(da_array2d, grids_info, light_path_states, folder=folder, 
                                sub_folder=sub_folder, exists=exists, 
                                file_extension=file_extension, threshold=threshold)

        return 'a'

    def continous_daylight_autonomy(
        self, threshold=300, states=None, grids_filter='*', folder='metrics',
        sub_folder='cda', file_extension='cda', exists=True):

        if grids_filter != '*':
            grids_info, _ = _process_input_folder(self.folder, grids_filter)
        else:
            grids_info = self.grids_info        

        light_path_states = self.get_light_path_states(states=states)

        self.calculate_function(cda_array2d, grids_info, light_path_states, folder=folder, 
                                sub_folder=sub_folder, exists=exists, 
                                file_extension=file_extension, threshold=threshold)

        return 'a'

    def useful_daylight_illuminance(
        self, min_t=100, max_t=3000, states=None, grids_filter='*', folder='metrics',
        sub_folder='udi', file_extension='udi', exists=True):

        if grids_filter != '*':
            grids_info, _ = _process_input_folder(self.folder, grids_filter)
        else:
            grids_info = self.grids_info

        light_path_states = self.get_light_path_states(states=states)

        self.calculate_function(udi_array2d, grids_info, light_path_states,
                                folder=folder, sub_folder=sub_folder, exists=exists, 
                                file_extension=file_extension, min_t=min_t, max_t=max_t)

        return 'a'

    def useful_daylight_illuminance_lower(
        self, min_t=100, states=None, grids_filter='*', folder='metrics',
        sub_folder='udi_lower', file_extension='udi', exists=True):

        if grids_filter != '*':
            grids_info, _ = _process_input_folder(self.folder, grids_filter)
        else:
            grids_info = self.grids_info

        light_path_states = self.get_light_path_states(states=states)

        self.calculate_function(udi_lower_array2d, grids_info, light_path_states,
                                folder=folder, sub_folder=sub_folder, exists=exists, 
                                file_extension=file_extension, min_t=min_t,
                                sun_down_occ_hours=self.sun_down_occ_hours)

        return 'a'

    def useful_daylight_illuminance_upper(
        self, max_t=3000, states=None, grids_filter='*', folder='metrics',
        sub_folder='udi_upper', file_extension='udi', exists=True):

        if grids_filter != '*':
            grids_info, _ = _process_input_folder(self.folder, grids_filter)
        else:
            grids_info = self.grids_info

        light_path_states = self.get_light_path_states(states=states)

        self.calculate_function(udi_upper_array2d, grids_info, light_path_states,
                                folder=folder, sub_folder=sub_folder, exists=exists, 
                                file_extension=file_extension, max_t=max_t)

        return 'a'

    def calculate_function(
        self, function, grids_info, light_path_states, folder='metrics',
        sub_folder='unnamed', file_extension='txt', exists=True, **kwargs):

        for grid_info in grids_info:
            grid_id = grid_info['identifier']
            arrays, states = self.grid_arrays_and_states(grid_info, light_path_states)

            file = pathlib.Path(
                self.folder, folder, sub_folder, 
                '%s..%s.%s' % (grid_id, '_'.join(map(str, states)), file_extension))
            file.parent.mkdir(parents=True, exist_ok=True)

            if file.exists() and exists:
                continue
            if arrays:
                array = sum(arrays)
                array_filter = np.apply_along_axis(
                    occupancy_filter, 1, array, mask=self.occupancy_mask)
                results = function(array_filter, total_occ=self.total_occ, **kwargs)
                np.savetxt(file, results, fmt='%.2f')
            else:
                # all states for the grid are -1
                results = np.zeros(grid_info['count'])
                np.savetxt(file, results, fmt='%.2f')
            print(results[:5])

    def grid_arrays_and_states(self, grid_info, light_path_states):
        """Get all arrays for the given grid as well as a list of states for the grid."""
        arrays = []
        states = []
        grid_id = grid_info['identifier']
        light_paths = grid_info['light_path']
        for light_path in light_paths:
            light_path = light_path[0]

            if light_path == 'static_apertures':
                if light_path_states[light_path] == 0:
                    states.append(light_path_states[light_path])
                    npy_file = os.path.join(self.folder, light_path, '%s.npy' % grid_id)
                    arrays.append(np.load(npy_file))
                else:
                    states.append(-1)
            else:
                state = light_path_states[light_path]
                states.append(state)
                if state == -1:
                    continue
                state_folder = os.path.join(self.folder, '%s_%s' % (state, light_path))
                if not os.path.isdir(state_folder):
                    raise ValueError('Folder of state %s for light path %s does not exist.' % (state, light_path))
                npy_file = os.path.join(state_folder, '%s.npy' % grid_id)
                arrays.append(np.load(npy_file))
        
        return arrays, states

    def get_light_path_states(self, states=None):
        light_path_states = {}
        if states:
            for lp, state in zip(self.light_path, states):
                light_path_states[lp] = state
        else:
            for lp in self.light_path:
                light_path_states[lp] = 0
        
        return light_path_states

    def _update_schedule(self, sun_up_hours, schedule):
        self._occ_pattern, self._total_occ, self._sun_down_occ_hours = \
            filter_schedule_by_hours(sun_up_hours=sun_up_hours, schedule=schedule)
        self._occupancy_mask = np.array(self.occ_pattern)
    
    def _update_grids(self, filter_pattern):
        self.grids_info, self._sun_up_hours = _process_input_folder(self.folder, filter_pattern)
