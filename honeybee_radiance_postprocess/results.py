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
    __slots__ = (
        '_folder', '_grids_info', '_sun_up_hours', '_light_path', '_default_states')

    def __init__(self, folder):
        self._folder = pathlib.Path(folder).as_posix()
        self._grids_info, self._sun_up_hours = _process_input_folder(self.folder, '*')
        self._light_path = self._load_light_path()
        self._default_states = self._load_default_states()

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

    @property
    def default_states(self):
        return self._default_states

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
            if not light_paths and 'static_apertures' not in lp:
                lp.insert(0, light_path)

        return lp

    def _load_default_states(self):
        default_states = {}
        for light_path in self.light_path:
            default_states[light_path] = 0
        return default_states


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
       
        light_path_states = self.states(states=states)

        output_files = self.calculate_function(
            da_array2d, grids_info, light_path_states, folder=folder, 
            sub_folder=sub_folder, exists=exists, file_extension=file_extension,
            threshold=threshold)

        return output_files

    def continous_daylight_autonomy(
        self, threshold=300, states=None, grids_filter='*', folder='metrics',
        sub_folder='cda', file_extension='cda', exists=True):

        if grids_filter != '*':
            grids_info, _ = _process_input_folder(self.folder, grids_filter)
        else:
            grids_info = self.grids_info        

        light_path_states = self.states(states=states)

        output_files = self.calculate_function(
            cda_array2d, grids_info, light_path_states, folder=folder, 
            sub_folder=sub_folder, exists=exists, file_extension=file_extension,
            threshold=threshold)

        return output_files

    def useful_daylight_illuminance(
        self, min_t=100, max_t=3000, states=None, grids_filter='*', folder='metrics',
        sub_folder='udi', file_extension='udi', exists=True):

        if grids_filter != '*':
            grids_info, _ = _process_input_folder(self.folder, grids_filter)
        else:
            grids_info = self.grids_info

        light_path_states = self.states(states=states)

        output_files = self.calculate_function(
            udi_array2d, grids_info, light_path_states, folder=folder,
            sub_folder=sub_folder, exists=exists, file_extension=file_extension,
            min_t=min_t, max_t=max_t)

        return output_files

    def useful_daylight_illuminance_lower(
        self, min_t=100, states=None, grids_filter='*', folder='metrics',
        sub_folder='udi_lower', file_extension='udi', exists=True):

        if grids_filter != '*':
            grids_info, _ = _process_input_folder(self.folder, grids_filter)
        else:
            grids_info = self.grids_info

        light_path_states = self.states(states=states)

        output_files = self.calculate_function(
            udi_lower_array2d, grids_info, light_path_states, folder=folder,
            sub_folder=sub_folder, exists=exists, file_extension=file_extension,
            min_t=min_t, sun_down_occ_hours=self.sun_down_occ_hours)

        return output_files

    def useful_daylight_illuminance_upper(
        self, max_t=3000, states=None, grids_filter='*', folder='metrics',
        sub_folder='udi_upper', file_extension='udi', exists=True):

        if grids_filter != '*':
            grids_info, _ = _process_input_folder(self.folder, grids_filter)
        else:
            grids_info = self.grids_info

        light_path_states = self.states(states=states)

        output_files = self.calculate_function(
            udi_upper_array2d, grids_info, light_path_states, folder=folder,
            sub_folder=sub_folder, exists=exists, file_extension=file_extension,
            max_t=max_t)

        return output_files

    def annual_metrics(
        self, threshold=300, min_t=100, max_t=3000, states=None, grids_filter='*',
        folder='metrics', exists=True):

        da_files = self.daylight_autonomy(
            threshold=threshold, states=states, grids_filter=grids_filter, folder=folder,
            exists=exists)
        cda_files = self.continous_daylight_autonomy(
            threshold=threshold, states=states, grids_filter=grids_filter, folder=folder,
            exists=exists)
        udi_files = self.useful_daylight_illuminance(
            min_t=min_t, max_t=max_t, states=states, grids_filter=grids_filter,
            folder=folder, exists=exists)
        udi_lower_files = self.useful_daylight_illuminance_lower(
            min_t=min_t, states=states, grids_filter=grids_filter, folder=folder,
            exists=exists)
        udi_upper_files = self.useful_daylight_illuminance_upper(
            max_t=max_t, states=states, grids_filter=grids_filter, folder=folder,
            exists=exists)
        
        return da_files, cda_files, udi_files, udi_lower_files, udi_upper_files

    def calculate_function(
        self, function, grids_info, light_path_states, folder='metrics',
        sub_folder='unnamed', file_extension='txt', exists=True, **kwargs):

        output_files = []
        for grid_info in grids_info:
            grid_id = grid_info['identifier']
            files, states = self.grid_files_and_states(grid_info, light_path_states)

            file = pathlib.Path(
                self.folder, folder, sub_folder, 
                '%s..%s.%s' % (grid_id, '_'.join(map(str, states)), file_extension))
            file.parent.mkdir(parents=True, exist_ok=True)

            output_files.append(file)
            if file.exists() and exists:
                continue
            if files:
                array = sum(Results.load_numpy_arrays(files))
                array_filter = np.apply_along_axis(
                    occupancy_filter, 1, array, mask=self.occupancy_mask)
                results = function(array_filter, total_occ=self.total_occ, **kwargs)
                np.savetxt(file, results, fmt='%.2f')
            else:
                # all states for the grid are -1
                results = np.zeros(grid_info['count'])
                np.savetxt(file, results, fmt='%.2f')

        return output_files

    @staticmethod
    def load_numpy_arrays(files):
        """Load a list of NumPy files to a list of NumPy arrays.

        Args:
            files: A list of NumPy files.
        
        Returns:
            A list of NumPy arrays.
        """
        arrays = [np.load(file) for file in files]
        return arrays

    def grid_files_and_states(self, grid_info, light_path_states):
        """Get NumPy files for a given grid as well as a list of states for the grid."""
        files = []
        states = []
        grid_id = grid_info['identifier']
        light_paths = grid_info['light_path']
        for light_path in light_paths:
            light_path = light_path[0]

            if light_path == 'static_apertures':
                if light_path_states[light_path] == 0:
                    states.append(light_path_states[light_path])
                    npy_file = os.path.join(self.folder, light_path, '%s.npy' % grid_id)
                    files.append(npy_file)
                elif light_path_states[light_path] == -1:
                    states.append(-1)
                else:
                    raise ValueError('State of static apertures must be either 0 for on '
                        'or -1 for off. Received state \'%s\'' % light_path_states[light_path])
            else:
                state = light_path_states[light_path]
                states.append(state)
                if state == -1:
                    continue
                state_folder = os.path.join(self.folder, '%s_%s' % (state, light_path))
                if not os.path.isdir(state_folder):
                    raise ValueError('Folder of state %s for light path %s does not exist.' % (state, light_path))
                npy_file = os.path.join(state_folder, '%s.npy' % grid_id)
                files.append(npy_file)
        
        return files, states

    def states(self, states=None):
        """Creates a dictionary of states for each light path including a state for
        static results.
        
        This method uses the default_states and overwrites the values in that
        dictionary by the values provided in the 'states' input. If some light paths are
        not given in the 'states' input, the default values of 0 will be used.

        Example of 'states' input:
        {
            'static_apertures': 0,
            'Room_1_South': 1,
            'Room_2_North': -1
        }

        Args:
            states: A dictionary of states. If no input is given the method will return
                the same dictionary that can be accessed by the default_states property.

        
        Returns:
            A dictionary of states.
        """
        if states:
            assert isinstance(states, dict), 'Expected dictionary ...'
            light_path_states = self.default_states
            for lp, state in states.items():
                if lp in self.default_states:
                    light_path_states[lp] = state
                else:
                    raise ValueError('States dictionary has an invalid key \'%s\'. '
                        'Valid keys are [%s].' % (lp, ', '.join(self.light_path)))
        else:
            light_path_states = self.default_states
        
        return light_path_states

    def _update_schedule(self, sun_up_hours, schedule):
        self._occ_pattern, self._total_occ, self._sun_down_occ_hours = \
            filter_schedule_by_hours(sun_up_hours=sun_up_hours, schedule=schedule)
        self._occupancy_mask = np.array(self.occ_pattern)
    
    def _update_grids(self, filter_pattern):
        self.grids_info, self._sun_up_hours = _process_input_folder(self.folder, filter_pattern)
