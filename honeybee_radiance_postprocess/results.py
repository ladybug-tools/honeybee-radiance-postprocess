import numpy as np
from pathlib import Path

from honeybee_radiance.postprocess.annual import (_process_input_folder,
    filter_schedule_by_hours, generate_default_schedule)
from .util import occupancy_filter


class _ResultsFolder(object):
    """Base class for ResultsFolder.

    This class includes properties that are independent of the results.
    
    Args:
        folder: Path to results folder.
    
    Properties:
        * folder
        * grids_info
        * sun_up_hours
        * light_paths
        * default_states
    
    """
    __slots__ = ('_folder', '_grids_info', '_sun_up_hours', '_light_paths',
                 '_default_states')

    def __init__(self, folder):
        """Initialize ResultsFolder."""
        self._folder = Path(folder).absolute().as_posix()
        self._grids_info, self._sun_up_hours = _process_input_folder(self.folder, '*')
        self._light_paths = self._load_light_paths()
        self._default_states = self._load_default_states()

    @property
    def folder(self):
        """Return full path to results folder."""
        return self._folder

    @property
    def grids_info(self):
        """Return grids information as list of dictionaries for each grid."""
        return self._grids_info
    
    @property
    def sun_up_hours(self):
        """Return sun up hours."""
        return self._sun_up_hours

    @property
    def light_paths(self):
        """Return the identifiers of the light paths."""
        return self._light_paths

    @property
    def default_states(self):
        """Return default states as a dictionary."""
        return self._default_states

    def _load_light_paths(self):
        """Find all light paths in grids_info."""
        lp = []
        for grid_info in self.grids_info:
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
                lp.insert(0, 'static_apertures')

        return lp

    def _load_default_states(self):
        """Set default state to 0 for all light paths."""
        default_states = {}
        for light_path in self.light_paths:
            default_states[light_path] = 0
        return default_states

    def __repr__(self):
        return '%s: %s' % (self.__class__.__name__, self.folder)


class Results(_ResultsFolder):
    """Results class.
    
    Args:
        folder: Path to results folder.
        schedule: 8760 values as a list. Values must be either 0 or 1. Values of 1
            indicates occupied hours. If not schedule is provided a default schedule
            will be used. (Default: None).
        load_arrays: Set to True to load all NumPy arrays. If False the arrays will be
            loaded only once they are needed. In both cases the loaded array(s) will be
            stored in a dictionary under the arrays property. (Default: False).

    Properties:
        * schedule
        * occ_pattern
        * total_occ
        * sun_down_occ_hours
        * occ_mask
        * arrays
        * valid_states
    """
    __slots__ = ('_schedule', '_occ_pattern', '_total_occ', '_sun_down_occ_hours',
                 '_occ_mask', '_arrays', '_valid_states')
    
    def __init__(self, folder, schedule=None, load_arrays=False):
        """Initialize Results."""
        _ResultsFolder.__init__(self, folder)
        self.schedule = schedule
        self._arrays = self._load_arrays() if load_arrays else dict()
        self._valid_states = self._get_valid_states()
    
    @property
    def schedule(self):
        """Return schedule."""
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        self._schedule = schedule if schedule else generate_default_schedule()
        self._update_occ()

    @property
    def occ_pattern(self):
        """Return a filtered version of the annual schedule that only includes the
        sun-up-hours."""
        return self._occ_pattern

    @property
    def total_occ(self):
        """Return an integer for the total occupied hours of the schedule."""
        return self._total_occ

    @property
    def sun_down_occ_hours(self):
        """Return an integer for the number of occupied hours where the sun is down and
        there's no possibility of being daylit or expereincing glare."""
        return self._sun_down_occ_hours
    
    @property
    def occ_mask(self):
        """Return an occupancy mask as a NumPy array that can be used to mask the
        results."""
        return self._occ_mask

    @property
    def arrays(self):
        """Return a dictionary of all the NumPy arrays that have been loaded."""
        return self._arrays

    @arrays.setter
    def arrays(self, value):
        self._arrays = value

    @property
    def valid_states(self):
        """Return a dictionary with valid states. Each light path is represented by a
        key-value pair where the light path identifier is the key and the value is a list
        of valid states, e.g., [0, 1, 2, ...]."""
        return self._valid_states

    def _update_occ(self):
        """Set properties related to occupancy."""
        sun_up_hours, schedule = [self.sun_up_hours, self.schedule]
        self._occ_pattern, self._total_occ, self._sun_down_occ_hours = \
            filter_schedule_by_hours(sun_up_hours=sun_up_hours, schedule=schedule)
        self._occ_mask = np.array(self.occ_pattern)

    def _filter_grids(self, grids_filter='*'):
        """Return grids information."""
        if grids_filter != '*':
            grids_info, _ = _process_input_folder(self.folder, grids_filter)
        else:
            grids_info = self.grids_info
        
        return grids_info

    def _load_arrays(self):
        """Load all the NumPy arrays in the results folder."""
        arrays = dict()
        grids_info = self.grids_info

        for grid_info in grids_info:
            grid_id = grid_info['identifier']
            light_paths = grid_info['light_path']
            arrays[grid_id] = dict()
            for light_path in light_paths:
                light_path = light_path[0]
                arrays[grid_id][light_path] = dict()
                light_path_folder = Path(self.folder, light_path)
                for state_folder in Path(light_path_folder).iterdir():
                    state = state_folder.name
                    arrays[grid_id][light_path][state] = dict()
                    for type_folder in Path(state_folder).iterdir():
                        type = type_folder.name
                        file = Path(type_folder, grid_id + '.npy')
                        array = np.load(file)
                        arrays[grid_id][light_path][state][type] = array

        return arrays

    def _get_valid_states(self):
        """Returns a dictionary with valid states for each light path.
        
        For each light path there will be a key (identifier of the light path) and its
        value will be a list of valid states as integers.

        Example of output format:
        {
            'static_apertures': [0],
            'Room1_North': [0, 1],
            'Room1_South': [0, 1],
            'Room2_North1': [0, 1],
            'Room2_North2': [0, 1]
        }
        """
        arrays = dict()

        for light_path in self.light_paths:
            if light_path == 'static_apertures':
                arrays[light_path] = [0]
                continue
            light_path_folder = Path(self.folder, light_path)
            for state_folder in Path(light_path_folder).iterdir():
                if light_path in arrays:
                    arrays[light_path].append(int(state_folder.stem.split('_')[0]))
                else:
                    arrays[light_path] = [int(state_folder.stem.split('_')[0])]

        return arrays
