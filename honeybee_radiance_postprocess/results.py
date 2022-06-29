import numpy as np
import json
from pathlib import Path
from itertools import islice, cycle
from typing import Union

from honeybee_radiance.postprocess.annual import (_process_input_folder,
    filter_schedule_by_hours, generate_default_schedule)
from honeybee_radiance_postprocess.metrics import (da_array2d, cda_array2d, udi_array2d,
    udi_lower_array2d, udi_upper_array2d)
from .util import occupancy_filter
from ladybug.dt import DateTime


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
        * grid_states

    """
    __slots__ = ('_folder', '_grids_info', '_sun_up_hours', '_datetimes', '_light_paths',
                 '_default_states', '_grid_states')

    def __init__(self, folder):
        """Initialize ResultsFolder."""
        self._folder = Path(folder).absolute().as_posix()
        self._grids_info, self._sun_up_hours = _process_input_folder(self.folder, '*')
        self._datetimes = [DateTime.from_hoy(hoy) for hoy in list(map(int, self.sun_up_hours))]
        self._light_paths = self._get_light_paths()
        self._default_states = self._get_default_states()
        self._grid_states = self._get_grid_states()

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
    def datetimes(self):
        """Return DateTimes for sun up hours."""
        return self._datetimes

    @property
    def light_paths(self):
        """Return the identifiers of the light paths."""
        return self._light_paths

    @property
    def default_states(self):
        """Return default states as a dictionary."""
        return self._default_states

    @property
    def grid_states(self):
        """Return grid states as a dictionary."""
        return self._grid_states

    def _get_light_paths(self) -> list:
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

    def _get_default_states(self) -> dict:
        """Set default state to 0 for all light paths."""
        default_states = {}
        for light_path in self.light_paths:
            default_states[light_path] = [0]
        return default_states

    def _get_grid_states(self) -> dict:
        """Read grid_states.json if available."""
        info = Path(self.folder, 'grid_states.json')
        if info.is_file():
            with open(info) as data_f:
                data = json.load(data_f)
            return data
        else:
            # only static results
            return None

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
        there's no possibility of being daylit or experiencing glare."""
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

    def daylight_autonomy(
        self, threshold: float = 300, states: dict = None, grids_filter: str = '*'):

        grids_info = self._filter_grids(grids_filter=grids_filter)

        da = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    occupancy_filter, 1, array, mask=self.occ_mask)
                results = da_array2d(
                    array_filter, total_occ=self.total_occ, threshold=threshold)
            else:
                results = np.zeros(grid_info['count'])
            da.append(results)

        return da

    def continuous_daylight_autonomy(
        self, threshold: float = 300, states: list = None, grids_filter: str = '*'):

        grids_info = self._filter_grids(grids_filter=grids_filter)

        cda = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    occupancy_filter, 1, array, mask=self.occ_mask)
                results = cda_array2d(
                    array_filter, total_occ=self.total_occ, threshold=threshold)
            else:
                results = np.zeros(grid_info['count'])
            cda.append(results)

        return cda

    def useful_daylight_illuminance(
        self, min_t: float = 100, max_t: float = 3000, states: list = None,
        grids_filter: str = '*'):

        grids_info = self._filter_grids(grids_filter=grids_filter)

        udi = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    occupancy_filter, 1, array, mask=self.occ_mask)
                results = udi_array2d(
                    array_filter, total_occ=self.total_occ, min_t=min_t, max_t=max_t)
            else:
                results = np.zeros(grid_info['count'])
            udi.append(results)

        return udi

    def useful_daylight_illuminance_lower(
        self, min_t: float = 100, states: list = None, grids_filter: str = '*'):

        grids_info = self._filter_grids(grids_filter=grids_filter)
        sun_down_occ_hours = self.sun_down_occ_hours

        udi_lower = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    occupancy_filter, 1, array, mask=self.occ_mask)
                results = udi_lower_array2d(array_filter, total_occ=self.total_occ,
                    min_t=min_t, sun_down_occ_hours=sun_down_occ_hours)
            else:
                results = np.zeros(grid_info['count'])
            udi_lower.append(results)

        return udi_lower

    def useful_daylight_illuminance_upper(
        self, max_t: float = 3000, states: list = None, grids_filter: str = '*'):

        grids_info = self._filter_grids(grids_filter=grids_filter)

        udi_upper = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    occupancy_filter, 1, array, mask=self.occ_mask)
                results = udi_upper_array2d(
                    array_filter, total_occ=self.total_occ, max_t=max_t)
            else:
                results = np.zeros(grid_info['count'])
            udi_upper.append(results)

        return udi_upper

    def annual_metrics(
        self, threshold: float = 300, min_t: float = 100, max_t: float = 3000,
        states: list = None, grids_filter: str = '*'):

        grids_info = self._filter_grids(grids_filter=grids_filter)
        sun_down_occ_hours = self.sun_down_occ_hours

        da = cda = udi = udi_lower = udi_upper = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    occupancy_filter, 1, array, mask=self.occ_mask)
                da_results = da_array2d(
                    array_filter, total_occ=self.total_occ, threshold=threshold)
                cda_results = cda_array2d(
                    array_filter, total_occ=self.total_occ, threshold=threshold)
                udi_results = udi_array2d(
                    array_filter, total_occ=self.total_occ, min_t=min_t, max_t=max_t)
                udi_lower_results = udi_lower_array2d(
                    array_filter, total_occ=self.total_occ, min_t=min_t,
                    sun_down_occ_hours=sun_down_occ_hours)
                udi_upper_results = udi_upper_array2d(
                    array_filter, total_occ=self.total_occ, max_t=max_t)
            else:
                da_results = cda_results = udi_results, udi_lower_results = \
                    udi_upper_results = np.zeros(grid_info['count'])
            da.append(da_results)
            cda.append(cda_results)
            udi.append(udi_results)
            udi_lower.append(udi_lower_results)
            udi_upper.append(udi_upper_results)

        return da, cda, udi, udi_lower, udi_upper

    def point_in_time(
        self, datetime: Union[int, DateTime], states: list = None, grids_filter='*',
        res_type='total'):

        grids_info = self._filter_grids(grids_filter=grids_filter)

        if isinstance(datetime, int):
            dt = DateTime.from_hoy(datetime)
        elif isinstance(datetime, DateTime):
            dt = datetime
        else:
            raise ValueError('Input datetime must be of type %s or %s. Received %s.'
                % (int, DateTime, type(datetime)))
        idx = self._index_from_datetime(dt)

        pit_values = []
        for grid_info in grids_info:
            if idx:
                array = self._array_from_states(
                    grid_info, states=states, res_type=res_type)
                pit_values.append(array[:, idx])
            else:
                # datetime not in sun up hours, add zeros
                pit_values.append(np.zeros(grid_info['count']))

        return pit_values

    def _index_from_datetime(self, datetime: DateTime):
        """Returns the index of the input datetime in the list of datetimes from the
        datetimes property.
        If the DateTime is not in the list, the function will return None.
        
        Args:
            datetime: A DateTime 
        """
        assert isinstance(datetime, DateTime), \
            'Expected Ladybug DateTime input but received %s' % type(datetime)
        try:
            index = self.datetimes.index(datetime)
        except:
            # DateTime not in sun up hours
            index = None

        return index

    def _get_array(self, grid_id: str, light_path: str, state: int = 0,
                   res_type: str = 'total', extension: str = '.npy') -> np.ndarray:
        state_identifier = self._state_identifier(grid_id, light_path, state=state)

        try:
            array = self.arrays[grid_id][light_path][state_identifier][res_type]
        except:
            array = self._load_array(
                grid_id, light_path, state=state, res_type=res_type, extension=extension)
        return array

    def _load_array(self, grid_id: str, light_path: str, state: int = 0,
                    res_type: str = 'total', extension: str = '.npy') -> np.ndarray:
        """Load a NumPy file to an array.

        This method will also update the arrays property value.

        Args:
            grid_id: Grid identifier.
            light_path: Light path identifier.
            state: Integer of the state. E.g., 0 for the default state
            res_type: Which type of result to return a file for. E.g., 'total' for total
                illuminance or 'direct' for direct illuminance.
            extension: File extension. (Default: .npy).

        Returns:
            A NumPy array from a NumPy file.
        """

        def merge_dicts(array_dict, arrays):
            for key, value in array_dict.items():
                if isinstance(value, dict):
                    node = arrays.setdefault(key, {})
                    merge_dicts(value, node)
                else:
                    arrays[key] = value
            return arrays

        state_identifier = self._state_identifier(grid_id, light_path, state=state)
        file = self._get_file(grid_id, light_path, state_identifier, res_type,
                              extension=extension)
        array = np.load(file)

        array_dict = {grid_id: {light_path: {state_identifier: {res_type: array}}}}
        arrays = merge_dicts(array_dict, self.arrays)
        self.arrays = arrays

        return array

    def _state_identifier(self, grid_id: str, light_path: str, state: int = 0) -> str:
        """Get the state identifier from a light path and state integer.

        Args:
            grid_id: Grid identifier.
            light_path: Light path identifier.
            state: Integer of the state. E.g., 0 for the default state. State integer
                must be minimum 0, or in case of -1 it will return None.

        Returns:
            State identifier. For static apertures the identifier is
            '__static_apertures__', and for other light paths it is the light path
            identifier preceded by the state integer, e.g., '0_light_path'.
        """
        # TODO: Figure out if there is a better way to handle the states.
        # I.e., state integer <--> state identifier.
        valid_states = self.valid_states[light_path]
        if state in valid_states:
            if light_path == 'static_apertures':
                state_identifier = '__static_apertures__'
            else:
                state_identifier = self.grid_states[grid_id][light_path][state]
            return state_identifier
        elif state == -1:
            return None
        else:
            raise ValueError('State of %s must be any of %s for on or -1 for off. '
                'Received state %s.' % (light_path, valid_states, state))

    def _get_file(self, grid_id: str, light_path: str, state_identifier: str,
                  res_type: str = 'total', extension: str = '.npy') -> Path:
        """Return the path of a results file.

        Args:
            grid_id: Grid identifier.
            light_path: Light path identifier.
            state_identifier: State identifier.
            res_type: Which type of result to return a file for. E.g., 'total' for total
                illuminance or 'direct' for direct illuminance.
            extension: File extension. (Default: .npy).
        
        Returns:
            Path to a NumPy file.
        """
        if light_path == 'static_apertures':
            file = Path(self.folder, light_path, '__static_apertures__', res_type,
                        grid_id + extension)
        else:
            file = Path(self.folder, light_path, state_identifier, res_type,
                        grid_id + extension)

        return file

    def _static_states(self, states: dict):
        if all(len(values) == 1 for values in states.values()):
            return True
        else:
            return False

    def _dynamic_states(self, states: dict):
        if any(len(values) != 1 for values in states.values()):
            return True
        else:
            return False

    def _validate_dynamic_states(self, states: dict):
        if all(len(values) == 8760 for values in states.values()):
            return states
        for light_path, values in states.items():
            if len(values) < 8670:
                states[light_path] = list(islice(cycle(values), 8760))
            elif len(values) > 8760:
                raise ValueError(
                    'The light path %s has %s values in its states schedule. Maximum '
                    'allowed number of values is 8760.' % (light_path, len(values))
                    )
        return states

    def _validate_states(self, states: dict):
        if all(isinstance(v, int) for values in states.values() for v in values):
            return states
        for light_path, values in states.items():
            try:
                states[light_path] = list(map(int, values))
            except ValueError as e:
                raise ValueError(
                    'Failed to convert states schedule for light path %s to '
                    'integers: %s.' % (light_path, str(e))
                    )
        return states

    def _filter_grid_states(self, grid_info, states: dict = None):
        """Filter a dictionary of states by grid. Only light paths relevant to the given
        grid will be returned.
        
        Args:
            grid_info: Grid information of the grid.
            states: A dictionary of states. Light paths as keys and lists of 8760 values
                for each key. The values should be integers matching the states or -1 for
                off.
        
        Returns:
            A filtered states dictionary.
        """
        light_paths = [lp[0] for lp in grid_info['light_path']]
        if states:
            for light_path in light_paths:
                if light_path not in states:
                    states[light_path] = self.default_states[light_path]
            states = {lp: states[lp] for lp in light_paths if lp in states}
        else:
            states = self.default_states
            states = {lp: states[lp] for lp in light_paths if lp in states}

        return states

    def _array_from_states(
        self, grid_info, states: dict = None, res_type: str = 'total') -> np.ndarray:
        """Create an array for a given grid by the states settings.

        Args:
            grid_info: Grid information of the grid.
            states: A dictionary of states. Light paths as keys and lists of 8760 values
                for each key. The values should be integers matching the states or -1 for
                off.
            res_type: Which type of result to create an array for. E.g., 'total' for total
                illuminance or 'direct' for direct illuminance.

        Returns:
            A NumPy array based on the states settings.
        """
        grid_id = grid_info['identifier']
        grid_count = grid_info['count']
        # get states that are relevant for the grid
        states = self._filter_grid_states(grid_info, states=states)
        states = self._validate_states(states)

        arrays = []
        if self._static_states(states):
            for light_path, state in states.items():
                state = state[0]
                if state == -1:
                    continue
                array = self._get_array(
                    grid_id, light_path, state=state, res_type=res_type)
                arrays.append(array)
            array = sum(arrays)
        else:
            states = self._validate_dynamic_states(states)
            for light_path, lp_states in states.items():
                # create default 0 array
                light_path_array = np.zeros((grid_count, len(self.sun_up_hours)))
                # slice states to match sun up hours
                states_array = np.array(lp_states)[list(map(int, self.sun_up_hours))]
                for state in set(states_array.tolist()):
                    if state == -1:
                        continue
                    array = self._get_array(
                        grid_id, light_path, state=state, res_type=res_type)
                    conds = [states_array == state, states_array != state]
                    light_path_array = np.select(conds, [array, light_path_array])
                arrays.append(light_path_array)
            array = sum(arrays)

        if not np.any(array):
            array = np.array([])

        return array

    def _update_occ(self):
        """Set properties related to occupancy."""
        sun_up_hours, schedule = [self.sun_up_hours, self.schedule]
        self._occ_pattern, self._total_occ, self._sun_down_occ_hours = \
            filter_schedule_by_hours(sun_up_hours=sun_up_hours, schedule=schedule)
        self._occ_mask = np.array(self.occ_pattern)

    def _filter_grids(self, grids_filter: str = '*'):
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
                    for res_type_folder in Path(state_folder).iterdir():
                        res_type = res_type_folder.name
                        file = Path(res_type_folder, grid_id + '.npy')
                        array = np.load(file)
                        arrays[grid_id][light_path][state][res_type] = array

        return arrays

    def _get_valid_states(self) -> dict:
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
        valid_states = dict()
        grid_states = self.grid_states
        if 'static_apertures' in self.light_paths:
            valid_states['static_apertures'] = [0]
        for grid_id, light_paths in grid_states.items():
            for light_path, states in light_paths.items():
                if light_path not in valid_states:
                    valid_states[light_path] = list(range(len(states)))

        return valid_states
