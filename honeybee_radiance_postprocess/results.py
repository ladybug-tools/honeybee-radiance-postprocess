"""Post-processing Results class."""
import json
from pathlib import Path
from itertools import islice, cycle
from typing import Tuple, Union, List
import numpy as np

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.illuminance import Illuminance
from ladybug.dt import DateTime
from ladybug.header import Header
from honeybee_radiance.postprocess.annual import _process_input_folder, \
    filter_schedule_by_hours, generate_default_schedule
from .metrics import (da_array2d, cda_array2d, udi_array2d, udi_lower_array2d,
    udi_upper_array2d, ase_array2d, average_values_array2d,
    cumulative_values_array2d, peak_values_array2d)
from .util import filter_array, hoys_mask, check_array_dim, \
    _filter_grids_by_pattern
from .annualdaylight import _annual_daylight_config
from .electriclight import array_to_dimming_fraction
from . import type_hints


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
        * timestep

    """
    __slots__ = ('_folder', '_grids_info', '_sun_up_hours', '_datetimes', '_light_paths',
                 '_default_states', '_grid_states', '_timestep')

    def __init__(self, folder: Union[str, Path]):
        """Initialize ResultsFolder."""
        self._folder = Path(folder).absolute().as_posix()
        self._grids_info, self._sun_up_hours = _process_input_folder(self.folder, '*')
        self._datetimes = [
            DateTime.from_hoy(hoy) for hoy in list(map(float, self.sun_up_hours))
        ]
        self._light_paths = self._get_light_paths()
        self._default_states = self._get_default_states()
        self._grid_states = self._get_grid_states()
        self._timestep = self._set_timestep()

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

    @property
    def timestep(self):
        """Return timestep as an integer."""
        return self._timestep

    def _get_light_paths(self) -> list:
        """Find all light paths in grids_info."""
        lp = []
        for grid_info in self.grids_info:
            try:
                light_paths = grid_info['light_path']
            except KeyError:
                grid_info['light_path'] = [['__static_apertures__']]
                light_paths = grid_info['light_path']
            for light_path in light_paths:
                light_path = light_path[0]
                if light_path in lp:
                    continue
                if light_path == '__static_apertures__':
                    lp.insert(0, light_path)
                else:
                    lp.append(light_path)
            if not light_paths and '__static_apertures__' not in lp:
                lp.insert(0, '__static_apertures__')

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
            return {}

    def _set_timestep(self) -> float:
        """Set timestep."""
        timestep_file = Path(self.folder, 'timestep.txt')
        if timestep_file.is_file():
            with open(timestep_file) as file:
                timestep = int(file.readline())
        else:
            timestep = 1

        return timestep

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.folder}'


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

    def __init__(self, folder, schedule: list = None, load_arrays: bool = False):
        """Initialize Results."""
        _ResultsFolder.__init__(self, folder)
        self.schedule = schedule
        self._arrays = self._load_arrays() if load_arrays else {}
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
            self, threshold: float = 300, states: dict = None,
            grids_filter: str = '*') -> type_hints.annual_metric:
        """Calculate daylight autonomy.

        Args:
            threshold: Threshold value for daylight autonomy. Defaults to 300.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the daylight autonomy and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        da = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=self.occ_mask)
                results = da_array2d(
                    array_filter, total_occ=self.total_occ, threshold=threshold)
            else:
                results = np.zeros(grid_info['count'])
            da.append(results)

        return da, grids_info

    def continuous_daylight_autonomy(
            self, threshold: float = 300, states: dict = None,
            grids_filter: str = '*') -> type_hints.annual_metric:
        """Calculate continuous daylight autonomy.

        Args:
            threshold: Threshold value for daylight autonomy. Defaults to 300.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the continuous daylight autonomy and grid
                information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        cda = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=self.occ_mask)
                results = cda_array2d(
                    array_filter, total_occ=self.total_occ, threshold=threshold)
            else:
                results = np.zeros(grid_info['count'])
            cda.append(results)

        return cda, grids_info

    def useful_daylight_illuminance(
            self, min_t: float = 100, max_t: float = 3000, states: dict = None,
            grids_filter: str = '*') -> type_hints.annual_metric:
        """Calculate useful daylight illuminance.

        Args:
            min_t: Minimum threshold for useful daylight illuminance. Defaults to 100.
            max_t: Maximum threshold for useful daylight illuminance. Defaults to 3000.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the useful daylight illuminance and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        udi = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=self.occ_mask)
                results = udi_array2d(
                    array_filter, total_occ=self.total_occ, min_t=min_t, max_t=max_t)
            else:
                results = np.zeros(grid_info['count'])
            udi.append(results)

        return udi, grids_info

    def useful_daylight_illuminance_lower(
            self, min_t: float = 100, states: dict = None,
            grids_filter: str = '*') -> type_hints.annual_metric:
        """Calculate lower than useful daylight illuminance.

        Args:
            min_t: Minimum threshold for useful daylight illuminance. Defaults to 100.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the lower than useful daylight illuminance and
                grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        sun_down_occ_hours = self.sun_down_occ_hours

        udi_lower = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=self.occ_mask)
                results = udi_lower_array2d(
                    array_filter, total_occ=self.total_occ,
                    min_t=min_t, sun_down_occ_hours=sun_down_occ_hours)
            else:
                results = np.zeros(grid_info['count'])
            udi_lower.append(results)

        return udi_lower, grids_info

    def useful_daylight_illuminance_upper(
            self, max_t: float = 3000, states: dict = None,
            grids_filter: str = '*') -> type_hints.annual_metric:
        """Calculate higher than useful daylight illuminance.

        Args:
            max_t: Maximum threshold for useful daylight illuminance. Defaults to 3000.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the higher than useful daylight illuminance and
                grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        udi_upper = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=self.occ_mask)
                results = udi_upper_array2d(
                    array_filter, total_occ=self.total_occ, max_t=max_t)
            else:
                results = np.zeros(grid_info['count'])
            udi_upper.append(results)

        return udi_upper, grids_info

    def annual_metrics(
            self, threshold: float = 300, min_t: float = 100,
            max_t: float = 3000, states: dict = None,
            grids_filter: str = '*') -> type_hints.annual_metrics:
        """Calculate multiple annual daylight metrics.

        This method will calculate the following metrics:
            * Daylight autonomy
            * Continuous daylight autonomy
            * Useful daylight illuminance
            * Lower than useful daylight illuminance
            * Higher than useful daylight illuminance

        Args:
            threshold: Threshold value for daylight autonomy. Defaults to 300.
            min_t: Minimum threshold for useful daylight illuminance. Defaults to 100.
            max_t: Maximum threshold for useful daylight illuminance. Defaults to 3000.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the five annual daylight metrics and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        sun_down_occ_hours = self.sun_down_occ_hours

        da = []
        cda = []
        udi = []
        udi_lower = []
        udi_upper = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=self.occ_mask)
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
                da_results = cda_results = udi_results = udi_lower_results = \
                    udi_upper_results = np.zeros(grid_info['count'])
            da.append(da_results)
            cda.append(cda_results)
            udi.append(udi_results)
            udi_lower.append(udi_lower_results)
            udi_upper.append(udi_upper_results)

        return da, cda, udi, udi_lower, udi_upper, grids_info

    def annual_metrics_to_folder(
            self, target_folder: str, threshold: float = 300,
            min_t: float = 100, max_t: float = 3000, states: dict = None,
            grids_filter: str = '*', config: dict = None):
        """Calculate and write multiple annual daylight metrics to a folder.

        This method will calculate the following metrics:
            * Daylight autonomy
            * Continuous daylight autonomy
            * Useful daylight illuminance
            * Lower than useful daylight illuminance
            * Higher than useful daylight illuminance

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            threshold: Threshold value for daylight autonomy. Defaults to 300.
            min_t: Minimum threshold for useful daylight illuminance. Defaults to 100.
            max_t: Maximum threshold for useful daylight illuminance. Defaults to 3000.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            config: _description_. Defaults to None.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        da, cda, udi, udi_lower, udi_upper, grids_info = self.annual_metrics(
            threshold=threshold, min_t=min_t, max_t=max_t, states=states,
            grids_filter=grids_filter)

        pattern = {
            'da': da, 'cda': cda, 'udi_lower': udi_lower, 'udi': udi,
            'udi_upper': udi_upper
        }
        for metric, data in pattern.items():
            metric_folder = folder.joinpath(metric)
            extension = metric.split('_')[0]
            for count, grid_info in enumerate(grids_info):
                d = data[count]
                full_id = grid_info['full_id']
                output_file = metric_folder.joinpath(f'{full_id}.{extension}')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(output_file, d, fmt='%.2f')

        for metric in pattern.keys():
            info_file = folder.joinpath(metric, 'grids_info.json')
            info_file.write_text(json.dumps(grids_info))

        config = config or _annual_daylight_config()
        config_file = folder.joinpath('config.json')
        config_file.write_text(json.dumps(config))

    def spatial_daylight_autonomy(
            self, threshold: float = 300, target_time: float = 50,
            states: dict = None, grids_filter: str = '*'
            ) -> type_hints.spatial_daylight_autonomy:
        """Calculate spatial daylight autonomy.

        Note: This component will only output a LEED compliant sDA if you've
        run the simulation with blinds and blinds schedules as per the
        IES-LM-83-12. Use the states option to calculate a LEED compliant sDA.

        Args:
            threshold: Threshold value for daylight autonomy. Defaults to 300.
            target_time: A minimum threshold of occupied time (eg. 50% of the
                time), above which a given sensor passes and contributes to the
                spatial daylight autonomy. Defaults to 50.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the spatial daylight autonomy and grid
                information.
        """
        da, grids_info = self.daylight_autonomy(
           threshold=threshold, states=states, grids_filter=grids_filter)

        sda = []
        for array in da:
            sda.append((array >= target_time).mean())

        return sda, grids_info

    def annual_sunlight_exposure(
            self, direct_threshold: float = 1000, occ_hours: int = 250,
            states: dict = None, grids_filter: str = '*'
            ) -> type_hints.annual_sunlight_exposure:
        """Calculate annual sunlight exposure.

        Args:
            direct_threshold: The threshold that determines if a sensor is
                overlit. Defaults to 1000.
            occ_hours: The number of occupied hours that cannot receive more
                than the direct_threshold. Defaults to 250.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the annual sunlight exposure, the number of
                hours that exceeds the direct threshold for each sensor, and
                grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        ase = []
        hours_above = []
        for grid_info in grids_info:
            array = self._array_from_states(
                grid_info, states=states, res_type='direct')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=self.occ_mask)
                results, h_above = ase_array2d(
                    array_filter, occ_hours=occ_hours,
                    direct_threshold=direct_threshold)
            else:
                results = np.float64(0)
                h_above = np.zeros(grid_info['count'])
            ase.append(results)
            hours_above.append(h_above)

        return ase, hours_above, grids_info

    def annual_sunlight_exposure_to_folder(
            self, target_folder: str, direct_threshold: float = 1000,
            occ_hours: int = 250, states: dict = None,
            grids_filter: str = '*'):
        """Calculate and write annual sunlight exposure to a folder.

        Args:
            direct_threshold: The threshold that determines if a sensor is
                overlit. Defaults to 1000.
            occ_hours: The number of occupied hours that cannot receive more
                than the direct_threshold. Defaults to 250.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        ase, hours_above, grids_info = self.annual_sunlight_exposure(
            direct_threshold=direct_threshold, occ_hours=occ_hours,
            states=states, grids_filter=grids_filter
            )

        pattern = {'ase': ase, 'hours_above': hours_above}
        for metric, data in pattern.items():
            metric_folder = folder.joinpath(metric)
            extension = metric.split('_')[0]
            #if metric == 'hours_above':
            for count, grid_info in enumerate(grids_info):
                d = data[count]
                full_id = grid_info['full_id']
                output_file = metric_folder.joinpath(f'{full_id}.{extension}')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                if metric == 'hours_above':
                    np.savetxt(output_file, d, fmt='%i')
                elif metric == 'ase':
                    output_file.write_text('%.2f' % d)

        for metric in pattern.keys():
            info_file = folder.joinpath(metric, 'grids_info.json')
            info_file.write_text(json.dumps(grids_info))

    def point_in_time(
            self, datetime: Union[int, DateTime], states: dict = None,
            grids_filter: str = '*', res_type: str = 'total'
            ) -> type_hints.point_in_time:
        """Get point in time values.

        Args:
            datetime: Hour of the as an integer or DateTime object.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with point in time values and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        if isinstance(datetime, int):
            dt = DateTime.from_hoy(datetime)
        elif isinstance(datetime, DateTime):
            dt = datetime
        else:
            error_message = (
                f'Input datetime must be of type {int} or {DateTime}. '
                f'Received {type(DateTime)}.'
            )
            raise ValueError(error_message)

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

        return pit_values, grids_info

    def average_values(
            self, hoys: list = [], states: dict = None, grids_filter: str = '*',
            res_type: str = 'total') -> type_hints.average_values:
        """Get average values for each sensor over a given period.

        The hoys input can be used to filter the data for a particular time
        period.

        Args:
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with the average value for each sensor and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        full_length = 8760 * self.timestep if len(hoys) == 0 else len(hoys)
        mask = hoys_mask(self.sun_up_hours, hoys, self.timestep)

        average_values = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type=res_type)
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=mask)
                results = average_values_array2d(
                    array_filter, full_length)
            else:
                results = np.zeros(grid_info['count'])
            average_values.append(results)

        return average_values, grids_info

    def average_values_to_folder(
            self, target_folder: str, hoys: list = [], states: dict = None,
            grids_filter: str = '*', res_type: str = 'total'):
        """Get average values for each sensor over a given period and write the
        values to a folder.

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        average_values, grids_info = self.average_values(
            hoys=hoys, states=states,
            grids_filter=grids_filter, res_type=res_type)

        metric_folder = folder.joinpath('average_values')

        for count, grid_info in enumerate(grids_info):
            d = average_values[count]
            full_id = grid_info['full_id']
            output_file = metric_folder.joinpath(f'{full_id}.average')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, d, fmt='%.2f')

        info_file = metric_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

    def cumulative_values(
            self, hoys: list = [], states: dict = None, grids_filter: str = '*',
            res_type: str = 'total') -> type_hints.cumulative_values:
        """Get cumulative values for each sensor over a given period.

        The hoys input can be used to filter the data for a particular time
        period.

        Args:
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with the cumulative value for each sensor and grid
                information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        mask = hoys_mask(self.sun_up_hours, hoys, self.timestep)

        cumulative_values = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type=res_type)
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=mask)
                results = cumulative_values_array2d(
                    array_filter, self.timestep)
            else:
                results = np.zeros(grid_info['count'])
            cumulative_values.append(results)

        return cumulative_values, grids_info

    def cumulative_values_to_folder(
            self, target_folder: str, hoys: list = [], states: dict = None,
            grids_filter: str = '*', res_type: str = 'total'):
        """Get cumulative values for each sensor over a given period and write
        the values to a folder.

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        cumulative_values, grids_info = self.cumulative_values(
            hoys=hoys, states=states, grids_filter=grids_filter, res_type=res_type)

        metric_folder = folder.joinpath('cumulative_values')

        for count, grid_info in enumerate(grids_info):
            d = cumulative_values[count]
            full_id = grid_info['full_id']
            output_file = metric_folder.joinpath(f'{full_id}.cumulative')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, d, fmt='%.2f')

        info_file = metric_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

    def peak_values(
            self, hoys: list = [], states: dict = None, grids_filter: str = '*',
            coincident: bool = False, res_type: str = 'total'
            ) -> type_hints.peak_values:
        """Get peak values for each sensor over a given period.

        The hoys input can be used to filter the data for a particular time
        period.

        Args:
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            coincident: Boolean to indicate whether output values represent the peak
                value for each sensor throughout the entire analysis (False) or they
                represent the highest overall value across each sensor grid at a
                particular timestep (True). Defaults to False.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with the peak value for each sensor and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        mask = hoys_mask(self.sun_up_hours, hoys, self.timestep)

        cumulative_values = []
        max_hoys = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type=res_type)
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=mask)
                results, max_i = peak_values_array2d(
                    array_filter, coincident=coincident)
            else:
                results = np.zeros(grid_info['count'])
            cumulative_values.append(results)
            if max_i:
                max_hoys.append(int(self.sun_up_hours[max_i]))
            else:
                max_hoys.append(max_i)

        return cumulative_values, max_hoys, grids_info

    def peak_values_to_folder(
            self, target_folder: str, hoys: list = [], states: dict = None,
            grids_filter: str = '*', coincident: bool = False, res_type='total'):
        """Get peak values for each sensor over a given period and write the
        values to a folder.

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
            coincident: Boolean to indicate whether output values represent the peak
                value for each sensor throughout the entire analysis (False) or they
                represent the highest overall value across each sensor grid at a
                particular timestep (True). Defaults to False.
            res_type: Type of results to load. Defaults to 'total'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        peak_values, max_hoys, grids_info = self.peak_values(
            hoys=hoys, states=states, grids_filter=grids_filter,
            coincident=coincident, res_type=res_type)

        metric_folder = folder.joinpath('peak_values')

        for count, grid_info in enumerate(grids_info):
            d = peak_values[count]
            full_id = grid_info['full_id']
            output_file = metric_folder.joinpath(f'{full_id}.peak')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, d, fmt='%.2f')

        info_file = metric_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

    def annual_data(
            self, states: dict = None, grids_filter: str = '*',
            sensor_index: dict = None, res_type: str = 'total'
            ) -> type_hints.annual_data:
        """Get annual data for one or multiple sensors.

        Args:
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            sensor_index: A dictionary with grids as keys and a list of sensor
                indices as values. Defaults to None.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with Data Collections for each sensor, grid information,
                and a dictionary of the sensors.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        analysis_period = AnalysisPeriod(timestep=self.timestep)

        # if no sensor_index, create dict with all sensors
        if not sensor_index:
            sensor_index = {}
            for grid_info in grids_info:
                sensor_index[grid_info['full_id']] = \
                    [i for i in range(grid_info['count'])]

        data_collections = []
        for grid_info in grids_info:
            data_collections_grid = []
            grid_id = grid_info['full_id']
            array = self._array_from_states(grid_info, states=states, res_type=res_type)
            indices = sensor_index[grid_id]
            for idx in indices:
                values = array[idx, :]
                annual_array = Results.values_to_annual(
                    self.sun_up_hours, values, self.timestep)
                header = Header(Illuminance(), 'lux', analysis_period)
                header.metadata['sensor grid'] = grid_id
                header.metadata['sensor index'] = idx
                data_collections_grid.append(
                    HourlyContinuousCollection(header, annual_array.tolist()))
            data_collections.append(data_collections_grid)

        return data_collections, grids_info, sensor_index

    def annual_data_to_folder(
            self, target_folder: str, states: dict = None, grids_filter: str = '*',
            sensor_index: dict = None, res_type: str = 'total'):
        """Get annual data for one or multiple sensors and write the data to a
        folder as Data Collections.

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            sensor_index: A dictionary with grids as keys and a list of sensor
                indices as values. Defaults to None.
            res_type: Type of results to load. Defaults to 'total'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        data_collections, grids_info, sensor_index = self.annual_data(
            states=states, grids_filter=grids_filter, sensor_index=sensor_index,
            res_type=res_type)

        metric_folder = folder.joinpath('datacollections')

        for count, grid_info in enumerate(grids_info):
            grid_collections = data_collections[count]
            for data_collection in grid_collections:
                grid_id = grid_info['full_id']
                sensor_id = data_collection.header.metadata['sensor index']
                data_dict = data_collection.to_dict()
                data_file = metric_folder.joinpath(f'{grid_id}_{sensor_id}.json')
                data_file.parent.mkdir(parents=True, exist_ok=True)
                data_file.write_text(json.dumps(data_dict))

    def daylight_control_schedules(
            self, states: dict = None, grids_filter: str = '*',
            base_schedule: list = None, ill_setpoint: float = 300,
            min_power_in: float = 0.3, min_light_out: float = 0.2,
            off_at_min: bool = False) -> Tuple[List[np.ndarray], List[str]]:
        """Generate electric lighting schedules from annual daylight results.

        Such controls will dim the lights according to whether the illuminance values
        at the sensor locations are at a target illuminance setpoint. The results can be
        used to account for daylight controls in energy simulations.

        This function will generate one schedule per sensor grid in the simulation. Each
        grid should have sensors at the locations in space where daylight dimming sensors
        are located. Grids with one, two, or more sensors can be used to model setups
        where fractions of each room are controlled by different sensors. If the sensor
        grids are distributed over the entire floor of the rooms, the resulting schedules
        will be idealized, where light dimming has been optimized to supply the minimum
        illuminance setpoint everywhere in the room.

        Args:
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            base_schedule: A list of 8760 fractional values for the lighting schedule
                representing the usage of lights without any daylight controls. The
                values of this schedule will be multiplied by the hourly dimming
                fraction to yield the output lighting schedules. If None, a schedule
                from 9AM to 5PM on weekdays will be used. (Default: None).
            ill_setpoint: A number for the illuminance setpoint in lux beyond which
                electric lights are dimmed if there is sufficient daylight.
                Some common setpoints are listed below. (Default: 300 lux).

                * 50 lux - Corridors and hallways.
                * 150 lux - Computer work spaces (screens provide illumination).
                * 300 lux - Paper work spaces (reading from surfaces that need illumination).
                * 500 lux - Retail spaces or museums illuminating merchandise/artifacts.
                * 1000 lux - Operating rooms and workshops where light is needed for safety.

            min_power_in: A number between 0 and 1 for the the lowest power the lighting
                system can dim down to, expressed as a fraction of maximum
                input power. (Default: 0.3).
            min_light_out: A number between 0 and 1 the lowest lighting output the lighting
                system can dim down to, expressed as a fraction of maximum light
                output. Note that setting this to 1 means lights aren't dimmed at
                all until the illuminance setpoint is reached. This can be used to
                approximate manual light-switching behavior when used in conjunction
                with the off_at_min input below. (Default: 0.2).
            off_at_min: Boolean to note whether lights should switch off completely when
                they get to the minimum power input. (Default: False).

        Returns:
            A tuple with two values.

            -   schedules: A list of lists where each sub-list represents an electric
                lighting dimming schedule for a sensor grid.

            -   schedule_ids: A list of text strings for the recommended names of the
                electric lighting schedules.
        """
        # process the base schedule input into a list of values
        if base_schedule is None:
            base_schedule = generate_default_schedule()
        base_schedule = np.array(base_schedule)

        grids_info = self._filter_grids(grids_filter=grids_filter)
        sun_up_hours = [int(h) for h in self.sun_up_hours]

        dim_fracts = []
        for grid_info in grids_info:
            array = self._array_from_states(
                grid_info, states=states, res_type='total')
            if np.any(array):
                fract_list = array_to_dimming_fraction(
                    array, sun_up_hours, ill_setpoint, min_power_in,
                    min_light_out, off_at_min
                )
            else:
                fract_list = np.ones(8760)
            dim_fracts.append(fract_list)

        schedules, schedule_ids = [], []
        for grid_info, dim_fract in zip(grids_info, dim_fracts):
            grid_id = grid_info['full_id']
            sch_vals = base_schedule * dim_fract
            sch_id = f'{grid_id} Daylight Control'
            schedules.append(sch_vals)
            schedule_ids.append(sch_id)

        return schedules, schedule_ids, grids_info

    def daylight_control_schedules_to_folder(
            self, target_folder: str, states: dict = None,
            grids_filter: str = '*', base_schedule: list = None,
            ill_setpoint: float = 300, min_power_in: float = 0.3,
            min_light_out: float = 0.2, off_at_min: bool = False):
        """Generate electric lighting schedules from annual daylight results and
        write the schedules to a folder.

        Such controls will dim the lights according to whether the illuminance values
        at the sensor locations are at a target illuminance setpoint. The results can be
        used to account for daylight controls in energy simulations.

        This function will generate one schedule per sensor grid in the simulation. Each
        grid should have sensors at the locations in space where daylight dimming sensors
        are located. Grids with one, two, or more sensors can be used to model setups
        where fractions of each room are controlled by different sensors. If the sensor
        grids are distributed over the entire floor of the rooms, the resulting schedules
        will be idealized, where light dimming has been optimized to supply the minimum
        illuminance setpoint everywhere in the room.

        Args:
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            base_schedule: A list of 8760 fractional values for the lighting schedule
                representing the usage of lights without any daylight controls. The
                values of this schedule will be multiplied by the hourly dimming
                fraction to yield the output lighting schedules. If None, a schedule
                from 9AM to 5PM on weekdays will be used. (Default: None).
            ill_setpoint: A number for the illuminance setpoint in lux beyond which
                electric lights are dimmed if there is sufficient daylight.
                Some common setpoints are listed below. (Default: 300 lux).

                * 50 lux - Corridors and hallways.
                * 150 lux - Computer work spaces (screens provide illumination).
                * 300 lux - Paper work spaces (reading from surfaces that need illumination).
                * 500 lux - Retail spaces or museums illuminating merchandise/artifacts.
                * 1000 lux - Operating rooms and workshops where light is needed for safety.

            min_power_in: A number between 0 and 1 for the the lowest power the lighting
                system can dim down to, expressed as a fraction of maximum
                input power. (Default: 0.3).
            min_light_out: A number between 0 and 1 the lowest lighting output the lighting
                system can dim down to, expressed as a fraction of maximum light
                output. Note that setting this to 1 means lights aren't dimmed at
                all until the illuminance setpoint is reached. This can be used to
                approximate manual light-switching behavior when used in conjunction
                with the off_at_min input below. (Default: 0.2).
            off_at_min: Boolean to note whether lights should switch off completely when
                they get to the minimum power input. (Default: False).
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        schedules, schedule_ids, grids_info = self.daylight_control_schedules(
            states=states, grids_filter=grids_filter,
            base_schedule=base_schedule, ill_setpoint=ill_setpoint,
            min_power_in=min_power_in, min_light_out=min_light_out,
            off_at_min=off_at_min)

        schedule_folder = folder.joinpath('control_schedules')

        for count, grid_info in enumerate(grids_info):
            d = schedules[count]
            full_id = grid_info['full_id']
            output_file = schedule_folder.joinpath(f'{full_id}.txt')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, d, fmt='%.2f')

            id_file = schedule_folder.joinpath(f'{full_id}.id')
            id_file.write_text(schedule_ids[count])

        info_file = schedule_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

    @staticmethod
    def values_to_annual(
            hours: Union[List[float], np.ndarray],
            values: Union[List[float], np.ndarray],
            timestep: float, base_value: int = 0) -> np.ndarray:
        """Map a 1D NumPy array based on a set of hours to an annual array.

        This method creates an array with a base value of length 8760 and
        replaces the base value with the input 'values' at the indices of the
        input 'hours'.

        Args:
            hours: A list of hours. This can be a regular list or a 1D NumPy
                array.
            values: A list of values to map to an annual array. This can be a
                regular list or a 1D NumPy array.
            timestep: Timestep of the simulation.
            base_value: A value that will be applied for all the base array.

        Returns:
            np.ndarray: A 1D NumPy array.
        """
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        check_array_dim(values, 1)
        hours = np.array(hours).astype(int)
        assert hours.shape == values.shape
        annual_array = np.repeat(base_value, 8760 * timestep).astype(np.float32)
        annual_array[hours] = values

        return annual_array

    def _index_from_datetime(self, datetime: DateTime) -> Union[int, None]:
        """Returns the index of the input datetime in the list of datetimes
        from the datetimes property.

        If the DateTime is not in the list, the function will return None.

        Args:
            datetime: A DateTime object.

        Returns:
            Index as an integer or None.
        """
        assert isinstance(datetime, DateTime), \
            f'Expected DateTime object but received {type(datetime)}'
        try:
            index = self.datetimes.index(datetime)
        except Exception:
            # DateTime not in sun up hours
            index = None

        return index

    def _get_array(
            self, grid_info: dict, light_path: str, state: int = 0,
            res_type: str = 'total', extension: str = '.npy') -> np.ndarray:
        """Get an array for a given grid, light path, and state.

        The array will be fetched from the 'arrays' property if it has been
        loaded already.

        Args:
            grid_info: Grid information.
            light_path: Identifier of the light path.
            state: State as an integer. E.g., 0 for the default state.
                Defaults to 0.
            res_type: Type of results to load. Defaults to 'total'.
            extension: File extension of the array to load. Defaults to '.npy'.

        Returns:
            np.ndarray: A NumPy array of a given grid, light path, and state.
        """
        grid_id = grid_info['full_id']

        state_identifier = self._state_identifier(grid_id, light_path, state=state)

        try:
            array = self.arrays[grid_id][light_path][state_identifier][res_type]
        except Exception:
            array = self._load_array(
                grid_info, light_path, state=state, res_type=res_type,
                extension=extension
            )

        return array

    def _load_array(
            self, grid_info: dict, light_path: str, state: int = 0,
            res_type: str = 'total', extension: str = '.npy') -> np.ndarray:
        """Load a NumPy file to an array.

        This method will also update the arrays property value.

        Args:
            grid_info: Grid information.
            light_path: Identifier of the light path.
            state: State as an integer. E.g., 0 for the default state.
                Defaults to 0.
            res_type: Which type of result to return a file for. E.g., 'total' for total
                illuminance or 'direct' for direct illuminance.
            extension: File extension of the array to load. Defaults to '.npy'.

        Returns:
            np.ndarray: A NumPy array of a given grid, light path, and state
                from a NumPy file.
        """
        grid_id = grid_info['identifier']
        full_id = grid_info['full_id']

        def merge_dicts(array_dict, arrays):
            for key, value in array_dict.items():
                if isinstance(value, dict):
                    node = arrays.setdefault(key, {})
                    merge_dicts(value, node)
                else:
                    arrays[key] = value
            return arrays

        state_identifier = self._state_identifier(grid_id, light_path, state=state)
        file = self._get_file(full_id, light_path, state_identifier, res_type,
                              extension=extension)
        array = np.load(file)

        array_dict = {grid_id: {light_path: {state_identifier: {res_type: array}}}}
        arrays = merge_dicts(array_dict, self.arrays)
        self.arrays = arrays

        return array

    def _state_identifier(
            self, grid_id: str, light_path: str, state: int = 0) -> Union[str, None]:
        """Get the state identifier from a light path and state integer.

        Args:
            grid_id: Grid identifier.
            light_path: Identifier of the light path.
            state: State as an integer. E.g., 0 for the default state.
                Defaults to 0.

        Returns:
            State identifier. For static apertures the identifier is 'default',
            and for other light paths it is the light path identifier preceded
            by the state integer, e.g., '0_light_path'. If the state is -1 the
            state identifier will be None.
        """
        # TODO: Figure out if there is a better way to handle the states.
        # I.e., state integer <--> state identifier.

        # This is to get around a bug in honeybee-radiance library that uses the
        # identifier and not the full_id to create the grid_states.json file.
        # we should fix this in the source. cc: Mikkel
        grid_id = grid_id.split('/')[-1]
        valid_states = self.valid_states[light_path]
        if state in valid_states:
            if light_path == '__static_apertures__':
                state_identifier = 'default'
            else:
                state_identifier = self.grid_states[grid_id][light_path][state]
            return state_identifier
        elif state == -1:
            return None
        else:
            error_message = (
                f'State of {light_path} must be any of {valid_states} for on '
                f'or -1 for off. Received state {state}.'
            )
            raise ValueError(error_message)

    def _get_file(
            self, grid_id: str, light_path: str, state_identifier: str,
            res_type: str = 'total', extension: str = '.npy') -> Path:
        """Return the path of a results file.

        Args:
            grid_id: Grid identifier.
            light_path: Identifier of the light path.
            state_identifier: State identifier.
            res_type: Which type of result to return a file for. E.g., 'total' for total
                illuminance or 'direct' for direct illuminance.
            extension: File extension of the array to load. Defaults to '.npy'.

        Returns:
            Path to a NumPy file.
        """
        file = Path(self.folder, light_path, state_identifier,
                    res_type, grid_id + extension)

        return file

    def _static_states(self, states: dict) -> bool:
        """Return True if states dictionary is static only."""
        if all(len(values) == 1 for values in states.values()):
            return True
        else:
            return False

    def _dynamic_states(self, states: dict) -> bool:
        """Return True if states dictionary is dynamic."""
        if any(len(values) != 1 for values in states.values()):
            return True
        else:
            return False

    def _validate_dynamic_states(self, states: dict) -> dict:
        """Validate dynamic states and return states dictionary.

        If all light paths in the dictionary have 8760 values, the states
        dictionary is returned as is. If some light paths have less than 8760
        values, pattern of values will be repeated until it reaches a length of
        8760.

        Args:
            states: A dictionary of states.

        Returns:
            dict: A dictionary of states.
        """
        if all(len(values) == 8760 for values in states.values()):
            return states
        for light_path, values in states.items():
            if len(values) < 8760:
                states[light_path] = list(islice(cycle(values), 8760))
            elif len(values) > 8760:
                error_message = (
                    f'The light path {light_path} has {len(values)} values in '
                    f'its states schedule. Maximum allowed number of values '
                    f'is 8760.'
                )
                raise ValueError(error_message)

        return states

    def _validate_states(self, states: dict) -> dict:
        """Validate states and return states dictionary.

        If all light paths in the dictionary have integers only as values, the
        states dictionary is returned as is. If some light paths have values
        that are not integers, these values will be mapped as integers if
        possible, e.g., if the values are strings ('0', '1', etc.) instead of
        integers.

        Args:
            states: A dictionary of states.

        Returns:
            dict: A dictionary of states.
        """
        if all(isinstance(v, int) for values in states.values() for v in values):
            return states
        for light_path, values in states.items():
            try:
                states[light_path] = list(map(int, values))
            except ValueError as err:
                error_message = (
                    f'Failed to convert states schedule for light path '
                    f'{light_path} to integers.'
                )
                raise ValueError(error_message) from err

        return states

    def _filter_grid_states(self, grid_info, states: dict = None) -> dict:
        """Filter a dictionary of states by grid. Only light paths relevant to
        the given grid will be returned.

        Args:
            grid_info: Grid information.
            states: A dictionary of states. Light paths as keys and lists of
                8760 values for each key. The values should be integers
                matching the states or -1 for off. Default to None.

        Returns:
            dict: A filtered states dictionary.
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
            self, grid_info, states: dict = None, res_type: str = 'total'
            ) -> np.ndarray:
        """Create an array for a given grid by the states settings.

        Args:
            grid_info: Grid information of the grid.
            states: A dictionary of states. Light paths as keys and lists of 8760 values
                for each key. The values should be integers matching the states or -1 for
                off.
            res_type: Which type of result to create an array for. E.g., 'total'
                for total illuminance or 'direct' for direct illuminance.

        Returns:
            A NumPy array based on the states settings.
        """
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
                    grid_info, light_path, state=state, res_type=res_type)
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
                        grid_info, light_path, state=state, res_type=res_type)
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

    def _filter_grids(self, grids_filter: str = '*') -> list:
        """Return grids information.

        Args:
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            list: List of grid information for filtered grids.
        """
        if grids_filter != '*':
            grids_info = \
                _filter_grids_by_pattern(self.grids_info, grids_filter)
        else:
            grids_info = self.grids_info

        return grids_info

    def _load_arrays(self) -> dict:
        """Load all the NumPy arrays in the results folder."""
        arrays = {}
        grids_info = self.grids_info

        for grid_info in grids_info:
            grid_id = grid_info['identifier']
            light_paths = grid_info['light_path']
            arrays[grid_id] = {}
            for light_path in light_paths:
                light_path = light_path[0]
                arrays[grid_id][light_path] = {}
                light_path_folder = Path(self.folder, light_path)
                for state_folder in Path(light_path_folder).iterdir():
                    state = state_folder.name
                    arrays[grid_id][light_path][state] = {}
                    for res_type_folder in Path(state_folder).iterdir():
                        res_type = res_type_folder.name
                        file = Path(res_type_folder, grid_id + '.npy')
                        array = np.load(file)
                        arrays[grid_id][light_path][state][res_type] = array

        return arrays

    def _get_valid_states(self) -> dict:
        """Returns a dictionary with valid states for each light path.

        For each light path there will be a key (identifier of the light path)
        and its value will be a list of valid states as integers.

        Example of output format:
        {
            '__static_apertures__': [0],
            'Room1_North': [0, 1],
            'Room1_South': [0, 1],
            'Room2_North1': [0, 1],
            'Room2_North2': [0, 1]
        }

        Returns:
            dict: Valid states integers for each light path.
        """
        valid_states = {}
        grid_states = self.grid_states
        if '__static_apertures__' in self.light_paths:
            valid_states['__static_apertures__'] = [0]
        for light_paths in grid_states.values():
            for light_path, states in light_paths.items():
                if light_path not in valid_states:
                    valid_states[light_path] = list(range(len(states)))

        return valid_states
