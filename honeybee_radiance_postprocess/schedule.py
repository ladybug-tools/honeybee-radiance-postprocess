"""Post-processing schedule classes."""
import json
import os


class ApertureGroupSchedule(object):
    __slots__ = ('_identifier', '_schedule')

    def __init__(self, identifier, schedule):
        self._identifier = identifier
        self.schedule = schedule

    @property
    def identifier(self):
        return self._identifier

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, values):
        self._schedule = values

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return '{}: {}'.format(self.__class__.__name__, self.identifier)


class DynamicSchedule(object):
    __slots__ = ('_values',)

    def __init__(self, values = None):
        self._values = values

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values

    def add_aperture_group_schedule(self, aperture_group_schedule):
        identifier = aperture_group_schedule.identifier
        schedule = aperture_group_schedule.schedule
        values = self.values
        values[identifier] = schedule

    def to_json(self, folder, file_name=None):
        file_name = file_name if file_name else 'dynamic_schedule'
        if not file_name.endswith('.json'):
            file_name += '.json'
        json_file = os.path.join(folder, file_name)
        with open(json_file, 'w') as fp:
            json.dump(self.values, fp)
        return json_file

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
