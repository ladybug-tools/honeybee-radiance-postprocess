"""Post-processing classes for dynamic schedules."""
import json
import os
import sys

from honeybee.config import folders


class ApertureGroupSchedule(object):
    """ApertureGroupSchedule.

    This class contains the dynamic states schedule of a single ApertureGroup.

    Args:
        identifier: Identifier of the ApertureGroup.
        schedule: A list or tuple of integer values.

    Properties:
        * identifier
        * schedule
    """
    __slots__ = ('_identifier', '_schedule')

    def __init__(self, identifier, schedule):
        """Initialize ApertureGroupSchedule."""
        self._identifier = identifier
        self.schedule = schedule

    @property
    def identifier(self):
        """Return identifier."""
        return self._identifier

    @property
    def schedule(self):
        """Return ApertureGroup schedule."""
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        assert isinstance(schedule, (list, tuple)), \
            ('Failed to add schedule. Received input of type: '
            '%s. Expected input of type: list or tuple.' % type(schedule))
        if isinstance(schedule, tuple):
            schedule = list(schedule)
        self._schedule = schedule

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return '{}: {}'.format(self.__class__.__name__, self.identifier)


class DynamicSchedule(object):
    """DynamicSchedule.

    This class contains a single property (dynamic_schedule). This property is
    a dictionary where keys are ApertureGroup identifiers and the value for
    each key is the dynamic schedule for the ApertureGroup.

    Args:
        dynamic_schedule: A dictionary of ApertureGroup identifier and
            schedules.

    Properties:
        * dynamic_schedule
    """
    __slots__ = ('_dynamic_schedule',)

    def __init__(self, dynamic_schedule = None):
        self.dynamic_schedule = dynamic_schedule

    @classmethod
    def from_dict(cls, data):
        """Initialize a DynamicSchedule from a dictionary.

        Args:
            data: A dictionary representation of a DynamicSchedule objects.
        """
        new_obj = cls(data)
        return new_obj

    @classmethod
    def from_json(cls, json_file):
        """Initialize a DynamicSchedule from a JSON file.

        Args:
            json_file: Path to JSON file.
        """
        assert os.path.isfile(json_file), 'Failed to find %s' % json_file
        if sys.version_info < (3, 0):
            with open(json_file) as inf:
                data = json.load(inf)
        else:
            with open(json_file, encoding='utf-8') as inf:
                data = json.load(inf)
        return cls.from_dict(data)

    @property
    def dynamic_schedule(self):
        """Return dynamic schedule as a dictionary."""
        return self._dynamic_schedule

    @dynamic_schedule.setter
    def dynamic_schedule(self, dynamic_schedule):
        if not dynamic_schedule:
            dynamic_schedule = {}
        assert isinstance(dynamic_schedule, dict), 'values is wrong type!'
        self._dynamic_schedule = dynamic_schedule

    def add_aperture_group_schedule(self, aperture_group_schedule):
        """Add an ApertureGroupSchedule to the DynamicSchedule instance.

        Args:
            aperture_group_schedule: An ApertureGroupSchedule object.
        """
        assert isinstance(aperture_group_schedule, ApertureGroupSchedule), \
            ('Failed to add ApertureGroupSchedule. Received input of type: '
            '%s. Expected input of type: ApertureGroupSchedule.' \
            % type(aperture_group_schedule))
        identifier = aperture_group_schedule.identifier
        schedule = aperture_group_schedule.schedule
        self.dynamic_schedule[identifier] = schedule

    def to_json(self, folder=None, file_name=None):
        """Write a DynamicSchedule to a JSON file.

        Args:
            folder: A text string for the directory where the JSON file will be
                written. If unspecified, the default simulation folder will be
                used. This is usually at "C:\\Users\\USERNAME\\simulation."
            file_name (_type_, optional): _description_. Defaults to None.

        Returns:
            json_file: Path to JSON file.
        """
        file_name = file_name if file_name else 'dynamic_schedule'
        if not file_name.endswith('.json'):
            file_name += '.json'
        folder = folder if folder is not None else folders.default_simulation_folder
        json_file = os.path.join(folder, file_name)
        with open(json_file, 'w') as fp:
            json.dump(self.dynamic_schedule, fp, indent=2)
        return json_file

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
