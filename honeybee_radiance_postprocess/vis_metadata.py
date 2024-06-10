"""Functions for visualization metadata."""
from ladybug.legend import LegendParameters
from ladybug.color import Color
from ladybug.datatype.generic import GenericType


def _abnt_nbr_15575_daylight_levels_vis_metadata():
    """Return visualization metadata for ABNT NBR 15575 Daylight levels."""
    level_value = {
        0: 'Não atende',
        1: 'Mínimo',
        2: 'Intermediário',
        3: 'Superior'
    }

    colors = [Color(255, 198, 143), Color(255, 255, 209), Color(192, 231, 189), Color(83, 169, 206)]
    illuminance_levels_lpar = \
        LegendParameters(min=0, max=3, colors=colors, segment_count=4,
                         title='Illuminance Level')
    illuminance_levels_lpar.ordinal_dictionary = level_value

    metric_info_dict = {
        '4_930AM': {
            'type': 'VisualizationMetaData',
            'data_type': GenericType('Illuminance April 23rd 9:30am', '').to_dict(),
            'unit': '',
            'legend_parameters': illuminance_levels_lpar.to_dict()
        },
        '4_330PM': {
            'type': 'VisualizationMetaData',
            'data_type': GenericType('Illuminance April 23rd 3:30pm', '').to_dict(),
            'unit': '',
            'legend_parameters': illuminance_levels_lpar.to_dict()
        },
        '10_930AM': {
            'type': 'VisualizationMetaData',
            'data_type': GenericType('Illuminance October 23rd 9:30am', '').to_dict(),
            'unit': '',
            'legend_parameters': illuminance_levels_lpar.to_dict()
        },
        '10_330PM': {
            'type': 'VisualizationMetaData',
            'data_type': GenericType('Illuminance October 23rd 3:30pm', '').to_dict(),
            'unit': '',
            'legend_parameters': illuminance_levels_lpar.to_dict()
        }
    }

    return metric_info_dict
