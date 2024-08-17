"""Test reader module."""
import numpy as np

from honeybee_radiance_postprocess.reader import binary_to_array


def test_binary_to_array_allclose():
    sky_f = './tests/assets/binary_to_array/sky_f.ill'
    sky_d = './tests/assets/binary_to_array/sky_d.ill'
    sky_a = './tests/assets/binary_to_array/sky_a.ill'

    sky_f = binary_to_array(sky_f)
    sky_d = binary_to_array(sky_d)
    sky_a = binary_to_array(sky_a)

    assert np.allclose(sky_f, sky_d)
    assert np.allclose(sky_d, sky_a)
    assert np.allclose(sky_f, sky_a)
