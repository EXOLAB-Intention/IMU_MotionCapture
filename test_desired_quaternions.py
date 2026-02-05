"""
Test that foot desired quaternions are set equal to thigh/shank
"""
import numpy as np
from core.calibration import CalibrationProcessor


def test_foot_equals_thigh():
    dq = CalibrationProcessor._get_desired_quaternions()
    assert np.allclose(dq['foot_right'], dq['thigh_right']), "foot_right != thigh_right"
    assert np.allclose(dq['foot_left'], dq['thigh_left']), "foot_left != thigh_left"


if __name__ == '__main__':
    test_foot_equals_thigh()
    print('OK')
