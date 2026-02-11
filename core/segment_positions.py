"""
Shared forward-kinematics helpers for joint positions and walking direction.
"""
from typing import Dict, Optional

import numpy as np

from config.settings import app_settings
from core.imu_data import MotionCaptureData


def _default_segment_lengths() -> Dict[str, float]:
    """Get segment lengths from settings (meters)."""
    try:
        return {
            'trunk': app_settings.get_segment_length('trunk') / 100.0,
            'thigh': app_settings.get_segment_length('thigh') / 100.0,
            'shank': app_settings.get_segment_length('shank') / 100.0,
            'foot': app_settings.get_segment_length('foot') / 100.0,
        }
    except Exception:
        return {
            'trunk': 0.50,
            'thigh': 0.40,
            'shank': 0.42,
            'foot': 0.25,
        }


def compute_joint_positions(
    data: MotionCaptureData,
    frame_index: int,
    segment_lengths: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate 3D positions of joints using forward kinematics.

    Global coordinate system: X=forward, Y=left, Z=up.
    """
    if not data or not data.imu_data:
        return {}

    lengths = segment_lengths or _default_segment_lengths()

    first_sensor = next(iter(data.imu_data.values()))
    n_frames = len(first_sensor.timestamps)
    frame_index = max(0, min(frame_index, n_frames - 1))

    positions: Dict[str, np.ndarray] = {}
    hip_pos = np.array([0.0, 0.0, 1.0])
    positions['hip'] = hip_pos

    def get_quaternion(segment_name: str) -> np.ndarray:
        if segment_name in data.imu_data:
            q = data.imu_data[segment_name].quaternions[frame_index]
            return q / np.linalg.norm(q)
        return np.array([1.0, 0.0, 0.0, 0.0])

    def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        return R @ v

    q_trunk = get_quaternion('trunk')
    q_thigh_r = get_quaternion('thigh_right')
    q_thigh_l = get_quaternion('thigh_left')
    q_shank_r = get_quaternion('shank_right')
    q_shank_l = get_quaternion('shank_left')
    q_foot_r = get_quaternion('foot_right')
    q_foot_l = get_quaternion('foot_left')

    trunk_local_dir = np.array([lengths['trunk'], 0.0, 0.0])
    thigh_local_dir = np.array([-lengths['thigh'], 0.0, 0.0])
    shank_local_dir = np.array([-lengths['shank'], 0.0, 0.0])
    foot_local_dir = np.array([0.0, 0.0, lengths['foot']])

    rhip_local_offset = np.array([0.0, 0.15, 0.0])
    lhip_local_offset = np.array([0.0, -0.15, 0.0])

    trunk_dir = rotate_vector(trunk_local_dir, q_trunk)
    trunk_top = hip_pos + trunk_dir
    positions['trunk_top'] = trunk_top

    rhip_offset = rotate_vector(rhip_local_offset, q_trunk)
    lhip_offset = rotate_vector(lhip_local_offset, q_trunk)
    rhip_pos = hip_pos + rhip_offset
    lhip_pos = hip_pos + lhip_offset
    positions['rhip'] = rhip_pos
    positions['lhip'] = lhip_pos

    thigh_r_dir = rotate_vector(thigh_local_dir, q_thigh_r)
    knee_r_pos = rhip_pos + thigh_r_dir
    positions['knee_right'] = knee_r_pos

    shank_r_dir = rotate_vector(shank_local_dir, q_shank_r)
    ankle_r_pos = knee_r_pos + shank_r_dir
    positions['ankle_right'] = ankle_r_pos

    foot_r_dir = rotate_vector(foot_local_dir, q_foot_r)
    toe_r_pos = ankle_r_pos + foot_r_dir
    positions['toe_right'] = toe_r_pos

    thigh_l_dir = rotate_vector(thigh_local_dir, q_thigh_l)
    knee_l_pos = lhip_pos + thigh_l_dir
    positions['knee_left'] = knee_l_pos

    shank_l_dir = rotate_vector(shank_local_dir, q_shank_l)
    ankle_l_pos = knee_l_pos + shank_l_dir
    positions['ankle_left'] = ankle_l_pos

    foot_l_dir = rotate_vector(foot_local_dir, q_foot_l)
    toe_l_pos = ankle_l_pos + foot_l_dir
    positions['toe_left'] = toe_l_pos

    return positions


def compute_walking_direction(
    data: MotionCaptureData,
    frame_index: int
) -> Optional[np.ndarray]:
    """Compute walking direction from trunk local +Z projected on XY."""
    if not data or 'trunk' not in data.imu_data:
        return None

    first_sensor = next(iter(data.imu_data.values()))
    n_frames = len(first_sensor.timestamps)
    frame_index = max(0, min(frame_index, n_frames - 1))

    trunk_quat = data.imu_data['trunk'].quaternions[frame_index]
    trunk_quat = trunk_quat / np.linalg.norm(trunk_quat)
    w, x, y, z = trunk_quat
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    trunk_local_z = np.array([0.0, 0.0, 1.0])
    trunk_z_global = R @ trunk_local_z
    walking_dir = np.array([trunk_z_global[0], trunk_z_global[1], 0.0])
    norm = np.linalg.norm(walking_dir[:2])
    if norm < 1e-8:
        return None
    return walking_dir / norm
