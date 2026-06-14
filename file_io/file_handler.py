"""
File I/O handler for IMU motion capture data
Supports import, save, and load operations
"""
import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from core.imu_data import (
    MotionCaptureData, IMUSensorData, JointAngles, KinematicsData, MarkerData
)
from config.settings import app_settings


class FileHandler:
    """Handles file operations for motion capture data"""
    
    # Supported file extensions
    RAW_EXTENSIONS = ['.csv', '.txt', '.dat', '.h5']  # Raw IMU data formats
    PROCESSED_EXTENSION = '.mcp'  # Motion Capture Processed data

    MARKER_VISUALIZATION_NAMES = [
        'rthi', 'rathi', 'rpthi',
        'lthi', 'lathi', 'lpthi',
        'rtib', 'ratib', 'rptib',
        'ltib', 'latib', 'lptib',
        'rmank', 'rank', 'rhee', 'rtoe',
        'lmank', 'lank', 'lhee', 'ltoe',
    ]

    H5_TRIAL_SENSOR_MAP = [
        ('back',        0, ['robot/imu/back_imu', 'robot/imu/xsens/back_imu', 'robot/back_imu']),
        ('thigh_left',  1, ['robot/imu/thigh_imu/left', 'robot/imu/xsens/thigh_imu/left', 'robot/thigh_imu/left']),
        ('thigh_right', 2, ['robot/imu/thigh_imu/right', 'robot/imu/xsens/thigh_imu/right', 'robot/thigh_imu/right']),
        ('shank_left',  3, ['robot/imu/shank_imu/left', 'robot/imu/xsens/shank_imu/left', 'robot/shank_imu/left']),
        ('shank_right', 4, ['robot/imu/shank_imu/right', 'robot/imu/xsens/shank_imu/right', 'robot/shank_imu/right']),
        ('foot_left',   5, ['robot/imu/foot_imu/left', 'robot/imu/xsens/foot_imu/left', 'robot/foot_imu/left']),
        ('foot_right',  6, ['robot/imu/foot_imu/right', 'robot/imu/xsens/foot_imu/right', 'robot/foot_imu/right']),
    ]

    H5_NEUTRAL_POSE_SENSOR_MAP = [
        ('back',        0, ['back_imu']),
        ('thigh_left',  1, ['thigh_imu/left']),
        ('thigh_right', 2, ['thigh_imu/right']),
        ('shank_left',  3, ['shank_imu/left']),
        ('shank_right', 4, ['shank_imu/right']),
        ('foot_left',   5, ['foot_imu/left']),
        ('foot_right',  6, ['foot_imu/right']),
    ]
    
    @staticmethod
    def import_raw_data(filepath: str, h5_path: str = None) -> MotionCaptureData:
        """
        Import raw IMU data from file

        Args:
            filepath: Path to raw data file
            h5_path: Internal HDF5 path (required for .h5 files),
                      e.g. "S009/level_100mps/lv0/trial_01"

        Returns:
            MotionCaptureData object with IMU data
        """
        file_ext = Path(filepath).suffix.lower()

        if file_ext == '.csv':
            return FileHandler._import_csv(filepath)
        elif file_ext == '.txt':
            return FileHandler._import_txt(filepath)
        elif file_ext == '.h5':
            if h5_path is None:
                raise ValueError("h5_path is required for HDF5 import")
            return FileHandler.import_h5_trial(filepath, h5_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    @staticmethod
    def _import_csv(filepath: str) -> MotionCaptureData:
        """
        Import CSV format with IMU data
        
        CSV Format:
        - TrunkIMU: TrunkIMU_LocalAccX/Y/Z, TrunkIMU_LocalGyrX/Y/Z, TrunkIMU_QuatW/X/Y/Z
        - Left Thigh: L_THIGH_IMU_QuatW/X/Y/Z, L_THIGH_IMU_AccX/Y/Z, L_THIGH_IMU_GyrX/Y/Z
        - Left Shank: L_SHANK_IMU_QuatW/X/Y/Z, L_SHANK_IMU_AccX/Y/Z, L_SHANK_IMU_GyrX/Y/Z
        - Left Foot: L_FOOT_IMU_QuatW/X/Y/Z, L_FOOT_IMU_AccX/Y/Z, L_FOOT_IMU_GyrX/Y/Z
        - Right Thigh: R_THIGH_IMU_QuatW/X/Y/Z, R_THIGH_IMU_AccX/Y/Z, R_THIGH_IMU_GyrX/Y/Z
        - Right Shank: R_SHANK_IMU_QuatW/X/Y/Z, R_SHANK_IMU_AccX/Y/Z, R_SHANK_IMU_GyrX/Y/Z
        - Right Foot: R_FOOT_IMU_QuatW/X/Y/Z, R_FOOT_IMU_AccX/Y/Z, R_FOOT_IMU_GyrX/Y/Z
        """
        import pandas as pd
        
        session_id = Path(filepath).stem
        data = MotionCaptureData(
            session_id=session_id,
            creation_time=datetime.now()
        )
        
        print(f"Importing CSV from {filepath}")
        
        # Read CSV file
        df = pd.read_csv(filepath)
        n_samples = len(df)
        
        # Create time array with 500Hz sampling rate
        # Time starts at 0 seconds
        if 'LoopCnt' in df.columns:
            loop_cnt = df['LoopCnt'].values
            # Actual sampling frequency: 500Hz
            sampling_freq = 500.0
            # Calculate timestamps and normalize to start at 0
            timestamps = (loop_cnt - loop_cnt[0]) / sampling_freq
        else:
            # Generate timestamps starting at 0, assuming 500Hz
            sampling_freq = 500.0
            timestamps = np.arange(n_samples) / sampling_freq
        
        print(f"  Sampling frequency: {sampling_freq} Hz")
        print(f"  Time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s")
        
        # Sensor mapping
        current_mode = app_settings.mode.mode_type
        if current_mode == 'Upper-body':
            sensor_configs = [
                {
                    'location': 'head',
                    'sensor_id': 0,
                    'quat_cols': ['HeadIMU_QuatW', 'HeadIMU_QuatX', 'HeadIMU_QuatY', 'HeadIMU_QuatZ'],
                    'acc_cols': ['HeadIMU_LocalAccX', 'HeadIMU_LocalAccY', 'HeadIMU_LocalAccZ'],
                    'gyr_cols': ['HeadIMU_LocalGyrX', 'HeadIMU_LocalGyrY', 'HeadIMU_LocalGyrZ']
                },
                {
                    'location': 'pelvis',
                    'sensor_id': 1,
                    'quat_cols': ['Pelvis_IMU_QuatW', 'Pelvis_IMU_QuatX', 'Pelvis_IMU_QuatY', 'Pelvis_IMU_QuatZ'],
                    'acc_cols': ['Pelvis_IMU_AccX', 'Pelvis_IMU_AccY', 'Pelvis_IMU_AccZ'],
                    'gyr_cols': ['Pelvis_IMU_GyrX', 'Pelvis_IMU_GyrY', 'Pelvis_IMU_GyrZ']
                },
                {
                    'location': 'upperarm_left',
                    'sensor_id': 2,
                    'quat_cols': ['L_Upperarm_IMU_QuatW', 'L_Upperarm_IMU_QuatX', 'L_Upperarm_IMU_QuatY', 'L_Upperarm_IMU_QuatZ'],
                    'acc_cols': ['L_Upperarm_IMU_AccX', 'L_Upperarm_IMU_AccY', 'L_Upperarm_IMU_AccZ'],
                    'gyr_cols': ['L_Upperarm_IMU_GyrX', 'L_Upperarm_IMU_GyrY', 'L_Upperarm_IMU_GyrZ']
                },
                {
                    'location': 'lowerarm_left',
                    'sensor_id': 3,
                    'quat_cols': ['L_Lowerarm_IMU_QuatW', 'L_Lowerarm_IMU_QuatX', 'L_Lowerarm_IMU_QuatY', 'L_Lowerarm_IMU_QuatZ'],
                    'acc_cols': ['L_Lowerarm_IMU_AccX', 'L_Lowerarm_IMU_AccY', 'L_Lowerarm_IMU_AccZ'],
                    'gyr_cols': ['L_Lowerarm_IMU_GyrX', 'L_Lowerarm_IMU_GyrY', 'L_Lowerarm_IMU_GyrZ']
                },
                {
                    'location': 'chest',
                    'sensor_id': 4,
                    'quat_cols': ['Chest_IMU_QuatW', 'Chest_IMU_QuatX', 'Chest_IMU_QuatY', 'Chest_IMU_QuatZ'],
                    'acc_cols': ['Chest_IMU_AccX', 'Chest_IMU_AccY', 'Chest_IMU_AccZ'],
                    'gyr_cols': ['Chest_IMU_GyrX', 'Chest_IMU_GyrY', 'Chest_IMU_GyrZ']
                },
                {
                    'location': 'upperarm_right',
                    'sensor_id': 5,
                    'quat_cols': ['R_Upperarm_IMU_QuatW', 'R_Upperarm_IMU_QuatX', 'R_Upperarm_IMU_QuatY', 'R_Upperarm_IMU_QuatZ'],
                    'acc_cols': ['R_Upperarm_IMU_AccX', 'R_Upperarm_IMU_AccY', 'R_Upperarm_IMU_AccZ'],
                    'gyr_cols': ['R_Upperarm_IMU_GyrX', 'R_Upperarm_IMU_GyrY', 'R_Upperarm_IMU_GyrZ']
                },
                {
                    'location': 'lowerarm_right',
                    'sensor_id': 6,
                    'quat_cols': ['R_Lowerarm_IMU_QuatW', 'R_Lowerarm_IMU_QuatX', 'R_Lowerarm_IMU_QuatY', 'R_Lowerarm_IMU_QuatZ'],
                    'acc_cols': ['R_Lowerarm_IMU_AccX', 'R_Lowerarm_IMU_AccY', 'R_Lowerarm_IMU_AccZ'],
                    'gyr_cols': ['R_Lowerarm_IMU_GyrX', 'R_Lowerarm_IMU_GyrY', 'R_Lowerarm_IMU_GyrZ']
                }
            ]
        else:
            sensor_configs = [
                {
                    'location': 'back',
                    'sensor_id': 0,
                    'quat_cols': ['TrunkIMU_QuatW', 'TrunkIMU_QuatX', 'TrunkIMU_QuatY', 'TrunkIMU_QuatZ'],
                    'acc_cols': ['TrunkIMU_LocalAccX', 'TrunkIMU_LocalAccY', 'TrunkIMU_LocalAccZ'],
                    'gyr_cols': ['TrunkIMU_LocalGyrX', 'TrunkIMU_LocalGyrY', 'TrunkIMU_LocalGyrZ']
                },
                {
                    'location': 'thigh_left',
                    'sensor_id': 1,
                    'quat_cols': ['L_THIGH_IMU_QuatW', 'L_THIGH_IMU_QuatX', 'L_THIGH_IMU_QuatY', 'L_THIGH_IMU_QuatZ'],
                    'acc_cols': ['L_THIGH_IMU_AccX', 'L_THIGH_IMU_AccY', 'L_THIGH_IMU_AccZ'],
                    'gyr_cols': ['L_THIGH_IMU_GyrX', 'L_THIGH_IMU_GyrY', 'L_THIGH_IMU_GyrZ']
                },
                {
                    'location': 'shank_left',
                    'sensor_id': 2,
                    'quat_cols': ['L_SHANK_IMU_QuatW', 'L_SHANK_IMU_QuatX', 'L_SHANK_IMU_QuatY', 'L_SHANK_IMU_QuatZ'],
                    'acc_cols': ['L_SHANK_IMU_AccX', 'L_SHANK_IMU_AccY', 'L_SHANK_IMU_AccZ'],
                    'gyr_cols': ['L_SHANK_IMU_GyrX', 'L_SHANK_IMU_GyrY', 'L_SHANK_IMU_GyrZ']
                },
                {
                    'location': 'foot_left',
                    'sensor_id': 3,
                    'quat_cols': ['L_FOOT_IMU_QuatW', 'L_FOOT_IMU_QuatX', 'L_FOOT_IMU_QuatY', 'L_FOOT_IMU_QuatZ'],
                    'acc_cols': ['L_FOOT_IMU_AccX', 'L_FOOT_IMU_AccY', 'L_FOOT_IMU_AccZ'],
                    'gyr_cols': ['L_FOOT_IMU_GyrX', 'L_FOOT_IMU_GyrY', 'L_FOOT_IMU_GyrZ']
                },
                {
                    'location': 'thigh_right',
                    'sensor_id': 4,
                    'quat_cols': ['R_THIGH_IMU_QuatW', 'R_THIGH_IMU_QuatX', 'R_THIGH_IMU_QuatY', 'R_THIGH_IMU_QuatZ'],
                    'acc_cols': ['R_THIGH_IMU_AccX', 'R_THIGH_IMU_AccY', 'R_THIGH_IMU_AccZ'],
                    'gyr_cols': ['R_THIGH_IMU_GyrX', 'R_THIGH_IMU_GyrY', 'R_THIGH_IMU_GyrZ']
                },
                {
                    'location': 'shank_right',
                    'sensor_id': 5,
                    'quat_cols': ['R_SHANK_IMU_QuatW', 'R_SHANK_IMU_QuatX', 'R_SHANK_IMU_QuatY', 'R_SHANK_IMU_QuatZ'],
                    'acc_cols': ['R_SHANK_IMU_AccX', 'R_SHANK_IMU_AccY', 'R_SHANK_IMU_AccZ'],
                    'gyr_cols': ['R_SHANK_IMU_GyrX', 'R_SHANK_IMU_GyrY', 'R_SHANK_IMU_GyrZ']
                },
                {
                    'location': 'foot_right',
                    'sensor_id': 6,
                    'quat_cols': ['R_FOOT_IMU_QuatW', 'R_FOOT_IMU_QuatX', 'R_FOOT_IMU_QuatY', 'R_FOOT_IMU_QuatZ'],
                    'acc_cols': ['R_FOOT_IMU_AccX', 'R_FOOT_IMU_AccY', 'R_FOOT_IMU_AccZ'],
                    'gyr_cols': ['R_FOOT_IMU_GyrX', 'R_FOOT_IMU_GyrY', 'R_FOOT_IMU_GyrZ']
                }
            ]
        
        # Parse each sensor
        for config in sensor_configs:
            # Check if all required columns exist
            all_cols = config['quat_cols'] + config['acc_cols'] + config['gyr_cols']
            if not all(col in df.columns for col in all_cols):
                print(f"Warning: Skipping {config['location']} - missing columns")
                continue
            
            # Extract data
            quaternions = df[config['quat_cols']].values  # (N, 4) [w, x, y, z]
            accelerations = df[config['acc_cols']].values  # (N, 3)
            gyroscopes = df[config['gyr_cols']].values  # (N, 3)
            
            # Fix back IMU: Replace all-zero quaternions with identity quaternion [1, 0, 0, 0]
            # This prevents "zero norm" errors in quaternion operations
            if config['location'] == 'back' and np.all(quaternions == 0):
                print(f"  Note: Back quaternions are zero, replacing with identity [1,0,0,0]")
                quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (n_samples, 1))
            
            # Create IMUSensorData
            sensor_data = IMUSensorData(
                sensor_id=config['sensor_id'],
                location=config['location'],
                timestamps=timestamps.copy(),
                quaternions=quaternions,
                accelerations=accelerations,
                gyroscopes=gyroscopes,
                sampling_frequency=sampling_freq
            )
            
            data.add_imu_sensor_data(sensor_data)
            print(f"  Loaded {config['location']}: {n_samples} samples")
        
        print(f"Successfully imported {len(data.imu_data)} sensors")
        return data
    
    @staticmethod
    def _import_txt(filepath: str) -> MotionCaptureData:
        """Import TXT format (placeholder)"""
        # TODO: Implement TXT parsing
        session_id = Path(filepath).stem
        data = MotionCaptureData(
            session_id=session_id,
            creation_time=datetime.now()
        )
        
        print(f"Importing TXT from {filepath}")
        
        return data

    @staticmethod
    def _get_h5_open_candidates(filepath: str) -> List[str]:
        """Build candidate paths for robust HDF5 open on Windows unicode paths."""
        abs_path = os.path.abspath(filepath)
        candidates = [filepath, abs_path]

        if os.name == 'nt':
            try:
                rel_path = os.path.relpath(abs_path, os.getcwd())
                if not rel_path.startswith('..') and os.path.exists(rel_path):
                    candidates.append(rel_path)
            except Exception:
                pass

            try:
                import ctypes
                short_buffer = ctypes.create_unicode_buffer(32768)
                if ctypes.windll.kernel32.GetShortPathNameW(abs_path, short_buffer, len(short_buffer)):
                    short_path = short_buffer.value
                    if short_path:
                        candidates.append(short_path)
            except Exception:
                pass

        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _open_h5_file(filepath: str, mode: str = 'r'):
        """Open HDF5 file with fallbacks for environments that fail on unicode absolute paths."""
        import h5py

        open_errors = []
        for candidate_path in FileHandler._get_h5_open_candidates(filepath):
            try:
                return h5py.File(candidate_path, mode)
            except OSError as e:
                open_errors.append(f"{candidate_path} -> {e}")

        raise OSError("Unable to open HDF5 file. " + " | ".join(open_errors))

    @staticmethod
    def _resolve_h5_marker_group(marker_group, marker_name: str):
        """Resolve a marker group, accepting legacy lowercase and new uppercase names."""
        candidates = [marker_name, marker_name.lower(), marker_name.upper()]
        for candidate in candidates:
            if candidate in marker_group:
                return marker_group[candidate], candidate

        lower_lookup = {str(key).lower(): key for key in marker_group.keys()}
        resolved = lower_lookup.get(marker_name.lower())
        if resolved is not None:
            return marker_group[resolved], resolved

        raise KeyError(f"Missing marker: {marker_name}")

    @staticmethod
    def _fit_h5_timeseries_length(values: np.ndarray, expected_len: int, label: str) -> np.ndarray:
        """Trim H5 series to the selected timeline; short series are treated as invalid."""
        arr = np.asarray(values, dtype=float)
        if len(arr) == expected_len:
            return arr
        if len(arr) > expected_len:
            print(
                f"  Warning: Trimming {label} from {len(arr)} to {expected_len} samples"
            )
            return arr[:expected_len]
        raise ValueError(
            f"Length mismatch for {label}: expected at least {expected_len}, data={len(arr)}"
        )

    @staticmethod
    def _read_h5_marker_xyz(marker_group, marker_name: str, expected_len: int = None) -> np.ndarray:
        """Read one mocap marker as a raw (N, 3) xyz array."""
        marker, resolved_name = FileHandler._resolve_h5_marker_group(marker_group, marker_name)

        coords = []
        lengths = []
        for axis in ('x', 'y', 'z'):
            if axis not in marker:
                raise KeyError(f"Missing marker dataset: {resolved_name}/{axis}")
            values = np.asarray(marker[axis][:], dtype=float).reshape(-1)
            coords.append(values)
            lengths.append(len(values))

        target_len = expected_len if expected_len is not None else min(lengths)
        fitted = [
            FileHandler._fit_h5_timeseries_length(values, target_len, f"{resolved_name}/{axis}")
            for values, axis in zip(coords, ('x', 'y', 'z'))
        ]
        return np.column_stack(fitted)

    @staticmethod
    def _import_h5_marker_data(trial, timestamps: np.ndarray, sampling_freq: float) -> Optional[MarkerData]:
        """Import only the markers used for marker-based segment absolute angles."""
        marker_path = 'mocap/marker'
        if marker_path not in trial:
            return None

        marker_group = trial[marker_path]
        marker_lengths = []
        for marker_name in FileHandler.MARKER_VISUALIZATION_NAMES:
            try:
                marker_lengths.append(
                    len(FileHandler._read_h5_marker_xyz(marker_group, marker_name))
                )
            except Exception as e:
                print(f"  Warning: Marker {marker_name} unavailable: {e}")

        if not marker_lengths:
            return None

        marker_len = min(marker_lengths)
        markers = {}
        for marker_name in FileHandler.MARKER_VISUALIZATION_NAMES:
            try:
                markers[marker_name] = FileHandler._read_h5_marker_xyz(
                    marker_group,
                    marker_name,
                    marker_len
                )
            except Exception as e:
                print(f"  Warning: Skipping marker {marker_name}: {e}")

        if not markers:
            return None

        if len(timestamps) >= marker_len:
            marker_timestamps = timestamps[:marker_len].copy()
        else:
            marker_timestamps = np.arange(marker_len, dtype=float) / sampling_freq

        if marker_len != len(timestamps):
            print(
                f"  Warning: Mocap marker length differs from IMU timeline "
                f"(markers={marker_len}, imu={len(timestamps)}); using marker-specific timeline"
            )

        print(f"  Loaded mocap markers for visualization: {len(markers)} markers")
        return MarkerData(
            timestamps=marker_timestamps,
            markers=markers,
            sampling_frequency=sampling_freq
        )

    @staticmethod
    def _resolve_h5_group(container, candidate_paths: List[str]):
        """Resolve the first existing group path inside an HDF5 container."""
        for candidate in candidate_paths:
            try:
                return container[candidate], candidate
            except KeyError:
                continue
        return None, None

    @staticmethod
    def _looks_like_h5_trial_group(group) -> bool:
        """Return True when a group looks like an importable motion trial."""
        return any(key in group for key in ('robot', 'mocap', 'common', 'kin_q'))

    @staticmethod
    def _get_h5_sensor_sample_count(sensor_grp, resolved_path: str) -> int:
        """Return the IMU sample count after checking the core quaternion datasets."""
        quat_keys = ['quat_w', 'quat_x', 'quat_y', 'quat_z']
        missing = [key for key in quat_keys if key not in sensor_grp]
        if missing:
            raise ValueError(f"Sensor {resolved_path} missing datasets: {missing}")

        lengths = [len(np.asarray(sensor_grp[key][:]).reshape(-1)) for key in quat_keys]
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent quaternion lengths for {resolved_path}: {lengths}")
        return lengths[0]

    @staticmethod
    def _score_h5_stand_trial_motion(trial) -> float:
        """
        Score how still a stand trial is.

        Lower scores are better. The score is the median, across available IMUs,
        of the 95th percentile quaternion angle change from the first frame.
        """
        sensor_scores = []

        for _, _, candidate_paths in FileHandler.H5_TRIAL_SENSOR_MAP:
            sensor_grp, _ = FileHandler._resolve_h5_group(trial, candidate_paths)
            if sensor_grp is None:
                continue

            quat_keys = ['quat_w', 'quat_x', 'quat_y', 'quat_z']
            if any(key not in sensor_grp for key in quat_keys):
                continue

            quaternions = np.column_stack([
                np.asarray(sensor_grp[key][:], dtype=float).reshape(-1)
                for key in quat_keys
            ])
            if len(quaternions) < 2:
                continue

            stride = max(1, len(quaternions) // 5000)
            q = quaternions[::stride]
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            valid = np.isfinite(q).all(axis=1) & (norms[:, 0] > 1e-12)
            q = q[valid] / norms[valid]
            if len(q) < 2:
                continue

            q0 = q[0]
            dots = np.abs(q @ q0)
            dots = np.clip(dots, -1.0, 1.0)
            angle_change = np.degrees(2.0 * np.arccos(dots))
            sensor_scores.append(float(np.nanpercentile(angle_change, 95)))

        if not sensor_scores:
            return float('inf')

        return float(np.nanmedian(sensor_scores))

    @staticmethod
    def _resolve_h5_trial_timestamps(trial, h5_path: str) -> Tuple[np.ndarray, float, str]:
        """
        Resolve a trial timestamp vector.

        Legacy S009 H5 files provide common/time. The combined ImuMarker H5 does
        not, so use a generated 100 Hz timeline from the available IMU length.
        """
        if 'common/time' in trial:
            time_values = np.asarray(trial['common/time'][:], dtype=float).reshape(-1)
            if len(time_values) == 0:
                raise ValueError(f"Empty common/time dataset for {h5_path}")

            raw_dt = np.diff(time_values)
            if np.any(raw_dt <= 0):
                n_bad = np.sum(raw_dt <= 0)
                print(f"  Warning: {n_bad} non-monotonic time steps detected")

            timestamps = time_values - time_values[0]
            sampling_freq = 100.0
            if len(raw_dt) > 0:
                median_dt = float(np.nanmedian(raw_dt))
                if np.isfinite(median_dt) and median_dt > 0:
                    if median_dt > 1.0:
                        timestamps = timestamps / 1000.0
                        sampling_freq = 1000.0 / median_dt
                    else:
                        sampling_freq = 1.0 / median_dt
                    if abs(sampling_freq - 100.0) > 10.0:
                        print(
                            f"  Warning: Estimated sampling frequency is "
                            f"{sampling_freq:.2f} Hz (expected about 100 Hz)"
                        )

            return timestamps, sampling_freq, "common/time"

        counts = []
        for _, _, candidate_paths in FileHandler.H5_TRIAL_SENSOR_MAP:
            sensor_grp, resolved_path = FileHandler._resolve_h5_group(trial, candidate_paths)
            if sensor_grp is None:
                continue
            counts.append(FileHandler._get_h5_sensor_sample_count(sensor_grp, resolved_path))

        if not counts:
            raise ValueError(f"No usable IMU timestamp source found for {h5_path}")

        n_samples = min(counts)
        if len(set(counts)) != 1:
            print(
                f"  Warning: IMU sensor lengths differ for {h5_path}: {counts}; "
                f"using shortest length {n_samples}"
            )

        sampling_freq = 100.0
        timestamps = np.arange(n_samples, dtype=float) / sampling_freq
        return timestamps, sampling_freq, "generated 100 Hz"

    @staticmethod
    def _build_h5_imu_sensor_data(
        sensor_grp,
        location: str,
        sensor_id: int,
        timestamps: np.ndarray,
        sampling_freq: float,
        resolved_path: str
    ) -> IMUSensorData:
        """Build one IMUSensorData object from an HDF5 IMU group."""
        required = [
            'quat_w', 'quat_x', 'quat_y', 'quat_z',
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z'
        ]
        missing = [d for d in required if d not in sensor_grp]
        if missing:
            raise ValueError(f"Sensor {resolved_path} missing datasets: {missing}")

        quaternions = np.column_stack([
            sensor_grp['quat_w'][:],
            sensor_grp['quat_x'][:],
            sensor_grp['quat_y'][:],
            sensor_grp['quat_z'][:]
        ])

        accelerations = np.column_stack([
            sensor_grp['accel_x'][:],
            sensor_grp['accel_y'][:],
            sensor_grp['accel_z'][:]
        ])

        gyroscopes = np.column_stack([
            sensor_grp['gyro_x'][:],
            sensor_grp['gyro_y'][:],
            sensor_grp['gyro_z'][:]
        ])

        n_samples = len(timestamps)
        quaternions = FileHandler._fit_h5_timeseries_length(
            quaternions, n_samples, f"{resolved_path}/quaternion"
        )
        accelerations = FileHandler._fit_h5_timeseries_length(
            accelerations, n_samples, f"{resolved_path}/acceleration"
        )
        gyroscopes = FileHandler._fit_h5_timeseries_length(
            gyroscopes, n_samples, f"{resolved_path}/gyroscope"
        )

        return IMUSensorData(
            sensor_id=sensor_id,
            location=location,
            timestamps=timestamps.copy(),
            quaternions=quaternions,
            accelerations=accelerations,
            gyroscopes=gyroscopes,
            sampling_frequency=sampling_freq
        )

    @staticmethod
    def import_h5_neutral_pose(filepath: str, subject_id: str) -> MotionCaptureData:
        """
        Import subject-level neutral pose IMU data from an HDF5 file.

        The neutral pose is stored outside motion trials at
        ``<subject_id>/sub_info/neutral_pose``. Use it as the shared calibration
        source for that subject instead of treating each trial's first frames as
        a trial-specific calibration pose.
        """
        neutral_path = f"{subject_id}/sub_info/neutral_pose"
        data = MotionCaptureData(
            session_id=f"{subject_id}_neutral_pose",
            creation_time=datetime.now(),
            subject_id=subject_id
        )

        print(f"Importing HDF5 neutral pose: {filepath} [{neutral_path}]")

        with FileHandler._open_h5_file(filepath, 'r') as f:
            if neutral_path not in f:
                raise ValueError(f"Neutral pose path '{neutral_path}' not found in HDF5 file")

            neutral = f[neutral_path]
            first_group = None
            for _, _, candidate_paths in FileHandler.H5_NEUTRAL_POSE_SENSOR_MAP:
                first_group, _ = FileHandler._resolve_h5_group(neutral, candidate_paths)
                if first_group is not None:
                    break

            if first_group is None or 'quat_w' not in first_group:
                raise ValueError(f"No usable neutral-pose IMU data found at '{neutral_path}'")

            n_samples = len(first_group['quat_w'])
            sampling_freq = 100.0
            timestamps = np.arange(n_samples, dtype=float) / sampling_freq

            print(f"  Time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s ({n_samples} samples)")
            print(f"  Sampling frequency: {sampling_freq} Hz (generated)")

            for location, sensor_id, candidate_paths in FileHandler.H5_NEUTRAL_POSE_SENSOR_MAP:
                sensor_grp, resolved_path = FileHandler._resolve_h5_group(neutral, candidate_paths)
                if sensor_grp is None:
                    print(
                        f"  Warning: Neutral-pose sensor for {location} not found "
                        f"(tried: {', '.join(candidate_paths)}), skipping"
                    )
                    continue

                sensor_data = FileHandler._build_h5_imu_sensor_data(
                    sensor_grp,
                    location,
                    sensor_id,
                    timestamps,
                    sampling_freq,
                    f"{neutral_path}/{resolved_path}"
                )
                data.add_imu_sensor_data(sensor_data)
                print(f"  Loaded neutral {location}: {n_samples} samples")

        print(f"Successfully imported {len(data.imu_data)} neutral-pose sensors from HDF5")
        return data

    @staticmethod
    def find_h5_calibration_pose(filepath: str, subject_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the best H5 calibration source for a subject.

        Preference order:
        1. <subject>/sub_info/neutral_pose for legacy per-subject calibration.
        2. The stillest stand trial for combined ImuMarker.
        """
        neutral_path = f"{subject_id}/sub_info/neutral_pose"

        with FileHandler._open_h5_file(filepath, 'r') as f:
            if neutral_path in f:
                return neutral_path, "neutral_pose"

            stand_root = f"{subject_id}/stand"
            if stand_root not in f:
                return None, None

            stand_group = f[stand_root]
            stand_candidates = []
            for level in sorted(stand_group.keys()):
                level_group = stand_group[level]
                if not hasattr(level_group, 'keys'):
                    continue
                for trial_id in sorted(level_group.keys()):
                    trial_group = level_group[trial_id]
                    if hasattr(trial_group, 'keys') and FileHandler._looks_like_h5_trial_group(trial_group):
                        stand_path = f"{stand_root}/{level}/{trial_id}"
                        score = FileHandler._score_h5_stand_trial_motion(trial_group)
                        stand_candidates.append((score, stand_path))

            if stand_candidates:
                stand_candidates.sort(key=lambda item: (item[0], item[1]))
                best_score, best_path = stand_candidates[0]
                if len(stand_candidates) > 1:
                    summary = ", ".join(
                        f"{path}={score:.2f}deg"
                        for score, path in stand_candidates[:5]
                    )
                    if len(stand_candidates) > 5:
                        summary += ", ..."
                    print(
                        f"  Selected stand calibration for {subject_id}: "
                        f"{best_path} (motion score={best_score:.2f}deg; candidates: {summary})"
                    )
                return best_path, "stand_trial"

        return None, None

    @staticmethod
    def import_h5_calibration_pose(filepath: str, subject_id: str) -> Tuple[MotionCaptureData, str, str]:
        """Import the selected H5 calibration source for a subject."""
        calibration_path, source_type = FileHandler.find_h5_calibration_pose(filepath, subject_id)
        if calibration_path is None or source_type is None:
            raise ValueError(f"No H5 calibration pose found for {subject_id}")

        if source_type == "neutral_pose":
            return FileHandler.import_h5_neutral_pose(filepath, subject_id), source_type, calibration_path

        calibration_data = FileHandler.import_h5_trial(filepath, calibration_path)
        calibration_data.session_id = f"{subject_id}_stand_calibration"
        calibration_data.subject_id = subject_id
        return calibration_data, source_type, calibration_path
    
    @staticmethod
    def import_h5_trial(filepath: str, h5_path: str) -> MotionCaptureData:
        """
        Import a single trial from an HDF5 file.

        HDF5 structure: Subject > Activity > Level > Trial, with sensor data under
        trial/robot/imu/ (new) or trial/robot/ (legacy). Quaternions are stored as
        quat_w/x/y/z (scalar-first [w,x,y,z]), matching this project's convention.

        Args:
            filepath: Path to HDF5 file (.h5)
            h5_path: Internal path to trial, e.g. "S009/level_100mps/lv0/trial_01"

        Returns:
            MotionCaptureData with 7 IMU sensors at 100 Hz
        """
        import h5py

        session_id = h5_path.replace('/', '_')
        data = MotionCaptureData(
            session_id=session_id,
            creation_time=datetime.now()
        )

        # Extract subject_id from h5_path (first component)
        subject_id = h5_path.split('/')[0]
        data.subject_id = subject_id

        print(f"Importing HDF5 trial: {filepath} [{h5_path}]")

        with FileHandler._open_h5_file(filepath, 'r') as f:
            if h5_path not in f:
                raise ValueError(f"Trial path '{h5_path}' not found in HDF5 file")

            trial = f[h5_path]

            # Read timestamps (milliseconds → seconds, normalized to start at 0)
            timestamps, sampling_freq, timestamp_source = FileHandler._resolve_h5_trial_timestamps(
                trial,
                h5_path
            )
            n_samples = len(timestamps)

            print(f"  Time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s ({n_samples} samples)")
            print(f"  Sampling frequency: {sampling_freq:.2f} Hz ({timestamp_source})")

            sensor_map = FileHandler.H5_TRIAL_SENSOR_MAP

            for location, sensor_id, candidate_paths in sensor_map:
                sensor_grp, resolved_path = FileHandler._resolve_h5_group(trial, candidate_paths)

                if sensor_grp is None:
                    print(
                        f"  Warning: Sensor for {location} not found "
                        f"(tried: {', '.join(candidate_paths)}), skipping"
                    )
                    continue

                # Check required datasets exist
                required = ['quat_w', 'quat_x', 'quat_y', 'quat_z',
                            'accel_x', 'accel_y', 'accel_z',
                            'gyro_x', 'gyro_y', 'gyro_z']
                missing = [d for d in required if d not in sensor_grp]
                if missing:
                    raise ValueError(
                        f"Sensor {resolved_path} missing datasets: {missing}")

                # Stack quaternions [w, x, y, z] — scalar-first, matching project convention
                quaternions = np.column_stack([
                    sensor_grp['quat_w'][:],
                    sensor_grp['quat_x'][:],
                    sensor_grp['quat_y'][:],
                    sensor_grp['quat_z'][:]
                ])

                accelerations = np.column_stack([
                    sensor_grp['accel_x'][:],
                    sensor_grp['accel_y'][:],
                    sensor_grp['accel_z'][:]
                ])

                gyroscopes = np.column_stack([
                    sensor_grp['gyro_x'][:],
                    sensor_grp['gyro_y'][:],
                    sensor_grp['gyro_z'][:]
                ])

                quaternions = FileHandler._fit_h5_timeseries_length(
                    quaternions, n_samples, f"{resolved_path}/quaternion"
                )
                accelerations = FileHandler._fit_h5_timeseries_length(
                    accelerations, n_samples, f"{resolved_path}/acceleration"
                )
                gyroscopes = FileHandler._fit_h5_timeseries_length(
                    gyroscopes, n_samples, f"{resolved_path}/gyroscope"
                )

                sensor_data = IMUSensorData(
                    sensor_id=sensor_id,
                    location=location,
                    timestamps=timestamps.copy(),
                    quaternions=quaternions,
                    accelerations=accelerations,
                    gyroscopes=gyroscopes,
                    sampling_frequency=sampling_freq
                )
                data.add_imu_sensor_data(sensor_data)
                print(f"  Loaded {location}: {n_samples} samples")

            try:
                data.marker_data = FileHandler._import_h5_marker_data(
                    trial,
                    timestamps,
                    sampling_freq
                )
            except Exception as e:
                print(f"  Warning: Failed to load mocap marker visualization data: {e}")

        print(f"Successfully imported {len(data.imu_data)} sensors from HDF5")
        return data

    @staticmethod
    def scan_h5_file(filepath: str) -> dict:
        """
        Scan an HDF5 file and return its hierarchical structure.

        Returns:
            Dict of {subject_id: {activity: {level: [trial_ids]}}}
        """
        import h5py

        structure = {}

        try:
            with FileHandler._open_h5_file(filepath, 'r') as f:
                for subject_id in f:
                    subject_grp = f[subject_id]
                    if not isinstance(subject_grp, h5py.Group):
                        continue

                    structure[subject_id] = {}

                    for activity in subject_grp:
                        if activity == 'sub_info':
                            continue
                        activity_grp = subject_grp[activity]
                        if not isinstance(activity_grp, h5py.Group):
                            continue

                        structure[subject_id][activity] = {}

                        for level in activity_grp:
                            level_grp = activity_grp[level]
                            if not isinstance(level_grp, h5py.Group):
                                continue

                            trials = sorted([
                                t for t in level_grp
                                if (
                                    isinstance(level_grp[t], h5py.Group)
                                    and FileHandler._looks_like_h5_trial_group(level_grp[t])
                                )
                            ])
                            structure[subject_id][activity][level] = trials

        except Exception as e:
            raise ValueError(f"Failed to scan HDF5 file: {e}")

        return structure

    @staticmethod
    def load_h5_subject_info(filepath: str, subject_id: str) -> dict:
        """
        Read subject information from an HDF5 file.

        Args:
            filepath: Path to HDF5 file
            subject_id: Subject group name (e.g. "S009")

        Returns:
            Dict with keys: age, height (cm), weight (kg), sex
        """
        import h5py

        with FileHandler._open_h5_file(filepath, 'r') as f:
            si_path = f"{subject_id}/sub_info"
            if si_path not in f:
                return {}

            si = f[si_path]
            info = {}

            # Read scalar datasets, decode byte strings
            if 'age' in si and isinstance(si['age'], h5py.Dataset):
                info['age'] = int(si['age'][()].decode('utf-8'))
            if 'height' in si and isinstance(si['height'], h5py.Dataset):
                height_mm = float(si['height'][()].decode('utf-8'))
                info['height'] = height_mm / 10.0  # mm → cm
            if 'weight' in si and isinstance(si['weight'], h5py.Dataset):
                info['weight'] = float(si['weight'][()].decode('utf-8'))
            if 'sex' in si and isinstance(si['sex'], h5py.Dataset):
                sex_val = si['sex'][()].decode('utf-8')
                info['sex'] = 'male' if sex_val == '0' else 'female'

            # Validate
            if info.get('height', 0) <= 0:
                print(f"  Warning: Invalid height: {info.get('height')}")
            if info.get('weight', 0) <= 0:
                print(f"  Warning: Invalid weight: {info.get('weight')}")

            print(f"  Subject info: height={info.get('height')}cm, "
                  f"weight={info.get('weight')}kg, age={info.get('age')}")

        return info

    @staticmethod
    def save_processed_data(data: MotionCaptureData, filepath: str):
        """
        Save processed motion capture data
        
        Args:
            data: MotionCaptureData object
            filepath: Path to save file (with .mcp extension)
        """
        # Ensure .mcp extension
        if not filepath.endswith('.mcp'):
            filepath += '.mcp'
        
        # Prepare data for serialization
        save_dict = {
            'version': '1.0',
            'session_id': data.session_id,
            'creation_time': data.creation_time.isoformat(),
            'subject_id': data.subject_id,
            'calibration_pose': data.calibration_pose,
            'calibration_duration': data.calibration_duration,
            'calibration_start_time': data.calibration_start_time,
            'is_processed': data.is_processed,
            'processing_timestamp': data.processing_timestamp.isoformat() if data.processing_timestamp else None,
            'notes': data.notes,
            'imu_data': {},
            'joint_angles': None,
            'kinematics': None,
            'marker_data': None
        }
        
        # Save IMU data
        for location, sensor_data in data.imu_data.items():
            save_dict['imu_data'][location] = {
                'sensor_id': sensor_data.sensor_id,
                'location': sensor_data.location,
                'timestamps': sensor_data.timestamps.tolist(),
                'quaternions': sensor_data.quaternions.tolist(),
                'accelerations': sensor_data.accelerations.tolist(),
                'gyroscopes': sensor_data.gyroscopes.tolist(),
                'sampling_frequency': sensor_data.sampling_frequency
            }
        
        # Save joint angles if available
        if data.joint_angles:
            save_dict['joint_angles'] = {
                'timestamps': data.joint_angles.timestamps.tolist(),
                'hip_right': data.joint_angles.hip_right.tolist(),
                'hip_left': data.joint_angles.hip_left.tolist(),
                'knee_right': data.joint_angles.knee_right.tolist(),
                'knee_left': data.joint_angles.knee_left.tolist(),
                'ankle_right': data.joint_angles.ankle_right.tolist(),
                'ankle_left': data.joint_angles.ankle_left.tolist()
            }
        
        # Save kinematics if available
        if data.kinematics:
            save_dict['kinematics'] = {
                'timestamps': data.kinematics.timestamps.tolist(),
                'back_angle': data.kinematics.back_angle.tolist(),
                'foot_contact_right': data.kinematics.foot_contact_right.tolist(),
                'foot_contact_left': data.kinematics.foot_contact_left.tolist(),
                'back_velocity': data.kinematics.back_velocity.tolist(),
                'back_speed': data.kinematics.back_speed.tolist(),
                'stride_times_right': data.kinematics.stride_times_right,
                'stride_times_left': data.kinematics.stride_times_left
            }

        if data.marker_data:
            save_dict['marker_data'] = {
                'timestamps': data.marker_data.timestamps.tolist(),
                'sampling_frequency': data.marker_data.sampling_frequency,
                'markers': {
                    name: values.tolist()
                    for name, values in data.marker_data.markers.items()
                }
            }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Saved processed data to {filepath}")
    
    @staticmethod
    def load_processed_data(filepath: str) -> MotionCaptureData:
        """
        Load processed motion capture data
        
        Args:
            filepath: Path to .mcp file
            
        Returns:
            MotionCaptureData object
        """
        with open(filepath, 'r') as f:
            save_dict = json.load(f)
        
        # Reconstruct MotionCaptureData
        data = MotionCaptureData(
            session_id=save_dict['session_id'],
            creation_time=datetime.fromisoformat(save_dict['creation_time']),
            subject_id=save_dict['subject_id'],
            calibration_pose=save_dict['calibration_pose'],
            calibration_duration=save_dict['calibration_duration'],
            calibration_start_time=save_dict['calibration_start_time'],
            is_processed=save_dict['is_processed'],
            notes=save_dict['notes']
        )
        
        if save_dict['processing_timestamp']:
            data.processing_timestamp = datetime.fromisoformat(save_dict['processing_timestamp'])
        
        # Migration: remap legacy 'trunk' → 'back' in imu_data
        if 'trunk' in save_dict['imu_data'] and 'back' not in save_dict['imu_data']:
            print("  Migrating legacy 'trunk' key to 'back' in imu_data")
            save_dict['imu_data']['back'] = save_dict['imu_data'].pop('trunk')
            save_dict['imu_data']['back']['location'] = 'back'

        # Reconstruct IMU data
        for location, imu_dict in save_dict['imu_data'].items():
            sensor_data = IMUSensorData(
                sensor_id=imu_dict['sensor_id'],
                location=imu_dict['location'],
                timestamps=np.array(imu_dict['timestamps']),
                quaternions=np.array(imu_dict['quaternions']),
                accelerations=np.array(imu_dict['accelerations']),
                gyroscopes=np.array(imu_dict['gyroscopes']),
                sampling_frequency=imu_dict['sampling_frequency']
            )
            data.imu_data[location] = sensor_data
        
        # Reconstruct joint angles if available
        if save_dict['joint_angles']:
            ja = save_dict['joint_angles']
            data.joint_angles = JointAngles(
                timestamps=np.array(ja['timestamps']),
                hip_right=np.array(ja['hip_right']),
                hip_left=np.array(ja['hip_left']),
                knee_right=np.array(ja['knee_right']),
                knee_left=np.array(ja['knee_left']),
                ankle_right=np.array(ja['ankle_right']),
                ankle_left=np.array(ja['ankle_left'])
            )
        
        # Reconstruct kinematics if available
        if save_dict['kinematics']:
            kd = save_dict['kinematics']
            # Migration: remap legacy 'trunk_*' keys to 'back_*'
            if 'trunk_angle' in kd and 'back_angle' not in kd:
                print("  Migrating legacy 'trunk_*' kinematics keys to 'back_*'")
                kd['back_angle'] = kd.pop('trunk_angle')
                kd['back_velocity'] = kd.pop('trunk_velocity')
                kd['back_speed'] = kd.pop('trunk_speed')
            data.kinematics = KinematicsData(
                timestamps=np.array(kd['timestamps']),
                back_angle=np.array(kd['back_angle']),
                foot_contact_right=np.array(kd['foot_contact_right']),
                foot_contact_left=np.array(kd['foot_contact_left']),
                back_velocity=np.array(kd['back_velocity']),
                back_speed=np.array(kd['back_speed']),
                stride_times_right=kd['stride_times_right'],
                stride_times_left=kd['stride_times_left']
            )

        marker_dict = save_dict.get('marker_data')
        if marker_dict:
            data.marker_data = MarkerData(
                timestamps=np.array(marker_dict['timestamps']),
                markers={
                    name: np.array(values)
                    for name, values in marker_dict.get('markers', {}).items()
                },
                sampling_frequency=marker_dict.get('sampling_frequency', 100.0)
            )
        
        print(f"Loaded processed data from {filepath}")
        return data
    
    @staticmethod
    def scan_directory(directory: str) -> Dict[str, List[str]]:
        """
        Scan directory for importable and processed files
        
        Args:
            directory: Path to directory to scan
            
        Returns:
            Dictionary with 'raw' and 'processed' file lists
        """
        files = {
            'raw': [],
            'processed': []
        }
        
        if not os.path.exists(directory):
            return files
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                ext = Path(filename).suffix.lower()
                
                if ext in FileHandler.RAW_EXTENSIONS:
                    files['raw'].append(filepath)
                elif ext == FileHandler.PROCESSED_EXTENSION:
                    files['processed'].append(filepath)
        
        return files
    
    @staticmethod
    def is_processed_file(filepath: str) -> bool:
        """Check if file is a processed .mcp file"""
        return Path(filepath).suffix.lower() == FileHandler.PROCESSED_EXTENSION
    
    @staticmethod
    def export_csv(data: MotionCaptureData, filepath: str, export_type: str = 'joint_angles'):
        """
        Export specific data to CSV format
        
        Args:
            data: MotionCaptureData object
            filepath: Output CSV file path
            export_type: Type of data to export ('joint_angles', 'kinematics', 'raw')
        """
        # TODO: Implement CSV export
        print(f"Exporting {export_type} to CSV: {filepath}")
        pass
