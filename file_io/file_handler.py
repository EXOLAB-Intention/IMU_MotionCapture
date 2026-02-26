"""
File I/O handler for IMU motion capture data
Supports import, save, and load operations
"""
import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

from core.imu_data import (
    MotionCaptureData, IMUSensorData, JointAngles, KinematicsData
)
from config.settings import app_settings


class FileHandler:
    """Handles file operations for motion capture data"""
    
    # Supported file extensions
    RAW_EXTENSIONS = ['.csv', '.txt', '.dat', '.h5']  # Raw IMU data formats
    PROCESSED_EXTENSION = '.mcp'  # Motion Capture Processed data
    
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
    def import_h5_trial(filepath: str, h5_path: str) -> MotionCaptureData:
        """
        Import a single trial from an HDF5 file.

        HDF5 structure: Subject > Activity > Level > Trial, with sensor data under
        trial/robot/. Quaternions are stored as quat_w/x/y/z (scalar-first [w,x,y,z]),
        matching this project's convention.

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

        with h5py.File(filepath, 'r') as f:
            if h5_path not in f:
                raise ValueError(f"Trial path '{h5_path}' not found in HDF5 file")

            trial = f[h5_path]

            # Read timestamps (milliseconds → seconds, normalized to start at 0)
            time_ms = trial['common/time'][:]
            timestamps = (time_ms - time_ms[0]) / 1000.0  # ms to seconds
            n_samples = len(timestamps)
            sampling_freq = 100.0  # 100 Hz (10ms intervals)

            # Validate time monotonicity
            dt = np.diff(time_ms)
            if np.any(dt <= 0):
                n_bad = np.sum(dt <= 0)
                print(f"  Warning: {n_bad} non-monotonic time steps detected")
            mean_dt = np.mean(dt)
            if abs(mean_dt - 10.0) > 1.0:
                print(f"  Warning: Mean time step is {mean_dt:.2f}ms (expected 10ms)")

            print(f"  Time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s ({n_samples} samples)")
            print(f"  Sampling frequency: {sampling_freq} Hz")

            # Sensor mapping: H5 path → (location name, sensor_id)
            # hip_imu is skipped (not used in lower-body pipeline)
            sensor_map = [
                ('robot/back_imu',         'back',        0),
                ('robot/thigh_imu/left',   'thigh_left',  1),
                ('robot/thigh_imu/right',  'thigh_right', 2),
                ('robot/shank_imu/left',   'shank_left',  3),
                ('robot/shank_imu/right',  'shank_right', 4),
                ('robot/foot_imu/left',    'foot_left',   5),
                ('robot/foot_imu/right',   'foot_right',  6),
            ]

            for h5_sensor_path, location, sensor_id in sensor_map:
                full_path = h5_path + '/' + h5_sensor_path
                if h5_sensor_path not in trial:
                    print(f"  Warning: Sensor {h5_sensor_path} not found, skipping")
                    continue

                sensor_grp = trial[h5_sensor_path]

                # Check required datasets exist
                required = ['quat_w', 'quat_x', 'quat_y', 'quat_z',
                            'accel_x', 'accel_y', 'accel_z',
                            'gyro_x', 'gyro_y', 'gyro_z']
                missing = [d for d in required if d not in sensor_grp]
                if missing:
                    raise ValueError(
                        f"Sensor {h5_sensor_path} missing datasets: {missing}")

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
            with h5py.File(filepath, 'r') as f:
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
                                if isinstance(level_grp[t], h5py.Group)
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

        with h5py.File(filepath, 'r') as f:
            si_path = f"{subject_id}/sub_info"
            if si_path not in f:
                raise ValueError(f"sub_info not found for {subject_id}")

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
            'kinematics': None
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
