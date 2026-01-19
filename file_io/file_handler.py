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


class FileHandler:
    """Handles file operations for motion capture data"""
    
    # Supported file extensions
    RAW_EXTENSIONS = ['.csv', '.txt', '.dat']  # Raw IMU data formats
    PROCESSED_EXTENSION = '.mcp'  # Motion Capture Processed data
    
    @staticmethod
    def import_raw_data(filepath: str) -> MotionCaptureData:
        """
        Import raw IMU data from file
        
        Args:
            filepath: Path to raw data file
            
        Returns:
            MotionCaptureData object with IMU data
            
        Note:
            File format specification to be determined
            Expected format: CSV or similar with columns for each sensor
        """
        # TODO: Implement based on actual data format
        # Placeholder implementation
        
        file_ext = Path(filepath).suffix.lower()
        
        if file_ext == '.csv':
            return FileHandler._import_csv(filepath)
        elif file_ext == '.txt':
            return FileHandler._import_txt(filepath)
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
        
        # Create time array (assuming constant sampling rate)
        # If there's a LoopCnt column, use it to generate time
        if 'LoopCnt' in df.columns:
            loop_cnt = df['LoopCnt'].values
            # Estimate sampling frequency from loop counter
            # Assuming 100Hz sampling rate (can be adjusted)
            sampling_freq = 100.0
            timestamps = loop_cnt / sampling_freq
        else:
            # Generate timestamps assuming 100Hz
            sampling_freq = 100.0
            timestamps = np.arange(n_samples) / sampling_freq
        
        # Sensor mapping
        sensor_configs = [
            {
                'location': 'trunk',
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
            
            # Check for valid data (non-zero)
            if np.all(quaternions == 0):
                print(f"Warning: Skipping {config['location']} - all zero data")
                continue
            
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
                'trunk_angle': data.kinematics.trunk_angle.tolist(),
                'foot_contact_right': data.kinematics.foot_contact_right.tolist(),
                'foot_contact_left': data.kinematics.foot_contact_left.tolist(),
                'trunk_velocity': data.kinematics.trunk_velocity.tolist(),
                'trunk_speed': data.kinematics.trunk_speed.tolist(),
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
            data.kinematics = KinematicsData(
                timestamps=np.array(kd['timestamps']),
                trunk_angle=np.array(kd['trunk_angle']),
                foot_contact_right=np.array(kd['foot_contact_right']),
                foot_contact_left=np.array(kd['foot_contact_left']),
                trunk_velocity=np.array(kd['trunk_velocity']),
                trunk_speed=np.array(kd['trunk_speed']),
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
