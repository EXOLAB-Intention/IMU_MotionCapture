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
        """Import CSV format (placeholder)"""
        # TODO: Implement CSV parsing
        # Expected columns: timestamp, sensor_id, qw, qx, qy, qz, ax, ay, az, gx, gy, gz
        
        session_id = Path(filepath).stem
        data = MotionCaptureData(
            session_id=session_id,
            creation_time=datetime.now()
        )
        
        # Placeholder: would parse CSV and populate IMU data
        print(f"Importing CSV from {filepath}")
        
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
