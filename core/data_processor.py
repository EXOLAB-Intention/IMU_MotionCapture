"""
Data processor for motion capture analysis
Orchestrates calibration, kinematics computation, and post-processing
"""
import numpy as np
from typing import Optional
from datetime import datetime

from core.imu_data import MotionCaptureData, JointAngles, KinematicsData
from core.calibration import CalibrationProcessor
from core.kinematics import KinematicsProcessor
from config.settings import app_settings


class DataProcessor:
    """Main data processing pipeline"""
    
    def __init__(self):
        self.calibration_processor = CalibrationProcessor()
        self.kinematics_processor = KinematicsProcessor()
    
    def process_motion_data(
        self, 
        data: MotionCaptureData,
        calibration_start: float,
        calibration_end: float
    ) -> MotionCaptureData:
        """
        Process complete motion capture data
        
        Args:
            data: Raw motion capture data
            calibration_start: Start time of calibration pose (seconds)
            calibration_end: End time of calibration pose (seconds)
            
        Returns:
            Processed MotionCaptureData with joint angles and kinematics
        """
        print(f"Processing motion data for session: {data.session_id}")
        
        # Step 1: Calibration
        print("Step 1: Performing calibration...")
        self.calibration_processor.calibrate(
            data, 
            calibration_start, 
            calibration_end,
            pose_type=app_settings.calibration.pose_type,
            mode=app_settings.mode.mode_type
        )
        
        data.calibration_start_time = calibration_start
        data.calibration_duration = calibration_end - calibration_start
        data.calibration_pose = app_settings.calibration.pose_type
        
        # Step 2: Compute joint angles
        print("Step 2: Computing joint angles...")
        joint_angles = self.kinematics_processor.compute_joint_angles(data)
        data.joint_angles = joint_angles
        
        # Step 3: Compute trunk orientation
        print("Step 3: Computing trunk orientation...")
        trunk_angle = self.kinematics_processor.compute_trunk_angle(data)
        
        # Step 4: Detect foot contacts
        print("Step 4: Detecting foot contacts...")
        foot_contact_right, foot_contact_left = self.kinematics_processor.detect_foot_contact(data)
        
        # Step 5: Compute velocity
        print("Step 5: Computing trunk velocity...")
        trunk_velocity, trunk_speed = self.kinematics_processor.compute_velocity(
            data, 
            foot_contact_right, 
            foot_contact_left
        )
        
        # Step 6: Detect strides
        print("Step 6: Detecting strides...")
        stride_times_right, stride_times_left = self.kinematics_processor.detect_strides(
            foot_contact_right, 
            foot_contact_left,
            data.imu_data['foot_right'].timestamps
        )
        
        # Assemble kinematics data
        data.kinematics = KinematicsData(
            timestamps=joint_angles.timestamps,
            trunk_angle=trunk_angle,
            foot_contact_right=foot_contact_right,
            foot_contact_left=foot_contact_left,
            trunk_velocity=trunk_velocity,
            trunk_speed=trunk_speed,
            stride_times_right=stride_times_right,
            stride_times_left=stride_times_left
        )
        
        # Mark as processed
        data.is_processed = True
        data.processing_timestamp = datetime.now()
        
        print("Processing complete!")
        return data
    
    def process_kinematics_only(self, data: MotionCaptureData) -> MotionCaptureData:
        """
        Process kinematics without performing calibration
        Use this when calibration is already loaded separately
        
        Args:
            data: Motion capture data (should already be calibrated)
            
        Returns:
            Processed MotionCaptureData with joint angles and kinematics
        """
        print(f"Processing kinematics for session: {data.session_id}")
        print("Note: Using existing calibration (no calibration step)")
        
        # Get reference timestamps from first available sensor
        first_sensor = next(iter(data.imu_data.values()))
        timestamps = first_sensor.timestamps
        n_samples = len(timestamps)
        
        # Step 1: Compute joint angles
        print("Step 1: Computing joint angles...")
        joint_angles = self.kinematics_processor.compute_joint_angles(data)
        data.joint_angles = joint_angles
        
        # Step 2: Compute trunk orientation
        print("Step 2: Computing trunk orientation...")
        trunk_angle = self.kinematics_processor.compute_trunk_angle(data)
        
        # Step 3: Detect foot contacts
        print("Step 3: Detecting foot contacts...")
        foot_contact_right, foot_contact_left = self.kinematics_processor.detect_foot_contact(data)
        
        # Step 4: Compute velocity
        print("Step 4: Computing trunk velocity...")
        trunk_velocity, trunk_speed = self.kinematics_processor.compute_velocity(
            data, 
            foot_contact_right, 
            foot_contact_left
        )
        
        # Step 5: Detect strides
        print("Step 5: Detecting strides...")
        stride_times_right, stride_times_left = self.kinematics_processor.detect_strides(
            foot_contact_right, 
            foot_contact_left,
            timestamps
        )
        
        # Assemble kinematics data
        data.kinematics = KinematicsData(
            timestamps=timestamps,
            trunk_angle=trunk_angle,
            foot_contact_right=foot_contact_right,
            foot_contact_left=foot_contact_left,
            trunk_velocity=trunk_velocity,
            trunk_speed=trunk_speed,
            stride_times_right=stride_times_right,
            stride_times_left=stride_times_left
        )
        
        # Mark as processed
        data.is_processed = True
        data.processing_timestamp = datetime.now()
        data.calibration_pose = self.calibration_processor.pose_type
        
        print("Kinematics processing complete!")
        return data
    
    def batch_process(
        self,
        data_list: list,
        calibration_params: dict
    ) -> list:
        """
        Process multiple trials with same settings
        
        Args:
            data_list: List of MotionCaptureData objects
            calibration_params: Dictionary with calibration parameters
            
        Returns:
            List of processed MotionCaptureData objects
        """
        processed_list = []
        
        for i, data in enumerate(data_list):
            print(f"\nProcessing trial {i+1}/{len(data_list)}: {data.session_id}")
            
            try:
                processed_data = self.process_motion_data(
                    data,
                    calibration_params.get('start_time', 0.0),
                    calibration_params.get('end_time', 2.0)
                )
                processed_list.append(processed_data)
                
            except Exception as e:
                print(f"Error processing {data.session_id}: {e}")
                continue
        
        print(f"\nBatch processing complete: {len(processed_list)}/{len(data_list)} successful")
        return processed_list
    
    def filter_data(self, data: np.ndarray, cutoff_freq: float, sampling_freq: float) -> np.ndarray:
        """
        Apply low-pass filter to data
        
        Args:
            data: Input data array
            cutoff_freq: Cutoff frequency (Hz)
            sampling_freq: Sampling frequency (Hz)
            
        Returns:
            Filtered data
        """
        # TODO: Implement filtering (e.g., Butterworth)
        return data
    
    def resample_data(self, timestamps: np.ndarray, data: np.ndarray, target_freq: float) -> tuple:
        """
        Resample data to target frequency
        
        Args:
            timestamps: Original timestamps
            data: Original data
            target_freq: Target sampling frequency (Hz)
            
        Returns:
            Tuple of (new_timestamps, resampled_data)
        """
        # TODO: Implement resampling
        return timestamps, data
