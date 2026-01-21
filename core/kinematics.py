"""
Kinematics processor for computing joint angles, orientations, and motion parameters
This module will contain the core biomechanical calculations

TODO: Implement the following functions:
- Quaternion-based joint angle computation
- Segment orientation calculations
- Hip, knee, ankle 3D joint angles
- Trunk orientation relative to ground
- Foot contact detection algorithms
- Velocity estimation from IMU and gait events
- Stride detection and gait parameters
"""
import numpy as np
from typing import Tuple, List, Optional

from core.imu_data import MotionCaptureData, JointAngles

import pandas as pd
from core.calibration import CalibrationProcessor


class KinematicsProcessor:
    """
    Computes kinematics from calibrated IMU data
    
    This is a placeholder class. Core algorithms to be implemented:
    - Joint angle computation using quaternion algebra
    - Sensor fusion for orientation estimation
    - Ground reference frame estimation
    - Biomechanical constraints
    """
    
    def __init__(self):
        pass
    
    def compute_joint_angles(self, data: MotionCaptureData) -> JointAngles:
        """
        Compute 3D joint angles from IMU quaternions
        
        Logic:
            Joint Angle = Relative orientation of Distal segment w.r.t Proximal segment
            q_rel = q_proximal^{-1} * q_distal

        Args:
            data: Calibrated motion capture data
            
        Returns:
            JointAngles object with hip, knee, ankle angles
        """

        # 1. Set reference timestamp (trunk IMU timestamps)
        if not data.imu_data:
            return None
        # Placeholder implementation
        n_samples = len(data.imu_data['trunk'].timestamps)
        timestamps = data.imu_data['trunk'].timestamps

        #2. Helper function to calculate joint angle between two segments
        def calculate_angle(proximal_name: str, distal_name: str) -> np.ndarray:
            """Calculate joint angle time series between two segments"""
            if proximal_name in data.imu_data and distal_name in data.imu_data:
                q_proximal = data.imu_data[proximal_name].quaternions  # (N,4)
                q_distal = data.imu_data[distal_name].quaternions      # (N,4)

                min_len = min(len(q_proximal), len(q_distal), n_samples)
                q_proximal = q_proximal[:min_len]
                q_distal = q_distal[:min_len]

                # Compute relative orientation
                q_rel = self.compute_relative_orientation(q_proximal, q_distal)  # (min_len,4)
                
                # Convert to Euler angles
                angles = self.quaternion_to_euler(q_rel)    # (N,3) [pitch, roll, yaw]
                
                return angles
            else:
                return np.zeros((n_samples, 3))  # Default zero angles if data missing

        #3. Calculate joint angles
        # Hip: Trunk -> Thigh
        hip_right = calculate_angle('trunk', 'thigh_right')
        hip_left = calculate_angle('trunk', 'thigh_left')
        # Knee: Thigh -> Shank
        knee_right = calculate_angle('thigh_right', 'shank_right')
        knee_left = calculate_angle('thigh_left', 'shank_left')
        # Ankle: Shank -> Foot
        ankle_right = calculate_angle('shank_right', 'foot_right')
        ankle_left = calculate_angle('shank_left', 'foot_left')

        #4. Create dummy data (zeros)
        joint_angles = JointAngles(
            timestamps=timestamps,
            hip_right=hip_right,
            hip_left=hip_left,
            knee_right=knee_right,
            knee_left=knee_left,
            ankle_right=ankle_right,
            ankle_left=ankle_left
        )
        return joint_angles
    
    def compute_trunk_angle(self, data: MotionCaptureData) -> np.ndarray:
        """
        Compute trunk orientation relative to ground
        
        Args:
            data: Motion capture data
            
        Returns:
            (N, 3) array of trunk angles [pitch, roll, yaw] in degrees
        """
        if not data.imu_data:
            return None
        n_samples = len(data.imu_data['trunk'].timestamps)
    
        q_trunk = data.imu_data['trunk'].quaternions  # (N,4)
        trunk_angles = self.quaternion_to_euler(q_trunk)  # (N,3)
        return trunk_angles
    
    def detect_foot_contact(
        self, 
        data: MotionCaptureData,
        accel_threshold: Optional[dict] = {'foot_right': 10.0, 'foot_left': 10.0},
        accel_threshold_weight: float = 0.2,
        gyro_threshold: float = 0.5,
        window_size: int = 10,
        min_contact_duration: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect foot contact events using IMU acceleration and gyroscope
        
        Args:
            data: Motion capture data containing foot IMU sensors
            accel_threshold: dict with 'foot_left' and 'foot_right' keys for acceleration thresholds
            accel_threshold_weight: Weight for threshold range (0-1)
            gyro_threshold: Gyroscope magnitude threshold (rad/s)
            window_size: Moving average window size
            min_contact_duration: Minimum samples for contact event
            
        Returns:
            Tuple of (foot_contact_right, foot_contact_left) boolean arrays
        """
        
        # Extract foot IMU data from MotionCaptureData
        # Expected structure: data.imu_data['foot_right'] and data.imu_data['foot_left']
        foot_right_data = data.imu_data.get('foot_right')
        foot_left_data = data.imu_data.get('foot_left')
        
        if foot_right_data is None or foot_left_data is None:
            raise ValueError("foot_right and foot_left IMU data not found in MotionCaptureData")
        
        # Detect for right foot
        accel_right = foot_right_data.accelerations  # (N, 3)
        gyro_right = foot_right_data.gyroscopes      # (N, 3)
        a_thr_right = accel_threshold.get('foot_right')
        
        foot_contact_right = self._detect_contact_single(
            accel_right, gyro_right,
            a_thr_right, accel_threshold_weight, gyro_threshold,
            window_size, min_contact_duration
        )
        
        # Detect for left foot
        accel_left = foot_left_data.accelerations    # (N, 3)
        gyro_left = foot_left_data.gyroscopes        # (N, 3)
        a_thr_left = accel_threshold.get('foot_left')
        
        foot_contact_left = self._detect_contact_single(
            accel_left, gyro_left,
            a_thr_left, accel_threshold_weight, gyro_threshold,
            window_size, min_contact_duration
        )
        
        return foot_contact_right, foot_contact_left
    
    def _detect_contact_single(
        self,
        accelerations: np.ndarray,
        gyroscopes: np.ndarray,
        accel_threshold: float,
        accel_threshold_weight: float,
        gyro_threshold: float,
        window_size: int,
        min_contact_duration: int
    ) -> np.ndarray:
        """
        Detect foot contact for a single foot
        
        Args:
            accelerations: (N, 3) acceleration array
            gyroscopes: (N, 3) gyroscope array
            accel_threshold: Center acceleration magnitude threshold
            accel_threshold_weight: Weight for threshold range
            gyro_threshold: Gyroscope magnitude threshold
            window_size: Moving average window size
            min_contact_duration: Minimum samples for contact event
            
        Returns:
            (N,) boolean array of contact events
        """
        # Calculate magnitudes
        accel_magnitude = np.linalg.norm(accelerations, axis=1)
        gyro_magnitude = np.linalg.norm(gyroscopes, axis=1)
        
        # Apply moving average smoothing (pandas rolling, center-aligned)
        accel_smooth = pd.Series(accel_magnitude).rolling(window=window_size, center=True, min_periods=1).mean().values
        gyro_smooth = pd.Series(gyro_magnitude).rolling(window=window_size, center=True, min_periods=1).mean().values
        
        # Calculate min/max thresholds
        accel_min_threshold = accel_threshold * (1 - accel_threshold_weight)
        accel_max_threshold = accel_threshold * (1 + accel_threshold_weight)
        
        # Detect contact - acceleration between min/max AND low gyroscope
        accel_contact = (accel_smooth >= accel_min_threshold) & (accel_smooth <= accel_max_threshold)
        gyro_contact = gyro_smooth <= gyro_threshold
        foot_contact = accel_contact & gyro_contact
        
        # Remove noise - fill short False gaps first, then remove short True spikes
        foot_contact = self._remove_noise_falsegapFirst(foot_contact, min_contact_duration)
        
        return foot_contact
    
    def _remove_noise_falsegapFirst(self, contact: np.ndarray, min_duration: int) -> np.ndarray:
        """
        Remove noise by 2-step process: fill short False gaps first, then remove short True regions
        
        Args:
            contact: (N,) boolean array
            min_duration: Minimum duration threshold
            
        Returns:
            (N,) cleaned boolean array
        """
        cleaned = contact.copy()
        
        # Step 1: Fill short False gaps between True regions
        diff = np.diff(cleaned.astype(int))
        true_starts = np.where(diff == 1)[0] + 1   # 0->1 transition
        true_ends = np.where(diff == -1)[0] + 1    # 1->0 transition
        
        if len(cleaned) > 0 and cleaned[0]:
            true_starts = np.r_[0, true_starts]
        if len(cleaned) > 0 and cleaned[-1]:
            true_ends = np.r_[true_ends, len(cleaned)]
        
        for i in range(len(true_ends) - 1):
            gap_start = true_ends[i]
            gap_end = true_starts[i + 1]
            gap_duration = gap_end - gap_start
            
            if gap_duration < min_duration:
                cleaned[gap_start:gap_end] = True
        
        # Step 2: Remove short isolated True regions
        diff = np.diff(cleaned.astype(int))
        true_starts = np.where(diff == 1)[0] + 1
        true_ends = np.where(diff == -1)[0] + 1
        
        if len(cleaned) > 0 and cleaned[0]:
            true_starts = np.r_[0, true_starts]
        if len(cleaned) > 0 and cleaned[-1]:
            true_ends = np.r_[true_ends, len(cleaned)]
        
        for start, end in zip(true_starts, true_ends):
            if end - start < min_duration:
                cleaned[start:end] = False
        
        return cleaned
    
    def compute_velocity(
        self, 
        data: MotionCaptureData,
        foot_contact_right: np.ndarray,
        foot_contact_left: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate trunk velocity using gait events and kinematics
        
        Args:
            data: Motion capture data
            foot_contact_right: Right foot contact events
            foot_contact_left: Left foot contact events
            
        Returns:
            Tuple of (trunk_velocity [N,3], trunk_speed [N])
            
        TODO: Implement velocity estimation algorithm
        """
        n_samples = len(data.imu_data['trunk'].timestamps)
        trunk_velocity = np.zeros((n_samples, 3))
        trunk_speed = np.zeros(n_samples)
        
        return trunk_velocity, trunk_speed
    
    def detect_strides(
        self,
        foot_contact_right: np.ndarray,
        foot_contact_left: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """
        Detect stride times from foot contact events
        
        Args:
            foot_contact_right: Right foot contact boolean array
            foot_contact_left: Left foot contact boolean array
            timestamps: Time array
            
        Returns:
            Tuple of (stride_times_right, stride_times_left)
            
        TODO: Implement stride detection
        """
        stride_times_right = []
        stride_times_left = []
        
        return stride_times_right, stride_times_left