"""
Kinematics processor for computing joint angles, orientations, and motion parameters

This module implements the quaternion operation pipeline based on MATLAB reference:
1. Quaternion normalization: quatnormalize(q_raw)
2. Desired orientation at calibration pose: plotQuatNearestAxes()
3. Correction quaternion: qCorr = qD * q_raw(T_pose)^{-1}
4. Apply correction: q_calibrated = qCorr * q_raw (LEFT multiplication)
5. Rotation matrix with Y180: R = R_y180 @ quat2rotm(q)
6. Relative quaternion (joint angle): q_rel = conj(q_proximal) * q_distal
"""
import numpy as np
from typing import Tuple, List, Optional, Dict

from core.imu_data import MotionCaptureData, JointAngles
from config.settings import app_settings

import pandas as pd


class KinematicsProcessor:
    """
    Computes kinematics from calibrated IMU data
    
    MATLAB Pipeline Implementation:
    1. Quaternion operations use LEFT multiplication convention
    2. Calibration: qCorr = qD * q_raw^{-1}, then q_cal = qCorr * q_raw
    3. Joint angles: q_rel = conj(q_proximal) * q_distal
    4. Rotation matrices: R = R_y180 @ quat2rotm(q)
    """
    
    # Y-axis 180 degree rotation matrix (MATLAB: R_y180 = [-1 0 0; 0 1 0; 0 0 -1])
    R_Y180 = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ], dtype=float)
    
    def __init__(self):
        pass

    @property
    def current_mode(self) -> str:
        """Return the current capture mode"""
        return app_settings.mode.mode_type
    
    def compute_joint_angles(self, data: MotionCaptureData) -> JointAngles:
        """
        Compute 3D joint angles from calibrated IMU quaternions.
        
        MATLAB Pipeline:
            q_rel = quatmultiply(quatconj(q_proximal), q_distal)
            euler = quat2eul(q_rel)  % ZYX sequence by default
        
        Joint definitions (lower body):
            - Hip: trunk (proximal) → thigh (distal)
            - Knee: thigh (proximal) → shank (distal)  
            - Ankle: shank (proximal) → foot (distal)
        Joint definitions (upper body):
            - Spine: pelvis (proximal) → chest (distal)
            - Neck: chest (proximal) → head (distal)
            - Shoulder: chest (proximal) → upperarm (distal)
            - Elbow: upperarm (proximal) → lowerarm (distal)
        
        Args:
            data: Calibrated motion capture data
            
        Returns:
            JointAngles object with hip, knee, ankle angles in degrees
        """

        # 1. Check data availability
        if not data.imu_data:
            return None
        
        # 2. Set reference timestamp (back IMU timestamps)
        ref_sensor_name = 'back' if self.current_mode == 'Lower-body' else 'pelvis'
        n_samples = len(data.imu_data[ref_sensor_name].timestamps)
        timestamps = data.imu_data[ref_sensor_name].timestamps

        # 3. Helper function to calculate joint angle between two segments
        def calculate_joint_angle(proximal_name: str, distal_name: str) -> np.ndarray:
            """
            Calculate joint angle time series following MATLAB convention.
            
            MATLAB: q_rel = quatmultiply(quatconj(q_proximal), q_distal)
            """
            if proximal_name in data.imu_data and distal_name in data.imu_data:
                q_proximal = data.imu_data[proximal_name].quaternions  # (N,4) [w,x,y,z]
                q_distal = data.imu_data[distal_name].quaternions      # (N,4) [w,x,y,z]

                # Ensure same length
                min_len = min(len(q_proximal), len(q_distal), n_samples)
                q_proximal = q_proximal[:min_len]
                q_distal = q_distal[:min_len]

                # Normalize quaternions
                q_proximal_norm = self.quaternion_normalize(q_proximal)
                q_distal_norm = self.quaternion_normalize(q_distal)

                # Compute relative orientation: q_rel = conj(q_proximal) * q_distal
                # This is the MATLAB convention (LEFT multiplication with conjugate)
                q_rel = self.compute_relative_quaternion(q_proximal_norm, q_distal_norm)
                
                # Convert to Euler angles (ZYX sequence like MATLAB quat2eul default)
                angles = self.quaternion_to_euler_zyx(q_rel)  # (N,3) in degrees
                
                # Pad to n_samples if needed
                if len(angles) < n_samples:
                    padding = np.zeros((n_samples - len(angles), 3))
                    angles = np.vstack([angles, padding])
                
                return angles
            else:
                # Return zero angles if data missing
                return np.zeros((n_samples, 3))

        # 4. Calculate joint angles for all joints
        if self.current_mode == 'Upper-body':
            # Spine: Pelvis (proximal) → Chest (distal)
            spine = calculate_joint_angle('pelvis', 'chest')
            
            # Neck: Chest (proximal) → Head (distal)
            neck = calculate_joint_angle('chest', 'head')
            
            # Shoulder: Chest (proximal) → Upperarm (distal)
            shoulder_right = calculate_joint_angle('chest', 'upperarm_right')
            shoulder_left = calculate_joint_angle('chest', 'upperarm_left')
            
            # Elbow: Upperarm (proximal) → Lowerarm (distal)
            elbow_right = calculate_joint_angle('upperarm_right', 'lowerarm_right')
            elbow_left = calculate_joint_angle('upperarm_left', 'lowerarm_left')

            # 5. Create JointAngles object
            joint_angles = JointAngles(
                timestamps=timestamps,
                spine=spine,
                neck=neck,
                shoulder_right=shoulder_right,
                shoulder_left=shoulder_left,
                elbow_right=elbow_right,
                elbow_left=elbow_left
            )
            return joint_angles
        
        # Lower-body joints
        else:
            # Hip: Back (proximal) → Thigh (distal)
            hip_right = calculate_joint_angle('back', 'thigh_right')
            hip_left = calculate_joint_angle('back', 'thigh_left')
            
            # Knee: Thigh (proximal) → Shank (distal)
            knee_right = calculate_joint_angle('thigh_right', 'shank_right')
            knee_left = calculate_joint_angle('thigh_left', 'shank_left')
            
            # Ankle: Shank (proximal) → Foot (distal)
            ankle_right = calculate_joint_angle('shank_right', 'foot_right')
            ankle_left = calculate_joint_angle('shank_left', 'foot_left')

            # 5. Create JointAngles object
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
    
    def compute_back_angle(self, data: MotionCaptureData) -> np.ndarray:
        """
        Compute back orientation relative to ground

        Args:
            data: Motion capture data

        Returns:
            (N, 3) array of back angles [pitch, roll, yaw] in degrees
        """
        if not data.imu_data:
            return None

        ref_sensor_name = 'back' if self.current_mode == 'Lower-body' else 'pelvis'
        n_samples = len(data.imu_data[ref_sensor_name].timestamps)

        q_back = data.imu_data[ref_sensor_name].quaternions  # (N,4)
        back_angles = self.quaternion_to_euler(q_back)  # (N,3)
        return back_angles
    
    def detect_foot_contact(
        self, 
        data: MotionCaptureData,
        accel_threshold: Optional[dict] = {'foot_right': 10.0, 'foot_left': 10.0},
        accel_threshold_weight: float = 0.2,
        gyro_threshold: float = 0.5,
        window_size: int = 10,
        min_contact_duration: int = 20
    ) -> Tuple[int, int, np.ndarray, np.ndarray]:
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
            Tuple of (gait_start_frame, gait_end_frame, foot_contact_right, foot_contact_left)
            where:
                - gait_start_frame (int): Frame where first foot leaves double support
                - gait_end_frame (int): Frame where both feet return to contact at end
                - foot_contact_right: Modified boolean array (no simultaneous contact/non-contact)
                - foot_contact_left: Modified boolean array (no simultaneous contact/non-contact)
        """
        if self.current_mode == 'Upper-body':
            return (None, None, None, None)  # No foot contact detection in upper-body mode
        
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
        
        # Post-processing: find gait frames and enforce constraints
        gait_start_frame, gait_end_frame, foot_contact_right, foot_contact_left = \
            self._process_gait_cycle(foot_contact_right, foot_contact_left)
        
        return (gait_start_frame, gait_end_frame, foot_contact_right, foot_contact_left)
    
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
    
    def _process_gait_cycle(
        self,
        foot_contact_right: np.ndarray,
        foot_contact_left: np.ndarray
    ) -> Tuple[int, int, np.ndarray, np.ndarray]:
        """
        Process foot contact arrays to extract gait cycle and enforce constraints
        
        Rules:
            1. Find gait_start_frame: first frame where one foot leaves initial double support
            2. Find gait_end_frame: first frame where both feet return to contact at the end
            3. Between gait_start and gait_end:
               - Remove simultaneous contact: keep only the foot that contacted earlier
               - Remove simultaneous non-contact: keep the foot that was last in contact
        
        Args:
            foot_contact_right: Right foot contact boolean array
            foot_contact_left: Left foot contact boolean array
            
        Returns:
            Tuple of (gait_start_frame, gait_end_frame, modified_right, modified_left)
        """
        N = len(foot_contact_right)
        right = foot_contact_right.copy().astype(bool)
        left = foot_contact_left.copy().astype(bool)
        
        # Step 1: Find gait_start_frame
        # Find first frame where both are True
        both_contact = right & left
        first_both_idx = np.where(both_contact)[0]
        
        if len(first_both_idx) == 0:
            # No double support at start, use first frame
            gait_start_frame = 0
        else:
            # Find first frame after initial double support where one foot leaves
            start_region = first_both_idx[0]
            # Look for first break in double support after start_region
            gait_start_frame = None
            for i in range(start_region, N):
                if not (right[i] and left[i]):
                    gait_start_frame = i
                    break
            if gait_start_frame is None:
                gait_start_frame = N - 1
        
        # Step 2: Find gait_end_frame
        # Find last section where both are True continuously
        gait_end_frame = None
        for i in range(N - 1, -1, -1):
            if right[i] and left[i]:
                # Check if this is the start of a final double support region
                # by checking if before this point there was single support
                if i == 0 or not (right[i-1] and left[i-1]):
                    gait_end_frame = i
                    break
        
        if gait_end_frame is None:
            gait_end_frame = N - 1
        
        # Step 3: Apply post-processing rules between gait_start_frame and gait_end_frame (exclusive)
        if gait_start_frame < gait_end_frame:
            right, left = self._enforce_gait_constraints(right, left, gait_start_frame, gait_end_frame - 1)
        
        # Step 4: Ensure both feet are True from gait_end_frame to the end
        right[gait_end_frame:] = True
        left[gait_end_frame:] = True
        
        return gait_start_frame, gait_end_frame, right, left
    
    def _enforce_gait_constraints(
        self,
        right: np.ndarray,
        left: np.ndarray,
        start_frame: int,
        end_frame: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enforce gait constraints between start_frame and end_frame
        
        Rule 1: Remove simultaneous contact
            If both feet contact at same frame, keep only the foot that contacted earlier
        Rule 2: Remove simultaneous non-contact
            If both feet not in contact at same frame, keep the foot that was last in contact
        
        Args:
            right: Right foot contact array
            left: Left foot contact array
            start_frame: Gait cycle start
            end_frame: Gait cycle end
            
        Returns:
            Modified (right, left) arrays
        """
        right = right.copy()
        left = left.copy()
        
        # Process each frame in the gait cycle
        for i in range(start_frame, end_frame + 1):
            both_contact = right[i] and left[i]
            both_notcontact = (not right[i]) and (not left[i])
            
            if both_contact:
                # Rule 1: Remove simultaneous contact
                # Keep the foot that contacted earlier (look back to find which one contacted first)
                if i == start_frame:
                    # At start, keep both or arbitrarily choose one (keep right)
                    pass
                else:
                    # Find which foot broke contact first before this frame
                    # by looking backward for the most recent contact change
                    right_contact_frame = -1
                    left_contact_frame = -1
                    
                    for j in range(i - 1, start_frame - 1, -1):
                        if not right[j] and right_contact_frame == -1:
                            right_contact_frame = j
                        if not left[j] and left_contact_frame == -1:
                            left_contact_frame = j
                        if right_contact_frame != -1 and left_contact_frame != -1:
                            break
                    
                    # The foot that broke contact later should be kept contacting
                    # (i.e., the one with larger frame index of non-contact)
                    if right_contact_frame > left_contact_frame:
                        # Right broke contact more recently, so it re-established first
                        # Keep right, remove left
                        left[i] = False
                    elif left_contact_frame > right_contact_frame:
                        # Left broke contact more recently, so it re-established first
                        # Keep left, remove right
                        right[i] = False
                    else:
                        # Both broke contact at same frame - shouldn't happen, keep right
                        left[i] = False
            
            elif both_notcontact:
                # Rule 2: Remove simultaneous non-contact
                # Keep the foot that was in contact most recently
                right_last_contact = -1
                left_last_contact = -1
                
                for j in range(i - 1, -1, -1):
                    if right[j] and right_last_contact == -1:
                        right_last_contact = j
                    if left[j] and left_last_contact == -1:
                        left_last_contact = j
                    if right_last_contact != -1 and left_last_contact != -1:
                        break
                
                # Set the foot that was last in contact to be in contact
                if right_last_contact > left_last_contact:
                    right[i] = True
                elif left_last_contact > right_last_contact:
                    left[i] = True
                else:
                    # Both never contacted or contacted at same frame - keep right
                    right[i] = True
        
        return right, left
    
    def compute_velocity(
        self, 
        data: MotionCaptureData,
        foot_contact_right: np.ndarray,
        foot_contact_left: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate back velocity using gait events and kinematics

        Args:
            data: Motion capture data
            foot_contact_right: Right foot contact events
            foot_contact_left: Left foot contact events

        Returns:
            Tuple of (back_velocity [N,3], back_speed [N])
            
        TODO: Implement velocity estimation algorithm
        """
        
        ref_sensor_name = 'back' if self.current_mode == 'Lower-body' else 'pelvis'
        n_samples = len(data.imu_data[ref_sensor_name].timestamps)
        back_velocity = np.zeros((n_samples, 3))
        back_speed = np.zeros(n_samples)

        return back_velocity, back_speed
    
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
    
    # Additional helper functions to be implemented
    
    @staticmethod
    def quaternion_normalize(q: np.ndarray) -> np.ndarray:
        """
        Normalize quaternion(s) to unit length.
        
        MATLAB equivalent: quatnormalize(q)
        
        Args:
            q: Quaternion(s) [w,x,y,z], shape (4,) or (N,4)
            
        Returns:
            Normalized quaternion(s) with same shape as input
        """
        q = np.asarray(q, dtype=float)
        
        if q.ndim == 1:
            norm = np.linalg.norm(q)
            if norm == 0.0:
                raise ValueError("Cannot normalize zero-norm quaternion")
            return q / norm
        elif q.ndim == 2:
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            if np.any(norms == 0.0):
                raise ValueError("Cannot normalize zero-norm quaternion in batch")
            return q / norms
        else:
            raise ValueError("Input must have shape (4,) or (N,4)")
    
    @staticmethod
    def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
        """
        Compute quaternion conjugate.
        
        MATLAB equivalent: quatconj(q)
        For q = [w, x, y, z], conjugate is [w, -x, -y, -z]
        
        Args:
            q: Quaternion(s) [w,x,y,z], shape (4,) or (N,4)
            
        Returns:
            Conjugate quaternion(s)
        """
        q = np.asarray(q, dtype=float)
        
        if q.ndim == 1:
            return np.array([q[0], -q[1], -q[2], -q[3]])
        elif q.ndim == 2:
            return q * np.array([1.0, -1.0, -1.0, -1.0])
        else:
            raise ValueError("Input must have shape (4,) or (N,4)")
    
    @staticmethod
    def quaternion_to_euler_zyx(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to Euler angles using ZYX sequence.
        
        MATLAB equivalent: quat2eul(q) which defaults to ZYX sequence
        Returns [yaw, pitch, roll] in radians, converted to degrees.
        
        This matches MATLAB's default quat2eul output order.
        
        Args:
            q: Quaternion(s) [w,x,y,z], shape (4,) or (N,4)
            
        Returns:
            Euler angles [Z, Y, X] in degrees, shape (3,) or (N,3)
        """
        q = np.asarray(q, dtype=float)
        
        def _single(qv: np.ndarray) -> np.ndarray:
            qv = qv / np.linalg.norm(qv)
            w, x, y, z = qv
            
            # ZYX Euler angles (yaw, pitch, roll)
            # This matches MATLAB quat2eul default
            
            # Roll (X-axis rotation)
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (Y-axis rotation)
            sinp = 2.0 * (w * y - z * x)
            sinp = np.clip(sinp, -1.0, 1.0)
            pitch = np.arcsin(sinp)
            
            # Yaw (Z-axis rotation)
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            # Return in ZYX order like MATLAB quat2eul
            return np.degrees(np.array([yaw, pitch, roll]))
        
        if q.ndim == 1:
            return _single(q)
        elif q.ndim == 2:
            qn = q / np.linalg.norm(q, axis=1, keepdims=True)
            w, x, y, z = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
            
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
            pitch = np.arcsin(sinp)
            
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            return np.degrees(np.vstack((yaw, pitch, roll)).T)
        else:
            raise ValueError("Input must have shape (4,) or (N,4)")
    
    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles (XYZ sequence).
        
        Alternative to ZYX - uses XYZ Cardan sequence.
        
        Args:
            q: Quaternion(s) [w,x,y,z], shape (4,) or (N,4)
            
        Returns:
            Euler angles [X, Y, Z] in degrees, shape (3,) or (N,3)
        """
        q = np.asarray(q, dtype=float)

        def _single(qv: np.ndarray) -> np.ndarray:
            qv = qv / np.linalg.norm(qv)
            w, x, y, z = qv

            # XYZ Cardan sequence
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            angle_x = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
            angle_y = np.arcsin(sinp)

            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            angle_z = np.arctan2(siny_cosp, cosy_cosp)

            return np.degrees(np.array([angle_x, angle_y, angle_z]))

        if q.ndim == 1:
            return _single(q)
        elif q.ndim == 2:
            qn = q / np.linalg.norm(q, axis=1, keepdims=True)
            w, x, y, z = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]

            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            angle_x = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
            angle_y = np.arcsin(sinp)

            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            angle_z = np.arctan2(siny_cosp, cosy_cosp)

            return np.degrees(np.vstack((angle_x, angle_y, angle_z)).T)
        else:
            raise ValueError("Input must have shape (4,) or (N,4)")
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        
        MATLAB equivalent: quat2rotm(quaternion(q))
        
        Args:
            q: Quaternion [w,x,y,z], shape (4,) or (N,4)
            
        Returns:
            Rotation matrix (3,3) or (3,3,N) for batch input
        """
        q = np.asarray(q, dtype=float)
        
        def _single(qv: np.ndarray) -> np.ndarray:
            qv = qv / np.linalg.norm(qv)
            w, x, y, z = qv
            
            R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
                [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
                [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ])
            return R
        
        if q.ndim == 1:
            return _single(q)
        elif q.ndim == 2:
            N = q.shape[0]
            R = np.zeros((3, 3, N))
            for i in range(N):
                R[:, :, i] = _single(q[i])
            return R
        else:
            raise ValueError("Input must have shape (4,) or (N,4)")
    
    @staticmethod
    def quaternion_to_rotation_matrix_y180(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix with Y180 transformation.
        
        MATLAB equivalent: R = pagemtimes(R_y180, quat2rotm(quaternion(q)))
        where R_y180 = [-1 0 0; 0 1 0; 0 0 -1]
        
        Args:
            q: Quaternion [w,x,y,z], shape (4,) or (N,4)
            
        Returns:
            Transformed rotation matrix (3,3) or (3,3,N) for batch input
        """
        R_y180 = KinematicsProcessor.R_Y180
        R = KinematicsProcessor.quaternion_to_rotation_matrix(q)
        
        if R.ndim == 2:
            return R_y180 @ R
        else:  # (3, 3, N)
            N = R.shape[2]
            R_out = np.zeros_like(R)
            for i in range(N):
                R_out[:, :, i] = R_y180 @ R[:, :, i]
            return R_out
    
    @staticmethod
    def compute_relative_quaternion(q_proximal: np.ndarray, q_distal: np.ndarray) -> np.ndarray:
        """
        Compute relative quaternion (joint angle) between two segments.
        
        MATLAB equivalent: q_rel = quatmultiply(quatconj(q_proximal), q_distal)
        
        This computes the rotation from proximal to distal frame,
        representing the joint angle.
        
        Args:
            q_proximal: Proximal segment quaternion(s) [w,x,y,z]
            q_distal: Distal segment quaternion(s) [w,x,y,z]
            
        Returns:
            Relative quaternion(s) representing joint angle
        """
        q_prox_conj = KinematicsProcessor.quaternion_conjugate(q_proximal)
        q_rel = KinematicsProcessor.quaternion_multiply(q_prox_conj, q_distal)
        return KinematicsProcessor.quaternion_normalize(q_rel)
    
    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions (q1 * q2) using Hamilton convention.
        
        MATLAB equivalent: quatmultiply(q1, q2)
        
        Quaternions are in [w, x, y, z] format.
        Supports (4,), (N,4) inputs with broadcasting.
        
        Args:
            q1: First quaternion(s)
            q2: Second quaternion(s)
            
        Returns:
            Product quaternion(s)
        """
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)

        # Single quaternion * single quaternion
        if q1.ndim == 1 and q2.ndim == 1:
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2

            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2

            return np.array([w, x, y, z])

        # Promote 1D to 2D for broadcasting
        if q1.ndim == 1:
            q1 = q1[np.newaxis, :]
        if q2.ndim == 1:
            q2 = q2[np.newaxis, :]

        # Broadcast along axis 0 if needed
        if q1.shape[0] != q2.shape[0]:
            if q1.shape[0] == 1:
                q1 = np.repeat(q1, q2.shape[0], axis=0)
            elif q2.shape[0] == 1:
                q2 = np.repeat(q2, q1.shape[0], axis=0)
            else:
                raise ValueError("Batch sizes must match or one must be 1")

        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.vstack((w, x, y, z)).T

    @staticmethod
    def quaternion_inverse(q: np.ndarray) -> np.ndarray:
        """
        Compute quaternion inverse.
        
        MATLAB equivalent: quatinv(q)
        For unit quaternion, inverse equals conjugate.
        For non-unit quaternion: q^{-1} = conj(q) / ||q||^2
        
        Args:
            q: Quaternion(s) [w,x,y,z], shape (4,) or (N,4)
            
        Returns:
            Inverse quaternion(s)
        """
        q = np.asarray(q, dtype=float)

        if q.ndim == 1:
            norm2 = np.dot(q, q)
            if norm2 == 0.0:
                raise ValueError("Cannot invert zero-norm quaternion")
            conj = np.array([q[0], -q[1], -q[2], -q[3]])
            return conj / norm2
        elif q.ndim == 2:
            norms2 = np.sum(q * q, axis=1, keepdims=True)
            if np.any(norms2 == 0.0):
                raise ValueError("Cannot invert zero-norm quaternion")
            conj = q * np.array([1.0, -1.0, -1.0, -1.0])
            return conj / norms2
        else:
            raise ValueError("Input must have shape (4,) or (N,4)")
    
    @staticmethod
    def find_nearest_axis_quaternion(q: np.ndarray) -> np.ndarray:
        """
        Find the nearest axis-aligned quaternion to the input quaternion.
        
        MATLAB equivalent: plotQuatNearestAxes() function
        
        This function finds the quaternion representing a rotation to the 
        nearest principal axis alignment. Used during calibration to establish
        the desired orientation (qD) for each segment.
        
        The algorithm:
        1. Convert quaternion to rotation matrix
        2. For each axis of the rotation matrix, find the nearest principal axis
        3. Construct a rotation matrix from these principal axes
        4. Convert back to quaternion
        
        Args:
            q: Input quaternion [w,x,y,z], shape (4,)
            
        Returns:
            Axis-aligned quaternion [w,x,y,z]
        """
        q = np.asarray(q, dtype=float)
        q = q / np.linalg.norm(q)
        
        # Convert to rotation matrix
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # Principal axes
        principal_axes = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=float)
        
        # Find nearest principal axis for each column of R
        R_aligned = np.zeros((3, 3))
        used_axes = set()
        
        for col in range(3):
            axis = R[:, col]
            best_dot = -np.inf
            best_idx = 0
            
            for idx, pa in enumerate(principal_axes):
                # Skip if this principal axis already used (or its negative)
                base_idx = idx // 2
                if base_idx in used_axes:
                    continue
                    
                dot = np.dot(axis, pa)
                if dot > best_dot:
                    best_dot = dot
                    best_idx = idx
            
            R_aligned[:, col] = principal_axes[best_idx]
            used_axes.add(best_idx // 2)
        
        # Ensure right-handed coordinate system
        if np.linalg.det(R_aligned) < 0:
            R_aligned[:, 2] = -R_aligned[:, 2]
        
        # Convert rotation matrix to quaternion
        return KinematicsProcessor.rotation_matrix_to_quaternion(R_aligned)
    
    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion [w, x, y, z].
        
        MATLAB equivalent: rotm2quat()
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion [w, x, y, z]
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        quat = np.array([w, x, y, z])
        return quat / np.linalg.norm(quat)
    
    @staticmethod  
    def compute_correction_quaternion(q_desired: np.ndarray, q_raw: np.ndarray) -> np.ndarray:
        """
        Compute correction quaternion for calibration.
        
        MATLAB equivalent: qCorr = quatmultiply(qD, quatinv(q_raw))
        
        This correction is applied via LEFT multiplication:
            q_calibrated = qCorr * q_raw
        
        Args:
            q_desired: Desired orientation (from find_nearest_axis_quaternion)
            q_raw: Raw IMU quaternion at calibration pose
            
        Returns:
            Correction quaternion
        """
        q_raw_inv = KinematicsProcessor.quaternion_inverse(q_raw)
        q_corr = KinematicsProcessor.quaternion_multiply(q_desired, q_raw_inv)
        return KinematicsProcessor.quaternion_normalize(q_corr)
    
    @staticmethod
    def apply_correction(q_corr: np.ndarray, q_raw: np.ndarray) -> np.ndarray:
        """
        Apply correction quaternion to raw data.
        
        MATLAB equivalent: q_calibrated = quatmultiply(qCorr, q_raw)
        Uses LEFT multiplication.
        
        Args:
            q_corr: Correction quaternion (single)
            q_raw: Raw quaternion(s) to correct
            
        Returns:
            Corrected quaternion(s)
        """
        q_calibrated = KinematicsProcessor.quaternion_multiply(q_corr, q_raw)
        return KinematicsProcessor.quaternion_normalize(q_calibrated)
    
    # Legacy functions for backward compatibility
    @staticmethod
    def compute_relative_orientation(
        q_proximal: np.ndarray, 
        q_distal: np.ndarray
    ) -> np.ndarray:
        """Legacy: Use compute_relative_quaternion instead."""
        return KinematicsProcessor.compute_relative_quaternion(q_proximal, q_distal)
    
    @staticmethod
    def compute_relative_orientation_isb(
        q_proximal: np.ndarray, 
        q_distal: np.ndarray
    ) -> np.ndarray:
        """Legacy: Use compute_relative_quaternion instead."""
        return KinematicsProcessor.compute_relative_quaternion(q_proximal, q_distal)
