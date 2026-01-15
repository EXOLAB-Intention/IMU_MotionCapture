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
            
        TODO: Implement ground reference frame estimation
        """
        n_samples = len(data.imu_data['trunk'].timestamps)
        return np.zeros((n_samples, 3))
    
    def detect_foot_contact(self, data: MotionCaptureData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect foot contact events using IMU acceleration
        
        Args:
            data: Motion capture data
            
        Returns:
            Tuple of (foot_contact_right, foot_contact_left) boolean arrays
            
        TODO: Implement foot contact detection algorithm
        """
        n_samples = len(data.imu_data['foot_right'].timestamps)
        foot_contact_right = np.zeros(n_samples, dtype=bool)
        foot_contact_left = np.zeros(n_samples, dtype=bool)
        
        return foot_contact_right, foot_contact_left
    
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
    
    # Additional helper functions to be implemented
    
    def quaternion_to_euler(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles (pitch, roll, yaw) in degrees.

        Input quaternion format: [w, x, y, z].
        Returns angles in **order** [pitch, roll, yaw] (degrees), where
        - pitch: rotation about Y axis
        - roll:  rotation about X axis
        - yaw:   rotation about Z axis

        Supports input shapes (4,) or (N,4). Output shape is (3,) or (N,3)
        respectively. Uses clipping of the asin argument for numerical stability.
        """
        q = np.asarray(q, dtype=float)

        def _single(qv: np.ndarray) -> np.ndarray:
            if qv.shape != (4,):
                raise ValueError("Quaternion must have shape (4,)")

            w, x, y, z = qv
            norm = np.linalg.norm(qv)
            if norm == 0.0:
                raise ValueError("Zero-norm quaternion")
            w, x, y, z = w / norm, x / norm, y / norm, z / norm

            # roll (x-axis rotation)
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            # pitch (y-axis rotation)
            sinp = 2.0 * (w * y - z * x)
            sinp = np.clip(sinp, -1.0, 1.0)
            pitch = np.arcsin(sinp)

            # yaw (z-axis rotation)
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            # Return in order [pitch, roll, yaw] (degrees)
            return np.degrees(np.array([pitch, roll, yaw]))

        if q.ndim == 1:
            return _single(q)
        elif q.ndim == 2:
            if q.shape[1] != 4:
                raise ValueError("Quaternion array must have shape (N,4)")

            norms = np.linalg.norm(q, axis=1)
            if np.any(norms == 0.0):
                raise ValueError("Zero-norm quaternion found in input array")
            qn = (q.T / norms).T  # normalize each quaternion

            w = qn[:, 0]
            x = qn[:, 1]
            y = qn[:, 2]
            z = qn[:, 3]

            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2.0 * (w * y - z * x)
            sinp = np.clip(sinp, -1.0, 1.0)
            pitch = np.arcsin(sinp)

            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            angles = np.vstack((pitch, roll, yaw)).T
            return np.degrees(angles)
        else:
            raise ValueError("Input quaternion must have shape (4,) or (N,4)")
    
    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (q1 * q2)

        Quaternions are expected in `[w, x, y, z]` format. Supports inputs of
        shape `(4,)` or `(N, 4)`. If one input is `(N,4)` and the other is `(4,)`,
        the latter will be broadcast to `(N,4)`. Returns the product with the
        same leading dimension as the broadcasted inputs.
        """
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)

        # Scalar (single quaternion) * scalar
        if q1.ndim == 1 and q2.ndim == 1:
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return np.array([w, x, y, z])

        # Promote 1D to 2D for broadcasting convenience
        if q1.ndim == 1:
            q1 = q1[np.newaxis, :]
        if q2.ndim == 1:
            q2 = q2[np.newaxis, :]

        if q1.shape[1] != 4 or q2.shape[1] != 4:
            raise ValueError("Quaternion arrays must have shape (4,) or (N,4)")

        # Broadcast along axis 0 if needed
        if q1.shape[0] != q2.shape[0]:
            if q1.shape[0] == 1:
                q1 = np.repeat(q1, q2.shape[0], axis=0)
            elif q2.shape[0] == 1:
                q2 = np.repeat(q2, q1.shape[0], axis=0)
            else:
                raise ValueError("Quaternion batch sizes must match or one must be of size 1")

        w1 = q1[:, 0]
        x1 = q1[:, 1]
        y1 = q1[:, 2]
        z1 = q1[:, 3]

        w2 = q2[:, 0]
        x2 = q2[:, 1]
        y2 = q2[:, 2]
        z2 = q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.vstack((w, x, y, z)).T

    def quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse.

        For quaternion q = [w, x, y, z], the inverse is q^{-1} = q* / ||q||^2,
        where q* is the conjugate [w, -x, -y, -z]. Supports inputs of shape
        (4,) or (N,4). Returns array with same leading dimension as input.
        """
        q = np.asarray(q, dtype=float)

        if q.ndim == 1: # Single quaternion
            if q.shape[0] != 4:
                raise ValueError("Quaternion must have shape (4,) or (N,4)")
            norm2 = float(np.dot(q, q))
            if norm2 == 0.0:
                raise ValueError("Cannot invert zero-norm quaternion")
            w, x, y, z = q
            conj = np.array([w, -x, -y, -z], dtype=float)
            return conj / norm2

        elif q.ndim == 2:   # Batch of quaternions
            if q.shape[1] != 4:
                raise ValueError("Quaternion array must have shape (N,4)")
            norms2 = np.sum(q * q, axis=1)
            if np.any(norms2 == 0.0):
                raise ValueError("Cannot invert quaternion with zero norm in batch")
            conj = q * np.array([1.0, -1.0, -1.0, -1.0])
            inv = (conj.T / norms2).T
            return inv

        else:
            raise ValueError("Input quaternion must have shape (4,) or (N,4)")
    
    def compute_relative_orientation(
        self, 
        q_proximal: np.ndarray, 
        q_distal: np.ndarray
    ) -> np.ndarray:
        """Compute relative orientation from proximal to distal segment.

        Computes q_rel such that q_distal = q_proximal * q_rel, therefore
        q_rel = q_proximal^{-1} * q_distal.

        Supports inputs of shape (4,) or (N,4) for either argument and
        performs broadcasting when one input is (N,4) and the other is (4,).
        The returned quaternion(s) are normalized.
        """
        qp = np.asarray(q_proximal, dtype=float)
        qd = np.asarray(q_distal, dtype=float)

        # Compute inverse of proximal (handles both (4,) and (N,4))
        qp_inv = self.quaternion_inverse(qp)

        # Multiply: qp_inv * qd (broadcasting handled by quaternion_multiply)
        q_rel = self.quaternion_multiply(qp_inv, qd)

        # Normalize result(s)
        q_rel = np.asarray(q_rel, dtype=float)
        if q_rel.ndim == 1:
            norm = np.linalg.norm(q_rel)
            if norm == 0.0:
                raise ValueError("Relative quaternion has zero norm")
            return q_rel / norm
        elif q_rel.ndim == 2:
            norms = np.linalg.norm(q_rel, axis=1)
            if np.any(norms == 0.0):
                raise ValueError("Relative quaternion has zero norm in batch")
            return (q_rel.T / norms).T
        else:
            raise ValueError("Relative quaternion must have shape (4,) or (N,4)")
