"""
Calibration processor for IMU motion capture
Handles initial pose calibration (N-pose) to establish reference frames

MATLAB Pipeline Implementation:
1. At calibration pose, get raw quaternion: q_raw(T_pose)
2. Find desired axis-aligned orientation: qD = plotQuatNearestAxes(q_raw)
3. Compute correction: qCorr = qD * q_raw^{-1} (quatmultiply(qD, quatinv(q_raw)))
4. Apply to all frames: q_calibrated = qCorr * q_raw (LEFT multiplication)
"""
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from core.imu_data import MotionCaptureData, IMUSensorData
from core.kinematics import KinematicsProcessor


class CalibrationProcessor:
    """
    Processes calibration poses to establish reference orientations
    
    Pipeline:
        1. q_desired = ideal IMU orientation for each segment (from attachment spec)
        2. q_calib = average of measured quaternions during N-pose
        3. q_offset = conj(q_calib) * q_desired (RIGHT multiplication)
        4. q_segment = q_measured * q_offset (RIGHT multiplication)
    """
    
    CALIBRATION_EXTENSION = '.cal'
    
    # =======================================================================
    # Desired IMU orientations for each segment at N-pose (차렷자세)
    # Ground coordinate: X=forward, Y=left, Z=up
    # =======================================================================
    @staticmethod
    def _get_desired_quaternions() -> Dict[str, np.ndarray]:
        """
        Get desired (ideal) IMU orientations as quaternions.
        
        IMU attachment directions at N-pose (UNIFIED):
        All segments use the same coordinate system:
        - trunk:       x-up, y-left, z-backward
        - thigh/shank: x-up, y-left, z-backward
        - foot:        x-up, y-left, z-backward
        
        Rotation matrix R: R @ v_local = v_global
        Each column of R is where the sensor's local axis points in global frame.
        """
        def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
            """Convert rotation matrix to quaternion [w, x, y, z]."""
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
            q = np.array([w, x, y, z])
            return q / np.linalg.norm(q)
        
        # trunk: x-up, y-left, z-backward (same as thigh/shank)
        # sensor x → global [0, 0, 1] (up)
        # sensor y → global [0, 1, 0] (left)
        # sensor z → global [-1, 0, 0] (backward)
        R_trunk = np.array([
            [0, 0, 1],    # column 0: sensor x in global
            [0, 1, 0],    # column 1: sensor y in global
            [-1, 0, 0]    # column 2: sensor z in global
        ]).T  # transpose to get columns right
        
        # thigh/shank: x-up, y-left, z-backward
        # sensor x → global [0, 0, 1] (up)
        # sensor y → global [0, 1, 0] (left)
        # sensor z → global [-1, 0, 0] (backward)
        R_thigh = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ]).T
        
        # foot: x-up, y-left, z-backward (same as thigh/shank)
        # sensor x → global [0, 0, 1] (up)
        # sensor y → global [0, 1, 0] (left)
        # sensor z → global [-1, 0, 0] (backward)
        R_foot = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ]).T
        
        q_trunk = rotmat_to_quat(R_trunk)
        q_thigh = rotmat_to_quat(R_thigh)
        q_foot = rotmat_to_quat(R_foot)
        
        return {
            'trunk': q_trunk,
            'thigh_right': q_thigh.copy(),
            'thigh_left': q_thigh.copy(),
            'shank_right': q_thigh.copy(),
            'shank_left': q_thigh.copy(),
            'foot_right': q_foot.copy(),
            'foot_left': q_foot.copy(),
        }
    
    def __init__(self):
        self.offset_quaternions: Dict[str, np.ndarray] = {}    # q_offset for each segment
        self.desired_quaternions: Dict[str, np.ndarray] = {}   # q_desired (ideal orientation)
        self.calib_quaternions: Dict[str, np.ndarray] = {}     # q_calib (average at N-pose)
        self.heading_offset: Optional[np.ndarray] = None
        self.is_calibrated = False
        self.pose_type: Optional[str] = None
        self.calibration_time: Optional[datetime] = None
        self.subject_id: Optional[str] = None
    
    @staticmethod
    def _extract_heading_quaternion(q: np.ndarray) -> np.ndarray:
        """Extract only the heading (yaw/Z-rotation) component from a quaternion."""
        w, x, y, z = q
        
        R = KinematicsProcessor.quaternion_to_rotation_matrix(q)
        
        local_forward = np.array([1.0, 0.0, 0.0])
        global_forward = R @ local_forward
        yaw = np.arctan2(global_forward[1], global_forward[0])
        
        q_heading = np.array([
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2)
        ])
        
        return q_heading
    
    @staticmethod
    def _average_quaternions(quaternions: np.ndarray) -> np.ndarray:
        """
        Average multiple quaternions using eigenvalue method.
        
        Args:
            quaternions: (N, 4) array of quaternions [w, x, y, z]
            
        Returns:
            Average quaternion [w, x, y, z]
        """
        if len(quaternions) == 0:
            raise ValueError("No quaternions provided for averaging")
        
        # Ensure consistent hemisphere (all quaternions pointing same direction)
        q0 = quaternions[0]
        for i in range(1, len(quaternions)):
            if np.dot(quaternions[i], q0) < 0:
                quaternions[i] = -quaternions[i]
        
        # Eigenvalue method for averaging
        M = np.dot(quaternions.T, quaternions)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        avg_quat = eigenvectors[:, np.argmax(eigenvalues)]
        return avg_quat / np.linalg.norm(avg_quat)
    
    def calibrate(
        self, 
        data: MotionCaptureData, 
        start_time: float, 
        end_time: float,
        pose_type: str = "N-pose"
    ):
        """
        Perform N-pose calibration.
        
        Algorithm:
            1. q_desired = ideal IMU orientation for each segment
            2. q_calib = average of measured quaternions during N-pose
            3. q_offset = conj(q_calib) * q_desired (RIGHT multiplication)
            4. Apply: q_segment = q_measured * q_offset (RIGHT multiplication)
        """
        print(f"Calibrating with {pose_type} from {start_time:.2f}s to {end_time:.2f}s")
        print("Pipeline: q_offset = conj(q_calib) * q_desired, q_segment = q_measured * q_offset")
        
        # Get desired quaternions for each segment
        desired_quats = self._get_desired_quaternions()
        
        for location, sensor_data in data.imu_data.items():
            mask = (sensor_data.timestamps >= start_time) & (sensor_data.timestamps <= end_time)
            
            if not mask.any():
                raise ValueError(f"No data found in calibration period for {location}")
            
            # Get all quaternions in calibration window
            calib_quats = sensor_data.quaternions[mask]
            
            # Normalize each quaternion
            norms = np.linalg.norm(calib_quats, axis=1, keepdims=True)
            calib_quats_norm = calib_quats / norms
            
            # Average quaternions to get q_calib
            q_calib = self._average_quaternions(calib_quats_norm)
            self.calib_quaternions[location] = q_calib.copy()
            
            # Get q_desired for this segment
            if location in desired_quats:
                q_desired = desired_quats[location]
            else:
                q_desired = np.array([1.0, 0.0, 0.0, 0.0])  # identity as fallback
            self.desired_quaternions[location] = q_desired.copy()
            
            # Compute q_offset = conj(q_calib) * q_desired (RIGHT multiplication)
            q_calib_conj = KinematicsProcessor.quaternion_conjugate(q_calib)
            q_offset = KinematicsProcessor.quaternion_multiply(q_calib_conj, q_desired)
            q_offset = q_offset / np.linalg.norm(q_offset)
            self.offset_quaternions[location] = q_offset
            
            print(f"  {location}:")
            print(f"    q_calib   = [{q_calib[0]:.4f}, {q_calib[1]:.4f}, {q_calib[2]:.4f}, {q_calib[3]:.4f}]")
            print(f"    q_desired = [{q_desired[0]:.4f}, {q_desired[1]:.4f}, {q_desired[2]:.4f}, {q_desired[3]:.4f}]")
            print(f"    q_offset  = [{q_offset[0]:.4f}, {q_offset[1]:.4f}, {q_offset[2]:.4f}, {q_offset[3]:.4f}]")
        
        # Extract heading offset from trunk
        if 'trunk' in self.calib_quaternions:
            q_trunk = self.calib_quaternions['trunk']
            q_heading = self._extract_heading_quaternion(q_trunk)
            self.heading_offset = q_heading.copy()
            heading_angle = 2 * np.arctan2(q_heading[3], q_heading[0]) * 180 / np.pi
            print(f"  Trunk heading: {heading_angle:.1f}° from global X")
        
        self.is_calibrated = True
        self.pose_type = pose_type
        self.calibration_time = datetime.now()
        data.calibration_pose = pose_type
        data.calibration_duration = end_time - start_time
        data.calibration_start_time = start_time
        print("N-pose calibration complete!")
    
    def _average_quaternions(self, quaternions: np.ndarray) -> np.ndarray:
        """Average multiple quaternions using eigenvalue method."""
        if len(quaternions) == 0:
            raise ValueError("No quaternions provided for averaging")
        M = np.dot(quaternions.T, quaternions)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        avg_quat = eigenvectors[:, np.argmax(eigenvalues)]
        return avg_quat / np.linalg.norm(avg_quat)
    
    def get_offset_quaternion(self, location: str) -> Optional[np.ndarray]:
        """Get offset quaternion for a sensor location"""
        return self.offset_quaternions.get(location)
    
    def get_desired_quaternion(self, location: str) -> Optional[np.ndarray]:
        """Get desired quaternion for a sensor location"""
        return self.desired_quaternions.get(location)
    
    def apply_calibration(self, quaternion: np.ndarray, location: str) -> np.ndarray:
        """
        Apply calibration using RIGHT multiplication.
        
        q_segment = q_measured * q_offset
        """
        if not self.is_calibrated or location not in self.offset_quaternions:
            return quaternion
        
        q_offset = self.offset_quaternions[location]
        q_segment = KinematicsProcessor.quaternion_multiply(quaternion, q_offset)
        q_segment = q_segment / np.linalg.norm(q_segment)

        return q_segment

    def validate_calibration_pose(
        self, 
        data: MotionCaptureData, 
        start_time: float, 
        end_time: float,
        max_movement_threshold: float = 0.1
    ) -> bool:
        """
        Validate that the subject was stationary during calibration
        
        Args:
            data: Motion capture data
            start_time: Start of calibration pose
            end_time: End of calibration pose
            max_movement_threshold: Maximum allowed gyroscope magnitude (rad/s)
            
        Returns:
            True if pose was sufficiently static
        """
        for location, sensor_data in data.imu_data.items():
            mask = (sensor_data.timestamps >= start_time) & (sensor_data.timestamps <= end_time)
            gyro_data = sensor_data.gyroscopes[mask]
            
            # Check if gyroscope magnitude is below threshold
            gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
            if np.max(gyro_magnitude) > max_movement_threshold:
                print(f"Warning: Excessive movement detected in {location} during calibration")
                return False
        
        return True
    
    def save_calibration(self, filepath: str):
        """
        Save calibration data to file
        
        Args:
            filepath: Path to save calibration file (.cal extension)
        """
        if not self.is_calibrated:
            raise ValueError("No calibration data to save. Perform calibration first.")
        
        # Ensure .cal extension
        filepath = str(Path(filepath).with_suffix(self.CALIBRATION_EXTENSION))
        
        # Prepare calibration data
        calib_data = {
            'version': '4.0',  # Version 4.0: q_segment = q_measured * q_offset (RIGHT multiplication)
            'pose_type': self.pose_type,
            'calibration_time': self.calibration_time.isoformat() if self.calibration_time else None,
            'subject_id': self.subject_id,
            'heading_offset': self.heading_offset.tolist() if self.heading_offset is not None else None,
            'offset_quaternions': {},
            'desired_quaternions': {},
            'calib_quaternions': {}
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for location, quat in self.offset_quaternions.items():
            calib_data['offset_quaternions'][location] = quat.tolist()
        
        for location, quat in self.desired_quaternions.items():
            calib_data['desired_quaternions'][location] = quat.tolist()
        
        for location, quat in self.calib_quaternions.items():
            calib_data['calib_quaternions'][location] = quat.tolist()
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"Calibration saved to: {filepath}")
    
    def load_calibration(self, filepath: str):
        """
        Load calibration data from file
        
        Args:
            filepath: Path to calibration file (.cal)
        """
        with open(filepath, 'r') as f:
            calib_data = json.load(f)
        
        version = calib_data.get('version', '1.0')
        print(f"Loading calibration file version {version}")
        
        # Load offset quaternions (v4.0+)
        self.offset_quaternions = {}
        if 'offset_quaternions' in calib_data:
            for location, quat_list in calib_data['offset_quaternions'].items():
                self.offset_quaternions[location] = np.array(quat_list)
        elif 'correction_quaternions' in calib_data:
            # Backward compatibility with older versions
            print("  Warning: Loading old format. Converting correction to offset.")
            for location, quat_list in calib_data['correction_quaternions'].items():
                self.offset_quaternions[location] = np.array(quat_list)
        
        # Load desired quaternions
        self.desired_quaternions = {}
        if 'desired_quaternions' in calib_data:
            for location, quat_list in calib_data['desired_quaternions'].items():
                self.desired_quaternions[location] = np.array(quat_list)
        
        # Load calib quaternions (v4.0+)
        self.calib_quaternions = {}
        if 'calib_quaternions' in calib_data:
            for location, quat_list in calib_data['calib_quaternions'].items():
                self.calib_quaternions[location] = np.array(quat_list)
        
        # Load heading offset
        self.heading_offset = None
        if 'heading_offset' in calib_data and calib_data['heading_offset'] is not None:
            self.heading_offset = np.array(calib_data['heading_offset'])
            heading_angle = 2 * np.arctan2(self.heading_offset[3], self.heading_offset[0]) * 180 / np.pi
            print(f"  Heading offset: {-heading_angle:.1f}°")
        
        self.pose_type = calib_data.get('pose_type')
        self.subject_id = calib_data.get('subject_id')
        
        if calib_data.get('calibration_time'):
            self.calibration_time = datetime.fromisoformat(calib_data['calibration_time'])
        
        self.is_calibrated = True
        print(f"Calibration loaded from: {filepath}")
        print(f"  Version: {version}")
        print(f"  Pose type: {self.pose_type}")
        print(f"  Sensors: {list(self.offset_quaternions.keys())}")
    
    def apply_to_data(self, data: MotionCaptureData) -> MotionCaptureData:
        """
        Apply calibration to motion capture data.
        
        q_segment = q_measured * q_offset (RIGHT multiplication)
        """
        if not self.is_calibrated:
            raise ValueError("No calibration loaded. Load calibration first.")
        
        print(f"Applying N-pose calibration to data: {data.session_id}")
        print("  Pipeline: q_segment = q_measured * q_offset (RIGHT multiplication)")
        
        from copy import deepcopy
        calibrated_data = deepcopy(data)
        
        if self.heading_offset is not None:
            calibrated_data.heading_offset = self.heading_offset.copy()
        
        for location, sensor_data in calibrated_data.imu_data.items():
            if location not in self.offset_quaternions:
                print(f"  Warning: No offset quaternion for {location}, skipping")
                continue
            
            n_samples = len(sensor_data.quaternions)
            if n_samples == 0:
                continue
            
            q_offset = self.offset_quaternions[location]
            
            # Normalize all raw quaternions first
            q_measured = sensor_data.quaternions
            q_measured_norm = KinematicsProcessor.quaternion_normalize(q_measured)
            
            # Apply offset: q_segment = q_measured * q_offset (RIGHT multiplication)
            q_segment = KinematicsProcessor.quaternion_multiply(q_measured_norm, q_offset)
            q_segment = KinematicsProcessor.quaternion_normalize(q_segment)
            
            sensor_data.quaternions = q_segment
            print(f"  Calibrated {location}: {n_samples} samples")
        
        calibrated_data.calibration_pose = self.pose_type
        print("N-pose calibration applied successfully!")
        return calibrated_data
