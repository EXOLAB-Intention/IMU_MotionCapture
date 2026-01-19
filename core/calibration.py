"""
Calibration processor for IMU motion capture
Handles initial pose calibration (T-pose, N-pose) to establish reference frames
"""
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from core.imu_data import MotionCaptureData, IMUSensorData
from core.kinematics import KinematicsProcessor


class CalibrationProcessor:
    """Processes calibration poses to establish reference orientations"""
    
    CALIBRATION_EXTENSION = '.cal'
    
    @staticmethod
    def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion [w, x, y, z].
        
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
    def _get_desired_orientations() -> Dict[str, np.ndarray]:
        """
        Get desired orientations for each segment in N-pose.
        
        Global coordinate frame (ground/lab frame):
            X: forward (사람의 앞)
            Y: left (사람의 왼쪽)
            Z: up (사람의 위)
        
        Segment local frames in N-pose:
            Trunk: local_x=골반→머리(up), local_y=앞(forward), local_z=왼쪽(left)
            Thigh/Shank: local_x=proximal→distal(down), local_y=뒤(back), local_z=오른쪽(right)
            Foot: local_x=proximal→distal(forward), local_y=발바닥(down), local_z=오른쪽(right)
        
        Returns:
            Dictionary mapping segment names to desired quaternions [w,x,y,z]
        """
        R_trunk = np.array([
            [0, 1, 0],   # global X = local Y
            [0, 0, 1],   # global Y = local Z
            [1, 0, 0]    # global Z = local X
        ])
        
        R_thigh_right = np.array([
            [0, -1, 0],   # global X = -local Y (back)
            [0, 0, -1],   # global Y = -local Z (right)
            [1,  0, 0]    # global Z =  local X (up)
        ])
        
        R_thigh_left = np.array([
            [0, -1, 0],   # global X = -local Y (back)
            [0, 0, -1],   # global Y = -local Z (right)
            [1,  0, 0]    # global Z =  local X (up)
        ])
        
        R_foot_right = np.array([
            [-1, 0, 0],   # global X = -local X (forward)
            [0, 0, -1],   # global Y = -local Z (right)
            [0, -1, 0]    # global Z = -local Y (발바닥은 아래)
        ])
        
        R_foot_left = np.array([
            [-1, 0, 0],   # global X = -local X (forward)
            [0, 0, -1],   # global Y = -local Z (right)
            [0, -1, 0]    # global Z = -local Y (발바닥은 아래)
        ])
        
        return {
            'trunk': CalibrationProcessor._rotation_matrix_to_quaternion(R_trunk),
            'thigh_right': CalibrationProcessor._rotation_matrix_to_quaternion(R_thigh_right),
            'thigh_left': CalibrationProcessor._rotation_matrix_to_quaternion(R_thigh_left),
            'shank_right': CalibrationProcessor._rotation_matrix_to_quaternion(R_thigh_right),  # Same as thigh
            'shank_left': CalibrationProcessor._rotation_matrix_to_quaternion(R_thigh_left),    # Same as thigh
            'foot_right': CalibrationProcessor._rotation_matrix_to_quaternion(R_foot_right),
            'foot_left': CalibrationProcessor._rotation_matrix_to_quaternion(R_foot_left),
        }
    
    def __init__(self):
        self.correction_quaternions: Dict[str, np.ndarray] = {}  # qCorr for each segment
        self.is_calibrated = False
        self.pose_type: Optional[str] = None
        self.calibration_time: Optional[datetime] = None
        self.subject_id: Optional[str] = None
    
    def calibrate(
        self, 
        data: MotionCaptureData, 
        start_time: float, 
        end_time: float,
        pose_type: str = "N-pose"
    ):
        """
        Perform N-pose calibration following MATLAB workflow.
        
        Computes correction quaternions to align IMU measurements with desired orientations.
        
        Algorithm (from MATLAB):
            1. Extract first frame (or average) during calibration period
            2. For each segment:
                qD = desired orientation (identity for N-pose)
                qRaw = measured raw quaternion at calibration frame
                qCorr = qD * qRaw^{-1}  (correction quaternion)
            3. Save qCorr for each segment
        
        Later, during motion trial processing:
            q_corrected = qCorr * qRaw
        
        This ensures that N-pose orientation becomes identity ("standing straight").
        
        Args:
            data: Motion capture data containing calibration period
            start_time: Start of calibration pose (seconds)
            end_time: End of calibration pose (seconds)
            pose_type: Type of calibration pose (only "N-pose" supported)
        """
        print(f"Calibrating with {pose_type} from {start_time:.2f}s to {end_time:.2f}s")
        
        if pose_type != "N-pose":
            print(f"Warning: Only N-pose is currently supported. Using N-pose calibration.")
            pose_type = "N-pose"
        
        # Extract calibration data for each sensor
        for location, sensor_data in data.imu_data.items():
            # Get time slice
            mask = (sensor_data.timestamps >= start_time) & (sensor_data.timestamps <= end_time)
            
            if not mask.any():
                raise ValueError(f"No data found in calibration period for {location}")
            
            # Get first frame quaternion (MATLAB uses stand_pose_time)
            calib_indices = np.where(mask)[0]
            first_frame_idx = calib_indices[0]
            q_raw = sensor_data.quaternions[first_frame_idx]
            
            # Normalize raw quaternion
            q_raw_norm = q_raw / np.linalg.norm(q_raw)
            
            # Get desired orientation for this segment
            desired_orientations = self._get_desired_orientations()
            if location in desired_orientations:
                q_desired = desired_orientations[location]
            else:
                # Default to identity if not defined
                q_desired = np.array([1.0, 0.0, 0.0, 0.0])
                print(f"  Warning: No desired orientation defined for {location}, using identity")
            
            # Compute correction quaternion: qCorr = qD * qRaw^{-1}
            # This is the transformation that takes qRaw to qD
            q_raw_inv = KinematicsProcessor.quaternion_inverse(q_raw_norm)
            q_corr = KinematicsProcessor.quaternion_multiply(q_desired, q_raw_inv)
            
            self.correction_quaternions[location] = q_corr
            
            print(f"  {location}: correction quaternion = [{q_corr[0]:.4f}, {q_corr[1]:.4f}, {q_corr[2]:.4f}, {q_corr[3]:.4f}]")
        
        self.is_calibrated = True
        self.pose_type = pose_type
        self.calibration_time = datetime.now()
        data.calibration_pose = pose_type
        data.calibration_duration = end_time - start_time
        data.calibration_start_time = start_time
        print("N-pose calibration complete!")
    
    def _average_quaternions(self, quaternions: np.ndarray) -> np.ndarray:
        """
        Average multiple quaternions
        
        Args:
            quaternions: (N, 4) array of quaternions [w, x, y, z]
            
        Returns:
            (4,) averaged quaternion
        """
        # Placeholder: simple mean and normalize
        if len(quaternions) == 0:
            raise ValueError("No quaternions provided for averaging")
        M = np.dot(quaternions.T, quaternions)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        avg_quat = eigenvectors[:, np.argmax(eigenvalues)]
        return avg_quat / np.linalg.norm(avg_quat)
    
    def get_correction_quaternion(self, location: str) -> Optional[np.ndarray]:
        """Get correction quaternion for a sensor location"""
        return self.correction_quaternions.get(location)
    
    def apply_calibration(self, quaternion: np.ndarray, location: str) -> np.ndarray:
        """
        Apply calibration to transform quaternion to calibrated frame.
        
        Following MATLAB approach:
            q_corrected = qCorr * qRaw
        
        This applies the correction transformation computed during calibration.
        
        Args:
            quaternion: Current raw orientation quaternion [w, x, y, z]
            location: Sensor location
            
        Returns:
            Calibrated quaternion (aligned to desired N-pose orientation)
        """
        if not self.is_calibrated or location not in self.correction_quaternions:
            return quaternion
        
        q_corr = self.correction_quaternions[location]
        # Apply correction: q_calibrated = qCorr * qRaw (MATLAB: quatmultiply(qCorr, qRaw))
        q_calibrated = KinematicsProcessor.quaternion_multiply(q_corr, quaternion)
        
        # Normalize result
        q_calibrated = q_calibrated / np.linalg.norm(q_calibrated)

        return q_calibrated

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
            'version': '2.0',  # Updated version for new calibration method
            'pose_type': self.pose_type,
            'calibration_time': self.calibration_time.isoformat() if self.calibration_time else None,
            'subject_id': self.subject_id,
            'correction_quaternions': {}
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for location, quat in self.correction_quaternions.items():
            calib_data['correction_quaternions'][location] = quat.tolist()
        
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
        
        # Load correction quaternions (support both old and new format)
        self.correction_quaternions = {}
        if 'correction_quaternions' in calib_data:
            # New format (v2.0)
            for location, quat_list in calib_data['correction_quaternions'].items():
                self.correction_quaternions[location] = np.array(quat_list)
        elif 'reference_orientations' in calib_data:
            # Old format (v1.0) - convert to correction quaternions
            print("  Warning: Loading old calibration format (v1.0). This may not work correctly with new N-pose method.")
            for location, quat_list in calib_data['reference_orientations'].items():
                # Old format stored reference, need to compute correction
                # For backward compatibility, use identity as correction
                self.correction_quaternions[location] = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.pose_type = calib_data.get('pose_type')
        self.subject_id = calib_data.get('subject_id')
        
        if calib_data.get('calibration_time'):
            self.calibration_time = datetime.fromisoformat(calib_data['calibration_time'])
        
        self.is_calibrated = True
        print(f"Calibration loaded from: {filepath}")
        print(f"  Version: {version}")
        print(f"  Pose type: {self.pose_type}")
        print(f"  Sensors: {list(self.correction_quaternions.keys())}")
    
    def apply_to_data(self, data: MotionCaptureData) -> MotionCaptureData:
        """
        Apply calibration to motion capture data.
        
        Uses incremental rotation approach:
            q_segment(t) = [q_imu(t) * q_imu(1)^{-1}] * qCorr * q_imu(1)
        
        Steps:
            1. Compute relative rotation from first frame: q_rel(t) = q_imu(t) * q_imu(1)^{-1}
            2. Apply correction: q_temp = q_rel * qCorr
            3. Transform back to original frame: q_segment(t) = q_temp * q_imu(1)
        
        This ensures consistent reference frame across trials while preserving initial orientation.
        
        Args:
            data: Motion capture data to calibrate
            
        Returns:
            Calibrated motion capture data
        """
        if not self.is_calibrated:
            raise ValueError("No calibration loaded. Load calibration first.")
        
        print(f"Applying N-pose calibration to data: {data.session_id}")
        
        # Create calibrated copy
        from copy import deepcopy
        calibrated_data = deepcopy(data)
        
        # Apply calibration to each sensor
        for location, sensor_data in calibrated_data.imu_data.items():
            if location not in self.correction_quaternions:
                print(f"  Warning: No correction quaternion for {location}, skipping")
                continue
            
            n_samples = len(sensor_data.quaternions)
            if n_samples == 0:
                continue
            
            # Get first frame quaternion (reference)
            q_first = sensor_data.quaternions[0]
            q_first_norm = q_first / np.linalg.norm(q_first)
            q_first_inv = KinematicsProcessor.quaternion_inverse(q_first_norm)
            
            # Get correction quaternion
            q_corr = self.correction_quaternions[location]
            
            # Apply to all frames: q_segment(t) = [q_imu(t) * q_imu(1)^{-1}] * qCorr * q_imu(1)
            for i in range(n_samples):
                q_imu = sensor_data.quaternions[i]
                q_imu_norm = q_imu / np.linalg.norm(q_imu)
                
                # Step 1: Compute relative rotation from first frame
                q_rel = KinematicsProcessor.quaternion_multiply(q_imu_norm, q_first_inv)
                
                # Step 2: Apply correction
                q_temp = KinematicsProcessor.quaternion_multiply(q_rel, q_corr)
                
                # Step 3: Transform back to original frame
                q_segment = KinematicsProcessor.quaternion_multiply(q_temp, q_first_norm)
                
                # Normalize and store
                sensor_data.quaternions[i] = q_segment / np.linalg.norm(q_segment)
            
            print(f"  Calibrated {location}: {n_samples} samples")
        
        # Mark as calibrated
        calibrated_data.calibration_pose = self.pose_type
        
        print("N-pose calibration applied successfully!")
        return calibrated_data
