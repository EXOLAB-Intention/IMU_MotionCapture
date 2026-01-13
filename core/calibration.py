"""
Calibration processor for IMU motion capture
Handles initial pose calibration (T-pose, N-pose) to establish reference frames
"""
import numpy as np
from typing import Optional, Dict

from core.imu_data import MotionCaptureData, IMUSensorData


class CalibrationProcessor:
    """Processes calibration poses to establish reference orientations"""
    
    def __init__(self):
        self.reference_orientations: Dict[str, np.ndarray] = {}
        self.is_calibrated = False
    
    def calibrate(
        self, 
        data: MotionCaptureData, 
        start_time: float, 
        end_time: float,
        pose_type: str = "T-pose"
    ):
        """
        Perform calibration using static pose data
        
        Args:
            data: Motion capture data containing calibration period
            start_time: Start of calibration pose (seconds)
            end_time: End of calibration pose (seconds)
            pose_type: Type of calibration pose ("T-pose" or "N-pose")
        """
        print(f"Calibrating with {pose_type} from {start_time:.2f}s to {end_time:.2f}s")
        
        # Extract calibration data for each sensor
        for location, sensor_data in data.imu_data.items():
            # Get time slice
            mask = (sensor_data.timestamps >= start_time) & (sensor_data.timestamps <= end_time)
            
            if not mask.any():
                raise ValueError(f"No data found in calibration period for {location}")
            
            # Average quaternions during calibration pose
            calib_quaternions = sensor_data.quaternions[mask]
            reference_quat = self._average_quaternions(calib_quaternions)
            
            self.reference_orientations[location] = reference_quat
            
            print(f"  {location}: reference orientation established")
        
        self.is_calibrated = True
        print("Calibration complete!")
    
    def _average_quaternions(self, quaternions: np.ndarray) -> np.ndarray:
        """
        Average multiple quaternions
        
        Args:
            quaternions: (N, 4) array of quaternions [w, x, y, z]
            
        Returns:
            (4,) averaged quaternion
            
        Note:
            TODO: Implement proper quaternion averaging (e.g., using eigenvalue method)
            Currently using simple mean normalization as placeholder
        """
        # Placeholder: simple mean and normalize
        mean_quat = np.mean(quaternions, axis=0)
        mean_quat = mean_quat / np.linalg.norm(mean_quat)
        return mean_quat
    
    def get_reference_orientation(self, location: str) -> Optional[np.ndarray]:
        """Get reference orientation for a sensor location"""
        return self.reference_orientations.get(location)
    
    def apply_calibration(self, quaternion: np.ndarray, location: str) -> np.ndarray:
        """
        Apply calibration to transform quaternion to calibrated frame
        
        Args:
            quaternion: Current orientation quaternion [w, x, y, z]
            location: Sensor location
            
        Returns:
            Calibrated quaternion
        """
        if not self.is_calibrated or location not in self.reference_orientations:
            return quaternion
        
        # TODO: Implement proper quaternion transformation
        # q_calibrated = q_current * q_reference_inverse
        
        return quaternion
    
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
