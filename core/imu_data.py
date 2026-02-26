"""
IMU data structures and containers
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime

@dataclass
class IMUSample:
    """Single IMU sensor sample"""
    timestamp: float  # seconds
    quaternion: np.ndarray  # [w, x, y, z]
    acceleration: np.ndarray  # [ax, ay, az] in m/s^2
    gyroscope: np.ndarray  # [gx, gy, gz] in rad/s
    
    def __post_init__(self):
        """Validate data shapes"""
        assert self.quaternion.shape == (4,), "Quaternion must be (4,)"
        assert self.acceleration.shape == (3,), "Acceleration must be (3,)"
        assert self.gyroscope.shape == (3,), "Gyroscope must be (3,)"


@dataclass
class IMUSensorData:
    """Time series data for a single IMU sensor"""
    sensor_id: int
    location: str  # back, thigh_right, etc.
    timestamps: np.ndarray  # (N,)
    quaternions: np.ndarray  # (N, 4) [w, x, y, z]
    accelerations: np.ndarray  # (N, 3)
    gyroscopes: np.ndarray  # (N, 3)
    sampling_frequency: float
    
    def __post_init__(self):
        """Validate data consistency"""
        n_samples = len(self.timestamps)
        assert self.quaternions.shape == (n_samples, 4)
        assert self.accelerations.shape == (n_samples, 3)
        assert self.gyroscopes.shape == (n_samples, 3)
    
    def get_sample(self, index: int) -> IMUSample:
        """Get a single sample at index"""
        return IMUSample(
            timestamp=self.timestamps[index],
            quaternion=self.quaternions[index],
            acceleration=self.accelerations[index],
            gyroscope=self.gyroscopes[index]
        )
    
    def get_time_slice(self, start_time: float, end_time: float) -> 'IMUSensorData':
        """Extract data within time range"""
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        return IMUSensorData(
            sensor_id=self.sensor_id,
            location=self.location,
            timestamps=self.timestamps[mask],
            quaternions=self.quaternions[mask],
            accelerations=self.accelerations[mask],
            gyroscopes=self.gyroscopes[mask],
            sampling_frequency=self.sampling_frequency
        )
    
    @property
    def duration(self) -> float:
        """Total duration in seconds"""
        return self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 0 else 0.0
    
    @property
    def n_samples(self) -> int:
        """Number of samples"""
        return len(self.timestamps)


@dataclass
class JointAngles:
    """Joint angle data following ISB convention.
    
    Angles are computed using XYZ Cardan sequence and expressed in degrees.
    
    Column order for each joint array (N, 3):
        - Column 0: Flexion/Extension (X-axis rotation)
            - Hip: flexion(+) / extension(-)
            - Knee: flexion(+) / extension(-)
            - Ankle: dorsiflexion(+) / plantarflexion(-)
        
        - Column 1: Abduction/Adduction (Y-axis rotation)
            - Hip: adduction(+) / abduction(-)
            - Knee: varus(+) / valgus(-)
            - Ankle: inversion(+) / eversion(-)
        
        - Column 2: Internal/External Rotation (Z-axis rotation)
            - All joints: internal rotation(+) / external rotation(-)
    """
    timestamps: np.ndarray  # (N,)
    
    # Joint angles in degrees [flexion, abduction, rotation]
    # Lower body joints
    hip_right: Optional[np.ndarray] = None  # (N, 3)
    hip_left: Optional[np.ndarray] = None  # (N, 3)
    knee_right: Optional[np.ndarray] = None  # (N, 3)
    knee_left: Optional[np.ndarray] = None  # (N, 3)
    ankle_right: Optional[np.ndarray] = None  # (N, 3)
    ankle_left: Optional[np.ndarray] = None  # (N, 3)

    # Upper body joints
    spine: Optional[np.ndarray] = None  # (N, 3)
    neck: Optional[np.ndarray] = None  # (N, 3)
    shoulder_right: Optional[np.ndarray] = None  # (N, 3)
    shoulder_left: Optional[np.ndarray] = None  # (N, 3)
    elbow_right: Optional[np.ndarray] = None  # (N, 3)
    elbow_left: Optional[np.ndarray] = None  # (N, 3)
    
    def get_joint_angle(self, joint: str, side: str) -> np.ndarray:
        """Get specific joint angle time series"""
        if joint in ['spine', 'neck']:
            return getattr(self, joint, None)
        
        attr_name = f"{joint}_{side}"
        return getattr(self, attr_name, None)


@dataclass
class KinematicsData:
    """Computed kinematics data"""
    timestamps: np.ndarray  # (N,)
    
    # Back orientation relative to ground (degrees)
    back_angle: np.ndarray  # (N, 3) [pitch, roll, yaw]

    # Foot contact detection
    foot_contact_right: np.ndarray  # (N,) boolean
    foot_contact_left: np.ndarray  # (N,) boolean

    # Back velocity
    back_velocity: np.ndarray  # (N, 3) [vx, vy, vz] in m/s
    back_speed: np.ndarray  # (N,) scalar speed in m/s
    
    # Stride detection
    stride_times_right: List[float] = field(default_factory=list)
    stride_times_left: List[float] = field(default_factory=list)


@dataclass
class MotionCaptureData:
    """Complete motion capture session data"""
    # Metadata
    session_id: str
    creation_time: datetime
    subject_id: Optional[str] = None
    
    # Raw IMU data (keyed by location)
    imu_data: Dict[str, IMUSensorData] = field(default_factory=dict)
    
    # Processed data
    joint_angles: Optional[JointAngles] = None
    kinematics: Optional[KinematicsData] = None
    
    # Gait parameters
    gait_start_frame: Optional[int] = None  # Frame where first foot leaves contact
    gait_end_frame: Optional[int] = None    # Frame where both feet return to contact
    foot_contact_right: Optional[np.ndarray] = None  # Right foot contact boolean array
    foot_contact_left: Optional[np.ndarray] = None   # Left foot contact boolean array
    
    # Calibration data
    calibration_pose: Optional[str] = None
    calibration_duration: Optional[float] = None
    calibration_start_time: Optional[float] = None
    heading_offset: Optional[np.ndarray] = None  # Back heading at N-pose for visualization
    is_calibrated: bool = False
    
    # Processing status
    is_processed: bool = False
    processing_timestamp: Optional[datetime] = None
    
    # Notes
    notes: str = ""
    
    def add_imu_sensor_data(self, sensor_data: IMUSensorData):
        """Add IMU sensor data for a location"""
        self.imu_data[sensor_data.location] = sensor_data
    
    def get_time_range(self) -> tuple:
        """Get min and max timestamp across all sensors"""
        if not self.imu_data:
            return (0.0, 0.0)
        
        min_time = min(data.timestamps[0] for data in self.imu_data.values())
        max_time = max(data.timestamps[-1] for data in self.imu_data.values())
        return (min_time, max_time)
    
    def get_time_slice(self, start_time: float, end_time: float) -> 'MotionCaptureData':
        """Extract data within time range"""
        sliced_data = MotionCaptureData(
            session_id=self.session_id + "_slice",
            creation_time=datetime.now(),
            subject_id=self.subject_id,
            calibration_pose=self.calibration_pose,
            notes=self.notes
        )
        
        # Slice IMU data
        for location, sensor_data in self.imu_data.items():
            sliced_data.imu_data[location] = sensor_data.get_time_slice(start_time, end_time)
        
        # TODO: Slice joint angles and kinematics data
        
        return sliced_data
    
    @property
    def duration(self) -> float:
        """Total duration in seconds"""
        start, end = self.get_time_range()
        return end - start
    
    @property
    def required_locations(self) -> List[str]:
        """Get required sensor locations based on capture mode"""

        from config.settings import app_settings
        current_mode = app_settings.mode.mode_type

        if current_mode == 'Lower-body':
            return ["back", "thigh_right", "shank_right", "foot_right",
                    "thigh_left", "shank_left", "foot_left"]
        else:
            return ['pelvis', 'chest', 'head', 'upperarm_right', 
                    'lowerarm_right', 'upperarm_left', 'lowerarm_left']
    
    @property
    def has_all_sensors(self) -> bool:
        """Check if all required sensors are present"""
        return all(loc in self.imu_data for loc in self.required_locations)
