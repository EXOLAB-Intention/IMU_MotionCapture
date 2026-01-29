"""
Configuration settings for IMU Motion Capture System
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModeConfig:
    """Body mode configuration"""
    mode_type: str = "Lower-body" # Lower-body or Upper-body

    # Mode options
    MODE_TYPES = ["Lower-body", "Upper-body"]

@dataclass
class IMUConfig:
    """IMU sensor configuration"""
    model: str = "Xsens MTi-630"
    sampling_frequency: int = 1000  # Hz
    num_sensors: int = 7
    
    # IMU placement
    sensor_locations_lower = [
        "trunk",
        "thigh_right",
        "shank_right",
        "foot_right",
        "thigh_left",
        "shank_left",
        "foot_left"
    ]

    sensor_locations_upper = [
        "pelvis",
        "chest",
        "head",
        "upperarm_right",
        "lowerarm_right",
        "upperarm_left",
        "lowerarm_left"
    ]

    def get_locations(self, mode: str):
        return self.sensor_locations_upper if mode == "Upper-body" else self.sensor_locations_lower


@dataclass
class CalibrationConfig:
    """Calibration pose configuration"""
    pose_type: str = "T-pose"  # T-pose or N-pose
    duration: float = 2.0  # seconds
    
    # Pose options
    POSE_TYPES = ["T-pose", "N-pose"]


@dataclass
class SubjectConfig:
    """Subject physical parameters"""
    height: float = 170.0  # cm
    shoe_size: float = 270.0  # mm
    name: Optional[str] = None
    
    # Segment length ratios (relative to height)
    trunk_ratio: float = 0.288
    thigh_ratio: float = 0.232
    shank_ratio: float = 0.246
    foot_ratio: float = 0.152
    upperarm_ratio: float = 0.186
    lowerarm_ratio: float = 0.146
    pelvis_chest_ratio: float = 0.190
    chest_neck_ratio: float = 0.150
    chest_shoulder_ratio: float = 0.156


@dataclass
class EnvironmentConfig:
    """Environment type configuration"""
    terrain_type: str = "flat"  # flat, stairs, irregular
    
    # Terrain options
    TERRAIN_TYPES = ["flat", "stairs", "irregular"]


@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    calculate_joint_angles: bool = True
    calculate_trunk_angle: bool = True
    detect_foot_contact: bool = True
    calculate_velocity: bool = True
    
    # Filter parameters
    lowpass_cutoff: float = 10.0  # Hz
    filter_order: int = 4


@dataclass
class GUIConfig:
    """GUI configuration"""
    window_width: int = 1600
    window_height: int = 900
    
    # 3D visualization
    visualization_fps: int = 60
    background_color: tuple = (0.2, 0.2, 0.2)
    
    # Graph settings
    graph_line_width: float = 2.0
    graph_colors = {
        'hip': '#FF6B6B',
        'knee': '#4ECDC4',
        'ankle': '#45B7D1',
        'shoulder': "#6BFF97",
        'elbow': "#4ECD8D"
    }


class AppSettings:
    """Application-wide settings"""
    
    def __init__(self):
        self.mode = ModeConfig()
        self.imu = IMUConfig()
        self.calibration = CalibrationConfig()
        self.subject = SubjectConfig()
        self.environment = EnvironmentConfig()
        self.processing = ProcessingConfig()
        self.gui = GUIConfig()
        
        # File paths
        self.current_file: Optional[str] = None
        self.working_directory: Optional[str] = None
        self.recent_files: list = []
        
        self.refresh_sensor_mapping()

    def refresh_sensor_mapping(self):
        """현재 모드에 맞는 센서 위치들로 매핑 테이블 초기화"""
        locations = self.imu.get_locations(self.mode.mode_type)
        self.sensor_mapping = {loc: None for loc in locations}

        # # Sensor mapping
        # self.sensor_mapping: dict = {
        #     "trunk": None,
        #     "thigh_right": None,
        #     "shank_right": None,
        #     "foot_right": None,
        #     "thigh_left": None,
        #     "shank_left": None,
        #     "foot_left": None
        # }
    
    def update_sensor_mapping(self, location: str, sensor_id: int):
        """Update sensor ID for a specific body location"""
        if location in self.sensor_mapping:
            self.sensor_mapping[location] = sensor_id
    
    def get_segment_length(self, segment: str) -> float:
        """Calculate segment length based on subject height"""
        ratios = {
            'trunk': self.subject.trunk_ratio,
            'thigh': self.subject.thigh_ratio,
            'shank': self.subject.shank_ratio,
            'foot': self.subject.foot_ratio,
            'upperarm': self.subject.upperarm_ratio,
            'lowerarm': self.subject.lowerarm_ratio
        }
        return self.subject.height * ratios.get(segment, 0.0)
    
    def validate_sensor_mapping(self) -> bool:
        """Check if all sensors are mapped"""
        return all(sid is not None for sid in self.sensor_mapping.values())


# Global settings instance
app_settings = AppSettings()
