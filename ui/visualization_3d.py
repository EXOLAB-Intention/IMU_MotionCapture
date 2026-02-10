"""
3D visualization widget for displaying body segments using PyQtGraph
Provides interactive 3D skeleton rendering with smooth playback
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSlider
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, pyqtSlot
import numpy as np

from config.settings import app_settings

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("Warning: PyQtGraph not available. 3D visualization will not work.")


class Visualization3D(QWidget):
    """
    3D visualization of body segments using IMU data.
    Provides interactive viewing with rotation, zoom, and smooth playback.
    
    Body structure (lower body):
        - Trunk (torso)
        - Left/Right Thigh (hip to knee)
        - Left/Right Shank (knee to ankle)
        - Left/Right Foot (ankle to toe)
    """
    
    # Signals
    frame_changed = pyqtSignal(int)  # current frame index
    
    # Segment lengths (meters) - approximate human proportions
    SEGMENT_LENGTHS = {
        'trunk': 0.50,     # Torso height
        'thigh': 0.40,     # Hip to knee
        'shank': 0.42,     # Knee to ankle
        'foot': 0.25,      # Ankle to toe
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = None
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0  # 1x speed
        self._updating = False  # Flag to prevent recursive signal loops
        self.timer = QTimer()
        self.timer.timeout.connect(self._advance_frame)
        
        # Segment lengths (meters) - initialized from settings
        self.SEGMENT_LENGTHS = {
            'trunk': 0.50,     # Torso height
            'thigh': 0.40,     # Hip to knee
            'shank': 0.42,     # Knee to ankle
            'foot': 0.25,      # Ankle to toe
            # Upper body segments can be added similarly
            'abdomen': 0.25,     # Pelvis to chest
            'chest': 0.20,       # Chest to head
            'head': 0.20,        # Head height
            'upperarm': 0.25,  # Shoulder to elbow
            'lowerarm': 0.25   # Elbow to wrist
        }
        
        # Load initial segment lengths from settings
        self._load_segment_lengths_from_settings()
        
        # Playback state
        self.playback_start_time = 0.0  # Real-time start (seconds)
        self.playback_start_frame = 0   # Frame index at start
        self.sampling_rate = 500.0      # Hz (will be updated from data)
        
        # Grid wrapping parameters (for treadmill effect)
        self.grid_wrap_threshold = 0.4  # Faster wrap/reset cycle
        self.grid_wrap_period = 0.4  # Scaled with grid_scale (0.4/0.2 = 2x)
        
        # 3D Graphics items
        self.skeleton_items = {}
        self.joint_items = {}
        self.axis_items = {}  # Segment coordinate axes
        self.grid = None  # Grid item for ground plane
        self.grid_scale = 0.4  # Store grid scale (larger for better visibility)
        
        # Foot contact tracking for grid movement
        self.foot_contact_right = None
        self.foot_contact_left = None
        self.gait_start_frame = None
        self.gait_end_frame = None
        self.grid_offset = np.array([0.0, 0.0, 0.0])  # Grid position offset
        self.reference_foot_pre_gait = None  # Which foot to track before gait_start
        self.reference_foot_post_gait = None  # Which foot to track after gait_end
        self._prev_tracked_foot = None  # Track which foot we're currently following
        self._prev_foot_position = None  # Previous position of tracked foot for displacement calc
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("<b>3D Visualization</b>")
        layout.addWidget(header)
        
        # 3D View using PyQtGraph
        if PYQTGRAPH_AVAILABLE:
            self.view_widget = gl.GLViewWidget()
            self.view_widget.setMinimumHeight(400)
            
            # Set background color
            self.view_widget.setBackgroundColor('k')  # Black background
            
            # Configure camera view
            self.view_widget.setCameraPosition(distance=3.0, elevation=20, azimuth=45)
            
            # Add grid
            self.grid = gl.GLGridItem()
            self.grid.scale(self.grid_scale, self.grid_scale, self.grid_scale)
            self.view_widget.addItem(self.grid)
            
            # Add coordinate axes
            self._add_axes()
            
        else:
            # Fallback if PyQtGraph not available
            self.view_widget = QWidget()
            self.view_widget.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
            self.view_widget.setMinimumHeight(400)
            
            placeholder_layout = QVBoxLayout()
            placeholder_label = QLabel("3D Visualization\n(PyQtGraph not installed)")
            placeholder_label.setStyleSheet("color: #888; font-size: 14px;")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_layout.addWidget(placeholder_label)
            self.view_widget.setLayout(placeholder_layout)
        
        layout.addWidget(self.view_widget, stretch=1)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop_playback)
        self.stop_btn.setEnabled(False)
        
        self.frame_label = QLabel("Frame: 0 / 0")
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.frame_label)
        
        layout.addLayout(controls_layout)
        
        self.setLayout(layout)
    
    def _load_segment_lengths_from_settings(self):
        """Load segment lengths from app settings based on subject height"""
        try:
            # Get segment lengths from settings (in cm, converted to m)
            self.SEGMENT_LENGTHS['trunk'] = app_settings.get_segment_length('trunk') / 100.0
            self.SEGMENT_LENGTHS['thigh'] = app_settings.get_segment_length('thigh') / 100.0
            self.SEGMENT_LENGTHS['shank'] = app_settings.get_segment_length('shank') / 100.0
            self.SEGMENT_LENGTHS['foot'] = app_settings.get_segment_length('foot') / 100.0
            self.SEGMENT_LENGTHS['abdomen'] = app_settings.get_segment_length('abdomen') / 100.0
            self.SEGMENT_LENGTHS['chest'] = app_settings.get_segment_length('chest') / 100.0
            self.SEGMENT_LENGTHS['head'] = app_settings.get_segment_length('head') / 100.0
            self.SEGMENT_LENGTHS['upperarm'] = app_settings.get_segment_length('upperarm') / 100.0
            self.SEGMENT_LENGTHS['lowerarm'] = app_settings.get_segment_length('lowerarm') / 100.0
        except Exception as e:
            # Fallback to default values if settings not available
            print(f"Warning: Could not load segment lengths from settings: {e}")
            self.SEGMENT_LENGTHS = {
                'trunk': 0.50,
                'thigh': 0.40,
                'shank': 0.42,
                'foot': 0.25,
                'abdomen': 0.35,
                'chest': 0.25,
                'head': 0.20,
                'upperarm': 0.30,
                'lowerarm': 0.25
            }
    
    def update_segment_lengths(self, subject_info: dict):
        """
        Update segment lengths based on subject information.
        
        Args:
            subject_info: Dictionary containing height and segment ratios
        """
        try:
            height = subject_info.get('height', 170.0)  # cm
            
            # Get ratios
            trunk_ratio = subject_info.get('trunk_ratio', 0.288)
            thigh_ratio = subject_info.get('thigh_ratio', 0.232)
            shank_ratio = subject_info.get('shank_ratio', 0.246)
            foot_ratio = subject_info.get('foot_ratio', 0.152)
            # Upper body ratios
            abdomen_ratio = subject_info.get('abdomen_ratio', 0.190)
            chest_ratio = subject_info.get('chest_ratio', 0.150)
            head_ratio = subject_info.get('head_ratio', 0.100)
            upperarm_ratio = subject_info.get('upperarm_ratio', 0.186)
            lowerarm_ratio = subject_info.get('lowerarm_ratio', 0.146)
            
            # Calculate segment lengths (convert cm to m)
            self.SEGMENT_LENGTHS['trunk'] = (height * trunk_ratio) / 100.0
            self.SEGMENT_LENGTHS['thigh'] = (height * thigh_ratio) / 100.0
            self.SEGMENT_LENGTHS['shank'] = (height * shank_ratio) / 100.0
            self.SEGMENT_LENGTHS['foot'] = (height * foot_ratio) / 100.0
            self.SEGMENT_LENGTHS['abdomen'] = (height * abdomen_ratio) / 100.0
            self.SEGMENT_LENGTHS['chest'] = (height * chest_ratio) / 100.0
            self.SEGMENT_LENGTHS['head'] = (height * head_ratio) / 100.0
            self.SEGMENT_LENGTHS['upperarm'] = (height * upperarm_ratio) / 100.0
            self.SEGMENT_LENGTHS['lowerarm'] = (height * lowerarm_ratio) / 100.0
            
            # If data is loaded, re-render current frame with new lengths
            if self.current_data is not None:
                self._render_frame(self.current_frame)
                
        except Exception as e:
            print(f"Error updating segment lengths: {e}")
    
    def _add_axes(self):
        """Add coordinate axes to the 3D view"""
        if not PYQTGRAPH_AVAILABLE:
            return
        
        # X-axis (red)
        x_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0.5, 0, 0]]),
            color=(1, 0, 0, 1),
            width=2,
            antialias=True
        )
        self.view_widget.addItem(x_axis)
        
        # Y-axis (green)
        y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0.5, 0]]),
            color=(0, 1, 0, 1),
            width=2,
            antialias=True
        )
        self.view_widget.addItem(y_axis)
        
        # Z-axis (blue)
        z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 0.5]]),
            color=(0, 0, 1, 1),
            width=2,
            antialias=True
        )
        self.view_widget.addItem(z_axis)
    
    def set_data(self, motion_data):
        """Set motion capture data for visualization"""
        # Stop any ongoing playback first
        if self.is_playing:
            self._stop_playback()
        
        self.current_data = motion_data
        self.current_frame = 0
        
        if motion_data and motion_data.imu_data:
            # Get number of frames from first sensor
            first_sensor = next(iter(motion_data.imu_data.values()))
            n_frames = len(first_sensor.timestamps)
            
            self.frame_label.setText(f"Frame: 0 / {n_frames}")
            self.play_btn.setEnabled(True)
            
            # Initialize skeleton
            self._initialize_skeleton()
            self._render_frame(0)
            # Reset grid offset
            self.grid_offset = np.array([0.0, 0.0, 0.0])
            
            # Detect foot contact if not already detected
            # This allows grid movement even without calibration
            if motion_data.foot_contact_right is None or motion_data.foot_contact_left is None:
                self._auto_detect_foot_contact(motion_data)
        else:
            self.frame_label.setText("Frame: 0 / 0")
            self.play_btn.setEnabled(False)

    @pyqtSlot(str)
    def refresh_view_mode(self, mode_name: str):
        """ Refresh the 3D view when the mode changes (e.g., Lower-body to Upper-body) """
        if self.current_data:
            self._initialize_skeleton()
            self._render_frame(self.current_frame)
    
    def set_foot_contact(self, gait_start_frame: int, gait_end_frame: int, 
                        foot_contact_right: np.ndarray, foot_contact_left: np.ndarray):
        """
        Set foot contact data for grid movement.
        
        Args:
            gait_start_frame: Frame where gait starts (first foot leaves contact)
            gait_end_frame: Frame where gait ends (both feet return to contact)
            foot_contact_right: Boolean array indicating right foot contact
            foot_contact_left: Boolean array indicating left foot contact
        """
        self.gait_start_frame = gait_start_frame
        self.gait_end_frame = gait_end_frame
        self.foot_contact_right = foot_contact_right
        self.foot_contact_left = foot_contact_left
        
        # Determine reference feet for pre/post gait periods
        # Pre-gait: use the foot that will be in contact at gait_start_frame
        if gait_start_frame < len(foot_contact_right):
            if foot_contact_right[gait_start_frame]:
                self.reference_foot_pre_gait = 'right'
            elif foot_contact_left[gait_start_frame]:
                self.reference_foot_pre_gait = 'left'
        
        # Post-gait: use the foot that is in contact at gait_end_frame
        if gait_end_frame < len(foot_contact_right):
            if foot_contact_right[gait_end_frame]:
                self.reference_foot_post_gait = 'right'
            elif foot_contact_left[gait_end_frame]:
                self.reference_foot_post_gait = 'left'
    
    def _auto_detect_foot_contact(self, motion_data):
        """
        Automatically detect foot contact when data is not already processed.
        This allows grid movement to work even without calibration.
        
        Args:
            motion_data: MotionCaptureData object
        """
        try:
            from core.kinematics import KinematicsProcessor
            
            # Check if foot sensors are available
            if 'foot_right' not in motion_data.imu_data or 'foot_left' not in motion_data.imu_data:
                print("Warning: Foot sensors not found. Grid movement will not work.")
                return
            
            # Create kinematics processor and detect foot contact
            kinematics_processor = KinematicsProcessor()
            gait_start_frame, gait_end_frame, foot_contact_right, foot_contact_left = \
                kinematics_processor.detect_foot_contact(motion_data)
            
            # Set the detected foot contact data
            self.set_foot_contact(gait_start_frame, gait_end_frame, foot_contact_right, foot_contact_left)
            
            # Also update the motion_data so it's preserved
            motion_data.gait_start_frame = gait_start_frame
            motion_data.gait_end_frame = gait_end_frame
            motion_data.foot_contact_right = foot_contact_right
            motion_data.foot_contact_left = foot_contact_left
            
        except Exception as e:
            print(f"Error detecting foot contact: {e}")
            print("Grid movement will not work. Make sure foot sensors are available.")
    
    def _initialize_skeleton(self):
        """Initialize skeleton graphics items"""
        if not PYQTGRAPH_AVAILABLE:
            return
        
        # Remove old skeleton items
        for item in self.skeleton_items.values():
            self.view_widget.removeItem(item)
        for item in self.joint_items.values():
            self.view_widget.removeItem(item)
        for axes in self.axis_items.values():
            for axis in axes:
                self.view_widget.removeItem(axis)
        
        self.skeleton_items.clear()
        self.joint_items.clear()
        self.axis_items.clear()
        
        current_mode = app_settings.mode.mode_type
        if current_mode == 'Upper-body':
            segments = {
                'abdomen': (1.0, 1.0, 1.0, 1.0),        # White
                'chest': (0.8, 0.8, 0.8, 1.0),          # Gray
                'head': (0.6, 0.6, 0.6, 1.0),           # Light Gray
                'shoulder': (1.0, 1.0, 0.0, 1.0),       # Yellow (connects left and right shoulder)
                'upperarm_right': (1.0, 0.3, 0.3, 1.0),  # Red
                'upperarm_left': (0.3, 0.3, 1.0, 1.0),   # Blue
                'lowerarm_right': (1.0, 0.5, 0.5, 1.0),  # Light red
                'lowerarm_left': (0.5, 0.5, 1.0, 1.0),   # Light blue
            }
            joint_names = ['hip', 'spine', 'neck', 'rshoulder', 'lshoulder', 'elbow_right', 'elbow_left', 'wrist_right', 'wrist_left']
            segment_names = ['pelvis', 'chest', 'head', 'upperarm_right', 'upperarm_left', 'lowerarm_right', 'lowerarm_left']
        else:
            # Define segments and their colors
            segments = {
                'trunk': (1.0, 1.0, 1.0, 1.0),      # White
                'pelvis': (0.8, 0.8, 0.8, 1.0),     # Gray (connects left and right hip)
                'thigh_right': (1.0, 0.3, 0.3, 1.0),  # Red
                'thigh_left': (0.3, 0.3, 1.0, 1.0),   # Blue
                'shank_right': (1.0, 0.5, 0.5, 1.0),  # Light red
                'shank_left': (0.5, 0.5, 1.0, 1.0),   # Light blue
                'foot_right': (1.0, 0.7, 0.7, 1.0),   # Lighter red
                'foot_left': (0.7, 0.7, 1.0, 1.0),    # Lighter blue
            }
            joint_names = ['hip', 'rhip', 'lhip', 'knee_right', 'knee_left', 'ankle_right', 'ankle_left', 'toe_right', 'toe_left']
            segment_names = ['trunk', 'thigh_right', 'thigh_left', 'shank_right', 'shank_left', 'foot_right', 'foot_left']

        # Create line items for each segment
        for segment_name, color in segments.items():
            line_item = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 0, 0]]),
                color=color,
                width=4,
                antialias=True
            )
            self.view_widget.addItem(line_item)
            self.skeleton_items[segment_name] = line_item
        
        # Create joint spheres (including rhip, lhip)
        for joint_name in joint_names:
            mesh = gl.MeshData.sphere(rows=10, cols=10, radius=0.03)
            joint_item = gl.GLMeshItem(
                meshdata=mesh,
                smooth=True,
                color=(1, 1, 0, 1),  # Yellow
                glOptions='translucent'
            )
            self.view_widget.addItem(joint_item)
            self.joint_items[joint_name] = joint_item
        
        # Create coordinate axes for each segment
        for segment_name in segment_names:
            # X-axis (red)
            x_axis = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 0, 0]]),
                color=(1, 0, 0, 1),
                width=2,
                antialias=True
            )
            # Y-axis (green)
            y_axis = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 0, 0]]),
                color=(0, 1, 0, 1),
                width=2,
                antialias=True
            )
            # Z-axis (blue)
            z_axis = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 0, 0]]),
                color=(0, 0, 1, 1),
                width=2,
                antialias=True
            )
            
            self.view_widget.addItem(x_axis)
            self.view_widget.addItem(y_axis)
            self.view_widget.addItem(z_axis)
            
            self.axis_items[segment_name] = [x_axis, y_axis, z_axis]
    
    def _render_frame(self, frame_index: int):
        """
        Render specific frame by updating skeleton pose.
        
        Uses forward kinematics:
        1. Start from hip position (trunk base)
        2. Apply quaternion rotations to get segment orientations
        3. Calculate joint positions along the kinematic chain
        """
        if not self.current_data or not PYQTGRAPH_AVAILABLE:
            return
        
        if not self.current_data.imu_data:
            return
        
        # Clamp frame index to valid range
        first_sensor = next(iter(self.current_data.imu_data.values()))
        n_frames = len(first_sensor.timestamps)
        frame_index = max(0, min(frame_index, n_frames - 1))
        
        self.current_frame = frame_index
        
        # Update frame info
        self.frame_label.setText(f"Frame: {frame_index} / {n_frames}")
        
        # Calculate joint positions using forward kinematics
        positions = self._calculate_joint_positions(frame_index)
        
        # Update skeleton visualization
        self._update_skeleton(positions)
        
        # Update grid position based on foot contact
        self._update_grid_position(frame_index, positions)
        
        # Emit signal AFTER all updates complete (only if not updating from external source)
        if not self._updating:
            self.frame_changed.emit(frame_index)
    
    def _calculate_joint_positions(self, frame_index: int) -> dict:
        """
        Calculate 3D positions of all joints using forward kinematics.
        
        Global coordinate system: X=forward, Y=left, Z=up
        
        UNIFIED IMU attachment directions (sensor local frame):
        All segments now use the same coordinate system after calibration:
        - trunk:       x-up, y-left, z-backward
        - thigh/shank: x-up, y-left, z-backward
        - foot:        x-up, y-left, z-backward
        
        After N-pose calibration with unified q_desired, the calibrated quaternion
        represents the rotation FROM sensor's local frame TO global frame.
        
        So to get segment direction in global frame:
        - trunk: local +X (up) → apply q_trunk
        - thigh: local -X (down, since x-up and leg goes down) → apply q_thigh
        - shank: local -X (down) → apply q_shank
        - foot:  local -X (down to forward, following leg direction) → apply q_foot
        """
        positions = {}
        current_mode = app_settings.mode.mode_type
        
        hip_pos = np.array([0.0, 0.0, 1.0])
        positions['hip'] = hip_pos
        
        def get_quaternion(segment_name: str) -> np.ndarray:
            if segment_name in self.current_data.imu_data:
                q = self.current_data.imu_data[segment_name].quaternions[frame_index]
                return q / np.linalg.norm(q)
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
            """Apply quaternion rotation to a vector."""
            w, x, y, z = q
            R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
                [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
                [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ])
            return R @ v
        
        if current_mode == 'Upper-body':
            
            q_pelvis = get_quaternion('pelvis')
            q_chest = get_quaternion('chest')
            q_head = get_quaternion('head')
            q_upperarm_r = get_quaternion('upperarm_right')
            q_upperarm_l = get_quaternion('upperarm_left')
            q_lowerarm_r = get_quaternion('lowerarm_right')
            q_lowerarm_l = get_quaternion('lowerarm_left')
            
            # ============================================================
            # Segment direction vectors in IMU LOCAL coordinates
            # These represent the segment's longitudinal axis in sensor frame
            # ============================================================
            
            # trunk: IMU x-axis points UP along trunk → local +X is segment direction
            abdomen_local_dir = np.array([self.SEGMENT_LENGTHS['abdomen'], 0.0, 0.0])
            chest_local_dir = np.array([self.SEGMENT_LENGTHS['chest'], 0.0, 0.0])
            head_local_dir = np.array([self.SEGMENT_LENGTHS['head'], 0.0, 0.0])
            
            # upperarm/lowerarm: IMU x-axis points UP, but arm points DOWN → local -X is segment direction
            upperarm_r_local_dir = np.array([-self.SEGMENT_LENGTHS['upperarm'], 0.0, 0.0])
            upperarm_l_local_dir = np.array([-self.SEGMENT_LENGTHS['upperarm'], 0.0, 0.0])
            lowerarm_r_local_dir = np.array([-self.SEGMENT_LENGTHS['lowerarm'], 0.0, 0.0])
            lowerarm_l_local_dir = np.array([-self.SEGMENT_LENGTHS['lowerarm'], 0.0, 0.0])
            
            # Shoulder offsets in chest's local frame (y-right means -Y is left)
            # chest: y-right, so right shoulder offset is local -Y, left shoulder offset is local +Y
            rshoulder_local_offset = np.array([0.0, 0.15, 0.0])
            lshoulder_local_offset = np.array([0.0, -0.15, 0.0])
            
            # ============================================================
            # Forward Kinematics: rotate local directions to global frame
            # ============================================================

            # Abdomen (pelvis to chest)
            abdomen_dir = rotate_vector(abdomen_local_dir, q_pelvis)
            spine_pos = hip_pos + abdomen_dir
            positions['spine'] = spine_pos

            # Chest
            chest_dir = rotate_vector(chest_local_dir, q_chest)
            neck_pos = spine_pos + chest_dir
            positions['neck'] = neck_pos

            # Head (head height)
            head_dir = rotate_vector(head_local_dir, q_head)
            head_top = neck_pos + head_dir
            positions['head'] = head_top

            # Shoulders (offset from chest using chest orientation)
            rshoulder_offset = rotate_vector(rshoulder_local_offset, q_chest)
            lshoulder_offset = rotate_vector(lshoulder_local_offset, q_chest)
            rshoulder_pos = neck_pos + rshoulder_offset
            lshoulder_pos = neck_pos + lshoulder_offset
            positions['rshoulder'] = rshoulder_pos
            positions['lshoulder'] = lshoulder_pos

            # Right arm chain
            upperarm_r_dir = rotate_vector(upperarm_r_local_dir, q_upperarm_r)
            elbow_r_pos = rshoulder_pos + upperarm_r_dir
            positions['elbow_right'] = elbow_r_pos
            
            lowerarm_r_dir = rotate_vector(lowerarm_r_local_dir, q_lowerarm_r)
            wrist_r_pos = elbow_r_pos + lowerarm_r_dir
            positions['wrist_right'] = wrist_r_pos
            
            # Left arm chain
            upperarm_l_dir = rotate_vector(upperarm_l_local_dir, q_upperarm_l)
            elbow_l_pos = lshoulder_pos + upperarm_l_dir
            positions['elbow_left'] = elbow_l_pos
            
            lowerarm_l_dir = rotate_vector(lowerarm_l_local_dir, q_lowerarm_l)
            wrist_l_pos = elbow_l_pos + lowerarm_l_dir
            positions['wrist_left'] = wrist_l_pos

        else:
            q_trunk = get_quaternion('trunk')
            q_thigh_r = get_quaternion('thigh_right')
            q_thigh_l = get_quaternion('thigh_left')
            q_shank_r = get_quaternion('shank_right')
            q_shank_l = get_quaternion('shank_left')
            q_foot_r = get_quaternion('foot_right')
            q_foot_l = get_quaternion('foot_left')
            
            
            # ============================================================
            # Segment direction vectors in IMU LOCAL coordinates
            # These represent the segment's longitudinal axis in sensor frame
            # All segments now have UNIFIED coordinate system (x-up, y-left, z-backward)
            # ============================================================
            
            # trunk: IMU x-axis points UP along trunk → local +X is segment direction
            trunk_local_dir = np.array([self.SEGMENT_LENGTHS['trunk'], 0.0, 0.0])
            
            # thigh/shank: IMU x-axis points UP, but leg points DOWN → local -X is segment direction
            thigh_local_dir = np.array([-self.SEGMENT_LENGTHS['thigh'], 0.0, 0.0])
            shank_local_dir = np.array([-self.SEGMENT_LENGTHS['shank'], 0.0, 0.0])
            
            # foot: IMU z-axis points BACKWARD, foot points FORWARD → local -Z is segment direction
            foot_local_dir = np.array([0.0, 0.0, -self.SEGMENT_LENGTHS['foot']])
            
            # Hip offsets in trunk's local frame
            # With unified coordinate system: y-left, so right hip is -Y, left hip is +Y
            rhip_local_offset = np.array([0.0, -0.15, 0.0])
            lhip_local_offset = np.array([0.0, 0.15, 0.0])
            
            # ============================================================
            # Forward Kinematics: rotate local directions to global frame
            # ============================================================
            
            # Trunk
            trunk_dir = rotate_vector(trunk_local_dir, q_trunk)
            trunk_top = hip_pos + trunk_dir
            positions['trunk_top'] = trunk_top
            
            # Hip joints (offset from pelvis center using trunk orientation)
            rhip_offset = rotate_vector(rhip_local_offset, q_trunk)
            lhip_offset = rotate_vector(lhip_local_offset, q_trunk)
            rhip_pos = hip_pos + rhip_offset
            lhip_pos = hip_pos + lhip_offset
            positions['rhip'] = rhip_pos
            positions['lhip'] = lhip_pos
            
            # Right leg chain
            thigh_r_dir = rotate_vector(thigh_local_dir, q_thigh_r)
            knee_r_pos = rhip_pos + thigh_r_dir
            positions['knee_right'] = knee_r_pos
            
            shank_r_dir = rotate_vector(shank_local_dir, q_shank_r)
            ankle_r_pos = knee_r_pos + shank_r_dir
            positions['ankle_right'] = ankle_r_pos
            
            foot_r_dir = rotate_vector(foot_local_dir, q_foot_r)
            toe_r_pos = ankle_r_pos + foot_r_dir
            positions['toe_right'] = toe_r_pos
            
            # Left leg chain
            thigh_l_dir = rotate_vector(thigh_local_dir, q_thigh_l)
            knee_l_pos = lhip_pos + thigh_l_dir
            positions['knee_left'] = knee_l_pos
            
            shank_l_dir = rotate_vector(shank_local_dir, q_shank_l)
            ankle_l_pos = knee_l_pos + shank_l_dir
            positions['ankle_left'] = ankle_l_pos
            
            foot_l_dir = rotate_vector(foot_local_dir, q_foot_l)
            toe_l_pos = ankle_l_pos + foot_l_dir
            positions['toe_left'] = toe_l_pos
        
        return positions
        
    def _update_skeleton(self, positions: dict):
        """Update skeleton graphics with new joint positions"""
        if not PYQTGRAPH_AVAILABLE:
            return
        
        current_mode = app_settings.mode.mode_type
        if current_mode == 'Upper-body':
            # Update segment lines
            segments_to_draw = [
                ('abdomen', positions['hip'], positions['spine']),
                ('chest', positions['spine'], positions['neck']),
                ('head', positions['neck'], positions['head']),
                ('shoulder', positions['rshoulder'], positions['lshoulder']),  # Shoulder connects left and right shoulder
                ('upperarm_right', positions['rshoulder'], positions['elbow_right']),
                ('upperarm_left', positions['lshoulder'], positions['elbow_left']),
                ('lowerarm_right', positions['elbow_right'], positions['wrist_right']),
                ('lowerarm_left', positions['elbow_left'], positions['wrist_left']),
            ]
        else:
            # Update segment lines
            segments_to_draw = [
                ('trunk', positions['hip'], positions['trunk_top']),
                ('pelvis', positions['lhip'], positions['rhip']),  # Pelvis connects left and right hip
                ('thigh_right', positions['rhip'], positions['knee_right']),
                ('thigh_left', positions['lhip'], positions['knee_left']),
                ('shank_right', positions['knee_right'], positions['ankle_right']),
                ('shank_left', positions['knee_left'], positions['ankle_left']),
                ('foot_right', positions['ankle_right'], positions['toe_right']),
                ('foot_left', positions['ankle_left'], positions['toe_left']),
            ]
        
        for segment_name, start_pos, end_pos in segments_to_draw:
            if segment_name in self.skeleton_items:
                line_data = np.array([start_pos, end_pos])
                self.skeleton_items[segment_name].setData(pos=line_data)
        
        # Update joint spheres
        for joint_name, position in positions.items():
            if joint_name in self.joint_items:
                self.joint_items[joint_name].resetTransform()
                self.joint_items[joint_name].translate(position[0], position[1], position[2])
        
        # Update segment coordinate axes
        self._update_segment_axes(positions)
    
    def _update_segment_axes(self, positions: dict):
        """Update coordinate axes for each segment based on their orientations."""
        if not PYQTGRAPH_AVAILABLE or not self.current_data:
            return
        
        def get_quaternion(segment_name: str) -> np.ndarray:
            if segment_name in self.current_data.imu_data:
                q = self.current_data.imu_data[segment_name].quaternions[self.current_frame]
                return q / np.linalg.norm(q)
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
            w, x, y, z = q
            R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
                [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
                [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ])
            return R @ v
        
        axis_length = 0.1
        axis_x_npose = np.array([axis_length, 0, 0])
        axis_y_npose = np.array([0, axis_length, 0])
        axis_z_npose = np.array([0, 0, axis_length])
        
        current_mode = app_settings.mode.mode_type
        # Define segment center positions and their quaternions
        if current_mode == 'Upper-body':
            segment_configs = [
                ('pelvis', (positions['hip'] + positions['spine']) / 2),
                ('chest', (positions['spine'] + positions['neck']) / 2),
                ('head', (positions['neck'] + positions['head']) / 2),
                ('upperarm_right', (positions['rshoulder'] + positions['elbow_right']) / 2),
                ('upperarm_left', (positions['lshoulder'] + positions['elbow_left']) / 2),
                ('lowerarm_right', (positions['elbow_right'] + positions['wrist_right']) / 2),
                ('lowerarm_left', (positions['elbow_left'] + positions['wrist_left']) / 2),
            ]
        else:
            segment_configs = [
                ('trunk', (positions['hip'] + positions['trunk_top']) / 2),
                ('thigh_right', (positions['rhip'] + positions['knee_right']) / 2),
                ('thigh_left', (positions['lhip'] + positions['knee_left']) / 2),
                ('shank_right', (positions['knee_right'] + positions['ankle_right']) / 2),
                ('shank_left', (positions['knee_left'] + positions['ankle_left']) / 2),
                ('foot_right', (positions['ankle_right'] + positions['toe_right']) / 2),
                ('foot_left', (positions['ankle_left'] + positions['toe_left']) / 2),
            ]   
        
        for segment_name, center_pos in segment_configs:
            if segment_name not in self.axis_items:
                continue
            
            q = get_quaternion(segment_name)
            
            # Rotate N-pose axes by segment's quaternion
            # This shows where the segment's local axes point in global space
            global_x = rotate_vector(axis_x_npose, q)
            global_y = rotate_vector(axis_y_npose, q)
            global_z = rotate_vector(axis_z_npose, q)
            
            # Update axis line items
            x_axis, y_axis, z_axis = self.axis_items[segment_name]
            x_axis.setData(pos=np.array([center_pos, center_pos + global_x]))
            y_axis.setData(pos=np.array([center_pos, center_pos + global_y]))
            z_axis.setData(pos=np.array([center_pos, center_pos + global_z]))
    
    def _update_grid_position(self, frame_index: int, positions: dict):
        """
        Update grid position to follow foot contact during gait.
        
        Logic:
        - Grid follows the displacement (change in position) of the tracked foot
        - When switching feet, the new foot becomes the tracking target
        - Grid position changes based on relative movement, not absolute position
        
        Grid follows X,Y coordinates only (height Z remains constant)
        Reference point: center of foot segment = (toe + ankle) / 2
        """
        if self.grid is None or self.foot_contact_right is None or self.foot_contact_left is None:
            return
        
        if self.gait_start_frame is None or self.gait_end_frame is None:
            return
        
        current_mode = app_settings.mode.mode_type
        if current_mode == 'Upper-body':
            # Grid movement not applicable in Upper-body mode
            return
        
        # Get foot center positions
        foot_right_center = (positions['toe_right'] + positions['ankle_right']) / 2
        foot_left_center = (positions['toe_left'] + positions['ankle_left']) / 2
        
        # Determine which foot to track
        tracked_foot = None
        tracked_position = None
        
        if frame_index < self.gait_start_frame:
            # Before gait starts: track the pre-gait reference foot
            tracked_foot = self.reference_foot_pre_gait
            if tracked_foot == 'right':
                tracked_position = foot_right_center
            else:
                tracked_position = foot_left_center
        elif frame_index <= self.gait_end_frame:
            # During gait: track the foot currently in contact
            if self.foot_contact_right[frame_index]:
                tracked_foot = 'right'
                tracked_position = foot_right_center
            elif self.foot_contact_left[frame_index]:
                tracked_foot = 'left'
                tracked_position = foot_left_center
        else:
            # After gait ends: track the post-gait reference foot
            tracked_foot = self.reference_foot_post_gait
            if tracked_foot == 'right':
                tracked_position = foot_right_center
            else:
                tracked_position = foot_left_center
        
        # If no valid foot to track, do nothing
        if tracked_foot is None or tracked_position is None:
            return
        
        # Initialize prev_position if not set (first frame or after switch)
        if not hasattr(self, '_prev_tracked_foot') or self._prev_tracked_foot != tracked_foot:
            # Foot switched or first initialization
            self._prev_tracked_foot = tracked_foot
            self._prev_foot_position = tracked_position.copy()
            # Snap to nearest grid boundary while keeping visual pattern seamless
            # Store the visual grid position (using modulo)
            visual_x = self.grid_offset[0] % self.grid_wrap_period
            visual_y = self.grid_offset[1] % self.grid_wrap_period
            # Update grid to this position
            self.grid.resetTransform()
            self.grid.scale(self.grid_scale, self.grid_scale, self.grid_scale)
            self.grid.translate(visual_x, visual_y, self.grid_offset[2])
            # Update offset to match the visual position (keeps it small, near origin)
            self.grid_offset = np.array([visual_x, visual_y, self.grid_offset[2]])
            # Don't update grid on switch, just set baseline for next calculation
            return
        
        # Calculate displacement of the tracked foot
        displacement = tracked_position - self._prev_foot_position
        
        # Apply displacement to grid (only X, Y components)
        self.grid_offset[0] += displacement[0]
        self.grid_offset[1] += displacement[1]
        
        # Update previous position for next frame
        self._prev_foot_position = tracked_position.copy()
        
        # Apply wrapping to keep grid visible (treadmill effect)
        # Reset offset when it exceeds threshold, using modulo for seamless grid pattern
        grid_translate_x = self.grid_offset[0]
        grid_translate_y = self.grid_offset[1]
        
        # Wrap offset to keep grid in view and maintain seamless pattern
        if abs(self.grid_offset[0]) > self.grid_wrap_threshold:
            self.grid_offset[0] = self.grid_offset[0] % self.grid_wrap_period
            self.grid_offset[1] = self.grid_offset[1] % self.grid_wrap_period  # Also reset Y axis
        
        if abs(self.grid_offset[1]) > self.grid_wrap_threshold:
            self.grid_offset[0] = self.grid_offset[0] % self.grid_wrap_period  # Also reset X axis
            self.grid_offset[1] = self.grid_offset[1] % self.grid_wrap_period
        
        grid_translate_x = self.grid_offset[0]
        grid_translate_y = self.grid_offset[1]
        
        # Apply offset to grid (restore scale and apply translation)
        self.grid.resetTransform()
        self.grid.scale(self.grid_scale, self.grid_scale, self.grid_scale)
        self.grid.translate(grid_translate_x, grid_translate_y, self.grid_offset[2])
    
    def _toggle_playback(self):
        """Toggle play/pause"""
        if self.is_playing:
            self._pause_playback()
        else:
            self._start_playback()
    
    def _start_playback(self):
        """Start playback"""
        self.is_playing = True
        self.play_btn.setText("⏸ Pause")
        self.stop_btn.setEnabled(True)
        
        # Initialize playback timing
        import time
        self.playback_start_time = time.time()
        self.playback_start_frame = self.current_frame
        
        # Get sampling rate from data
        if self.current_data and self.current_data.imu_data:
            first_sensor = next(iter(self.current_data.imu_data.values()))
            timestamps = first_sensor.timestamps
            if len(timestamps) > 1:
                # Calculate actual sampling rate
                self.sampling_rate = 1.0 / np.mean(np.diff(timestamps))
        
        # Fixed display refresh rate: 50 FPS (20ms interval)
        # This provides smooth visualization regardless of data sampling rate
        self.timer.start(20)  # 50 Hz display refresh
    
    def _pause_playback(self):
        """Pause playback"""
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        self.timer.stop()
    
    def _stop_playback(self):
        """Stop playback and reset"""
        self._pause_playback()
        self.current_frame = 0
        self._render_frame(0)
        self.stop_btn.setEnabled(False)
    
    def _advance_frame(self):
        """Advance to next frame during playback using time-based frame calculation."""
        if not self.current_data or not self.current_data.imu_data:
            return
        
        first_sensor = next(iter(self.current_data.imu_data.values()))
        n_frames = len(first_sensor.timestamps)
        
        # Calculate elapsed real time
        import time
        elapsed_real_time = time.time() - self.playback_start_time
        
        # Calculate elapsed data time (accelerated by playback_speed)
        elapsed_data_time = elapsed_real_time * self.playback_speed
        
        # Convert data time to frame index
        # frame_index = start_frame + (elapsed_data_time * sampling_rate)
        target_frame = self.playback_start_frame + int(elapsed_data_time * self.sampling_rate)
        
        # Check if reached end
        if target_frame >= n_frames:
            # Loop back to beginning
            self.playback_start_time = time.time()
            self.playback_start_frame = 0
            target_frame = 0
        
        # Render the calculated frame (skips intermediate frames for smooth playback)
        self._render_frame(target_frame)
    
    def set_frame(self, frame_index: int):
        """Set specific frame to display (called externally, don't emit signal)"""
        self._updating = True
        self._render_frame(frame_index)
        self._updating = False
    
    def set_playback_speed(self, speed: float):
        """Set playback speed multiplier (1.0 = real-time, 0.5 = half speed, 2.0 = double speed)"""
        self.playback_speed = max(0.1, min(5.0, speed))
        
        # If currently playing, update timing to maintain continuity
        if self.is_playing:
            import time
            # Calculate where we should be now with old speed
            elapsed_real_time = time.time() - self.playback_start_time
            elapsed_data_time = elapsed_real_time * (self.playback_speed / speed)  # Reverse old speed
            
            # Reset start time for new speed
            self.playback_start_time = time.time()
            self.playback_start_frame = self.current_frame
    
    def clear(self):
        """Clear visualization"""
        self.current_data = None
        self.current_frame = 0
        self._stop_playback()
        self.frame_label.setText("Frame: 0 / 0")
        self.play_btn.setEnabled(False)
        
        # Remove skeleton items
        if PYQTGRAPH_AVAILABLE:
            for item in self.skeleton_items.values():
                self.view_widget.removeItem(item)
            for item in self.joint_items.values():
                self.view_widget.removeItem(item)
            
            self.skeleton_items.clear()
            self.joint_items.clear()

