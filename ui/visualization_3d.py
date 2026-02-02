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
        # Upper body segments can be added similarly
        'spine': 0.50,     # Pelvis to chest
        'upperarm': 0.25,  # Shoulder to elbow
        'lowerarm': 0.25   # Elbow to wrist
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
        
        # Playback state
        self.playback_start_time = 0.0  # Real-time start (seconds)
        self.playback_start_frame = 0   # Frame index at start
        self.sampling_rate = 500.0      # Hz (will be updated from data)
        
        # 3D Graphics items
        self.skeleton_items = {}
        self.joint_items = {}
        self.axis_items = {}  # Segment coordinate axes
        
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
            grid = gl.GLGridItem()
            grid.scale(0.2, 0.2, 0.2)
            self.view_widget.addItem(grid)
            
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
        
        # Frame slider for manual scrubbing
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.frame_slider, stretch=1)
        
        layout.addLayout(slider_layout)
        
        self.setLayout(layout)
    
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
            
            # Update UI elements
            self.frame_slider.blockSignals(True)
            self.frame_slider.setMaximum(n_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            self.frame_slider.blockSignals(False)
            
            self.frame_label.setText(f"Frame: 0 / {n_frames}")
            self.play_btn.setEnabled(True)
            
            # Initialize skeleton
            self._initialize_skeleton()
            self._render_frame(0)
        else:
            self.frame_label.setText("Frame: 0 / 0")
            self.frame_slider.setEnabled(False)
            self.play_btn.setEnabled(False)

    @pyqtSlot(str)
    def refresh_view_mode(self, mode_name: str):
        """ Refresh the 3D view when the mode changes (e.g., Lower-body to Upper-body) """
        if self.current_data:
            self._initialize_skeleton()
            self._render_frame(self.current_frame)
    
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
                'spine': (1.0, 1.0, 1.0, 1.0),        # White
                'shoulder': (0.8, 0.8, 0.8, 1.0),     # Gray (connects left and right shoulder)
                'upperarm_right': (1.0, 0.3, 0.3, 1.0),  # Red
                'upperarm_left': (0.3, 0.3, 1.0, 1.0),   # Blue
                'lowerarm_right': (1.0, 0.5, 0.5, 1.0),  # Light red
                'lowerarm_left': (0.5, 0.5, 1.0, 1.0),   # Light blue
            }
            joint_names = ['shoulder', 'rshoulder', 'lshoulder', 'elbow_right', 'elbow_left', 'wrist_right', 'wrist_left']
            segment_names = ['spine', 'upperarm_right', 'upperarm_left', 'lowerarm_right', 'lowerarm_left']
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
        
        # Update slider without triggering valueChanged signal
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_index)
        self.frame_slider.blockSignals(False)
        
        # Calculate joint positions using forward kinematics
        positions = self._calculate_joint_positions(frame_index)
        
        # Update skeleton visualization
        self._update_skeleton(positions)
        
        # Emit signal AFTER all updates complete (only if not updating from external source)
        if not self._updating:
            self.frame_changed.emit(frame_index)
    
    def _calculate_joint_positions(self, frame_index: int) -> dict:
        """
        Calculate 3D positions of all joints using forward kinematics.
        
        Global coordinate system: X=forward, Y=left, Z=up
        
        IMU attachment directions (sensor local frame):
        - trunk:       x-up, y-right, z-forward
        - thigh/shank: x-up, y-left,  z-backward
        - foot:        x-backward, y-left, z-down
        
        After N-pose calibration with qD=[1,0,0,0], the calibrated quaternion
        represents the rotation FROM sensor's local frame TO global frame.
        
        So to get segment direction in global frame:
        - trunk: local +X (up) → apply q_trunk
        - thigh: local -X (down, since x-up and leg goes down) → apply q_thigh
        - shank: local -X (down) → apply q_shank
        - foot:  local -Z (forward, since z-down and foot points forward) → apply q_foot
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
            q_upperarm_r = get_quaternion('upperarm_right')
            q_upperarm_l = get_quaternion('upperarm_left')
            q_lowerarm_r = get_quaternion('lowerarm_right')
            q_lowerarm_l = get_quaternion('lowerarm_left')
            
            # ============================================================
            # Segment direction vectors in IMU LOCAL coordinates
            # These represent the segment's longitudinal axis in sensor frame
            # ============================================================
            
            # spine: IMU x-axis points DOWN along spine → local -X is segment direction
            spine_local_dir = np.array([-self.SEGMENT_LENGTHS['spine'], 0.0, 0.0])
            
            # upperarm/lowerarm: IMU x-axis points UP, but arm points DOWN → local -X is segment direction
            upperarm_r_local_dir = np.array([0.0, self.SEGMENT_LENGTHS['upperarm'], 0.0])
            upperarm_l_local_dir = np.array([0.0, -self.SEGMENT_LENGTHS['upperarm'], 0.0])
            lowerarm_r_local_dir = np.array([0.0, self.SEGMENT_LENGTHS['lowerarm'], 0.0])
            lowerarm_l_local_dir = np.array([0.0, -self.SEGMENT_LENGTHS['lowerarm'], 0.0])
            
            # Shoulder offsets in spine's local frame (y-right means -Y is left)
            # spine: y-right, so right shoulder offset is local -Y, left shoulder offset is local +Y
            rshoulder_local_offset = np.array([0.0, 0.15, 0.0])
            lshoulder_local_offset = np.array([0.0, -0.15, 0.0])
            
            # ============================================================
            # Forward Kinematics: rotate local directions to global frame
            # ============================================================

            # Spine
            spine_dir = rotate_vector(spine_local_dir, q_pelvis)
            chest_pos = hip_pos + spine_dir
            positions['chest'] = chest_pos

            # Shoulders (offset from chest using chest orientation)
            rshoulder_offset = rotate_vector(rshoulder_local_offset, q_chest)
            lshoulder_offset = rotate_vector(lshoulder_local_offset, q_chest)
            rshoulder_pos = chest_pos + rshoulder_offset
            lshoulder_pos = chest_pos + lshoulder_offset
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
            # ============================================================
            
            # trunk: IMU x-axis points UP along trunk → local +X is segment direction
            trunk_local_dir = np.array([self.SEGMENT_LENGTHS['trunk'], 0.0, 0.0])
            
            # thigh/shank: IMU x-axis points UP, but leg points DOWN → local -X is segment direction
            thigh_local_dir = np.array([-self.SEGMENT_LENGTHS['thigh'], 0.0, 0.0])
            shank_local_dir = np.array([-self.SEGMENT_LENGTHS['shank'], 0.0, 0.0])
            
            # foot: IMU z-axis points DOWN, foot points FORWARD → local -Z is segment direction
            # (Actually, if IMU x=backward, z=down, then forward is -X)
            foot_local_dir = np.array([-self.SEGMENT_LENGTHS['foot'], 0.0, 0.0])
            
            # Hip offsets in trunk's local frame (y-right means -Y is left)
            # trunk: y-right, so right hip offset is local -Y, left hip offset is local +Y
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
                ('spine', positions['hip'], positions['chest']),
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
                ('spine', (positions['hip'] + positions['chest']) / 2),
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
    
    def _on_slider_changed(self, value: int):
        """Handle frame slider changes (manual scrubbing)"""
        if not self.is_playing:  # Only respond if not playing
            self._render_frame(value)
    
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
        self.frame_slider.setEnabled(False)
        self.frame_slider.setValue(0)
        self.frame_slider.setMaximum(0)
        self.play_btn.setEnabled(False)
        
        # Remove skeleton items
        if PYQTGRAPH_AVAILABLE:
            for item in self.skeleton_items.values():
                self.view_widget.removeItem(item)
            for item in self.joint_items.values():
                self.view_widget.removeItem(item)
            
            self.skeleton_items.clear()
            self.joint_items.clear()

