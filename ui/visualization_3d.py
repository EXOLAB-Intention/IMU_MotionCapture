"""
3D visualization widget for displaying body segments
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
import numpy as np

try:
    from PyQt5.QtOpenGL import QGLWidget
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


class Visualization3D(QWidget):
    """3D visualization of body segments using IMU data"""
    
    # Signals
    frame_changed = pyqtSignal(int)  # current frame index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = None
        self.current_frame = 0
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._advance_frame)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("<b>3D Visualization</b>")
        layout.addWidget(header)
        
        # 3D View placeholder
        # TODO: Implement actual 3D rendering (OpenGL, PyQtGraph, or similar)
        self.view_widget = QWidget()
        self.view_widget.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        self.view_widget.setMinimumHeight(400)
        
        placeholder_layout = QVBoxLayout()
        placeholder_label = QLabel("3D Visualization\n(OpenGL rendering to be implemented)")
        placeholder_label.setStyleSheet("color: #888; font-size: 14px;")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_layout.addWidget(placeholder_label)
        self.view_widget.setLayout(placeholder_layout)
        
        layout.addWidget(self.view_widget, stretch=1)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_playback)
        self.stop_btn.setEnabled(False)
        
        self.frame_label = QLabel("Frame: 0 / 0")
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.frame_label)
        
        layout.addLayout(controls_layout)
        
        self.setLayout(layout)
    
    def set_data(self, motion_data):
        """Set motion capture data for visualization"""
        self.current_data = motion_data
        self.current_frame = 0
        
        if motion_data and motion_data.imu_data:
            # Get number of frames from first sensor
            first_sensor = next(iter(motion_data.imu_data.values()))
            n_frames = len(first_sensor.timestamps)
            self.frame_label.setText(f"Frame: 0 / {n_frames}")
            self.play_btn.setEnabled(True)
            self._render_frame(0)
        else:
            self.frame_label.setText("Frame: 0 / 0")
            self.play_btn.setEnabled(False)
    
    def _render_frame(self, frame_index: int):
        """Render specific frame"""
        if not self.current_data:
            return
        
        # TODO: Implement 3D rendering of body segments
        # This should:
        # 1. Extract quaternion orientations for each segment at frame_index
        # 2. Calculate segment positions based on joint angles
        # 3. Render segments as cylinders/boxes with correct orientations
        # 4. Render joints as spheres
        
        self.current_frame = frame_index
        
        if self.current_data.imu_data:
            first_sensor = next(iter(self.current_data.imu_data.values()))
            n_frames = len(first_sensor.timestamps)
            self.frame_label.setText(f"Frame: {frame_index} / {n_frames}")
            self.frame_changed.emit(frame_index)
    
    def _toggle_playback(self):
        """Toggle play/pause"""
        if self.is_playing:
            self._pause_playback()
        else:
            self._start_playback()
    
    def _start_playback(self):
        """Start playback"""
        self.is_playing = True
        self.play_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        
        # Start timer (60 FPS)
        self.timer.start(1000 // 60)
    
    def _pause_playback(self):
        """Pause playback"""
        self.is_playing = False
        self.play_btn.setText("Play")
        self.timer.stop()
    
    def _stop_playback(self):
        """Stop playback and reset"""
        self._pause_playback()
        self.current_frame = 0
        self._render_frame(0)
        self.stop_btn.setEnabled(False)
    
    def _advance_frame(self):
        """Advance to next frame during playback"""
        if not self.current_data or not self.current_data.imu_data:
            return
        
        first_sensor = next(iter(self.current_data.imu_data.values()))
        n_frames = len(first_sensor.timestamps)
        
        self.current_frame += 1
        if self.current_frame >= n_frames:
            self.current_frame = 0  # Loop
        
        self._render_frame(self.current_frame)
    
    def set_frame(self, frame_index: int):
        """Set specific frame to display"""
        self._render_frame(frame_index)
    
    def clear(self):
        """Clear visualization"""
        self.current_data = None
        self.current_frame = 0
        self._stop_playback()
        self.frame_label.setText("Frame: 0 / 0")
        self.play_btn.setEnabled(False)
