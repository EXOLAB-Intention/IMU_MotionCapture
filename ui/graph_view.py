"""
Graph view widget for displaying time-series joint angles
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QCheckBox, QPushButton, QSplitter, QButtonGroup
)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class GraphView(QWidget):
    """Graph view for time-series data visualization"""
    
    # Signals
    time_selected = pyqtSignal(float)  # selected time in seconds
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = None
        self.view_mode = 'joint'
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Header and controls
        header_layout = QHBoxLayout()
        header_label = QLabel("<b>Joint Angles</b>")
        
        # View mode selection
        self.joint_view_btn = QPushButton("Joint View")
        self.joint_view_btn.setCheckable(True)
        self.joint_view_btn.setChecked(True)
        self.joint_view_btn.clicked.connect(lambda: self._set_view_mode('joint'))

        self.gait_view_btn = QPushButton("Gait View")
        self.gait_view_btn.setCheckable(True)
        self.gait_view_btn.clicked.connect(lambda: self._set_view_mode('gait'))

        view_group = QButtonGroup(self)
        view_group.setExclusive(True)
        view_group.addButton(self.joint_view_btn)
        view_group.addButton(self.gait_view_btn)

        # Joint selection
        self.joint_combo = QComboBox()
        self.joint_combo.addItems(['Hip', 'Knee', 'Ankle', 'Trunk'])
        self.joint_combo.currentTextChanged.connect(self._update_graph)
        
        # Side selection
        self.right_check = QCheckBox("Right")
        self.right_check.setChecked(True)
        self.right_check.stateChanged.connect(self._update_graph)
        
        self.left_check = QCheckBox("Left")
        self.left_check.setChecked(True)
        self.left_check.stateChanged.connect(self._update_graph)
        
        # Export button
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_graph)
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.joint_view_btn)
        header_layout.addWidget(self.gait_view_btn)
        header_layout.addSpacing(10)
        header_layout.addWidget(QLabel("Joint:"))
        header_layout.addWidget(self.joint_combo)
        header_layout.addWidget(self.right_check)
        header_layout.addWidget(self.left_check)
        header_layout.addWidget(export_btn)
        
        layout.addLayout(header_layout)
        
        # Graph canvas
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(8, 6))
            self.canvas = FigureCanvasQTAgg(self.figure)

            self.axes = []
            self._configure_axes()

            self.figure.tight_layout()
            layout.addWidget(self.canvas)
        else:
            placeholder = QLabel("Matplotlib not available\nInstall: pip install matplotlib")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; font-size: 12px;")
            layout.addWidget(placeholder)
        
        self.setLayout(layout)
    
    def set_data(self, motion_data):
        """Set motion capture data for visualization"""
        self.current_data = motion_data
        self._update_graph()
    
    def _configure_axes(self):
        """Configure axes based on current view mode."""
        self.figure.clear()
        self.axes = []

        if self.view_mode == 'joint':
            for i in range(3):
                ax = self.figure.add_subplot(3, 1, i + 1)
                self.axes.append(ax)
        else:
            for i in range(2):
                ax = self.figure.add_subplot(2, 1, i + 1)
                self.axes.append(ax)

    def _set_view_mode(self, mode: str):
        """Switch between joint and gait views."""
        if mode == self.view_mode:
            return
        self.view_mode = mode
        if MATPLOTLIB_AVAILABLE:
            self._configure_axes()
            self._update_graph()

    def _update_graph(self):
        """Update graph based on current selections"""
        if not MATPLOTLIB_AVAILABLE or not self.current_data:
            return

        if self.view_mode == 'joint':
            self._update_joint_view()
        else:
            self._update_gait_view()

        self.figure.tight_layout()
        self.canvas.draw()

    def _update_joint_view(self):
        """Render joint angle plots."""
        # Clear axes
        for ax in self.axes:
            ax.clear()

        joint = self.joint_combo.currentText().lower()

        if joint == 'trunk':
            self._update_trunk_view()
            return

        self.right_check.setEnabled(True)
        self.left_check.setEnabled(True)

        if not self.current_data.joint_angles:
            for ax in self.axes:
                ax.text(0.5, 0.5, 'No joint angle data', ha='center', va='center', transform=ax.transAxes)
                ax.grid(True, alpha=0.3)
            return

        timestamps = self.current_data.joint_angles.timestamps
        dof_labels = ['Flexion/Extension', 'Abduction/Adduction', 'Internal/External Rotation']

        if self.right_check.isChecked():
            angles_right = self.current_data.joint_angles.get_joint_angle(joint, 'right')
            if angles_right is not None:
                for i, ax in enumerate(self.axes):
                    ax.plot(timestamps, angles_right[:, i], 'r-', label='Right')

        if self.left_check.isChecked():
            angles_left = self.current_data.joint_angles.get_joint_angle(joint, 'left')
            if angles_left is not None:
                for i, ax in enumerate(self.axes):
                    ax.plot(timestamps, angles_left[:, i], 'b-', label='Left')

        for i, ax in enumerate(self.axes):
            ax.set_ylabel(f'{dof_labels[i]}\n(deg)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            if i < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (s)', fontsize=10)

        self.axes[0].set_title(f'{joint.capitalize()} Joint Angles', fontsize=12, fontweight='bold')

    def _update_trunk_view(self):
        """Render trunk orientation plots (yaw, roll, pitch)."""
        self.right_check.setEnabled(False)
        self.left_check.setEnabled(False)

        if not self.current_data.kinematics or self.current_data.kinematics.trunk_angle is None:
            for ax in self.axes:
                ax.text(0.5, 0.5, 'No trunk orientation data', ha='center', va='center', transform=ax.transAxes)
                ax.grid(True, alpha=0.3)
            return

        trunk_angles = self.current_data.kinematics.trunk_angle
        timestamps = self.current_data.kinematics.timestamps

        # Stored order is [roll, pitch, yaw]; display as yaw, roll, pitch
        series = [trunk_angles[:, 2], trunk_angles[:, 0], trunk_angles[:, 1]]
        labels = ['Yaw', 'Roll', 'Pitch']

        for i, ax in enumerate(self.axes):
            if len(series[i]) > 0:
                # Unwrap to avoid 0/180 deg flips, then zero at start
                unwrapped = np.degrees(np.unwrap(np.radians(series[i])))
                unwrapped = unwrapped - unwrapped[0]
                series[i] = unwrapped
            ax.plot(timestamps, series[i], 'k-', label='Trunk')
            ax.set_ylabel(f'{labels[i]}\n(deg)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            if len(series[i]) > 0:
                y_min = np.min(series[i])
                y_max = np.max(series[i])
                step = 360.0 if (y_max - y_min) >= 360.0 else 180.0
                tick_start = step * np.floor(y_min / step)
                tick_end = step * np.ceil(y_max / step)
                if tick_start == tick_end:
                    tick_start -= step
                    tick_end += step
                ticks = np.arange(tick_start, tick_end + step, step)
                ax.set_yticks(ticks)

            if i < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (s)', fontsize=10)

        self.axes[0].set_title('Trunk Orientation', fontsize=12, fontweight='bold')

    def _update_gait_view(self):
        """Render stride distance, gait speed, and foot contact plots."""
        for ax in self.axes:
            ax.clear()

        if not self.current_data.imu_data:
            for ax in self.axes:
                ax.text(0.5, 0.5, 'No data loaded', ha='center', va='center', transform=ax.transAxes)
                ax.grid(True, alpha=0.3)
            return

        first_sensor = next(iter(self.current_data.imu_data.values()))
        timestamps = first_sensor.timestamps

        stride_ax = self.axes[0]
        contact_ax = self.axes[1]

        stride_ax.set_ylabel('Stride\n(m)', fontsize=10)
        stride_ax.grid(True, alpha=0.3)

        contact_ax.set_ylabel('Foot Contact', fontsize=10)
        contact_ax.set_xlabel('Time (s)', fontsize=10)
        contact_ax.grid(True, alpha=0.3)
        contact_ax.set_ylim(-1.2, 1.2)
        contact_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        contact_ax.set_yticks([-1, 0, 1])
        contact_ax.set_yticklabels(['Right', '', 'Left'])

        stride_distances = []
        stride_transitions = []
        stride_sides = []

        from core.kinematics import KinematicsProcessor

        kinematics = KinematicsProcessor()
        try:
            _, _, foot_right, foot_left = kinematics.detect_foot_contact(self.current_data)
        except Exception:
            foot_right, foot_left = None, None

        if foot_right is not None and foot_left is not None:
            stride_distances, stride_transitions, stride_sides = kinematics.stride_distance(
                self.current_data,
                foot_right,
                foot_left,
                timestamps
            )

        if stride_distances and stride_transitions:
            stride_times = timestamps[stride_transitions]
            stride_times = np.asarray(stride_times)
            stride_distances = np.asarray(stride_distances)
            stride_sides = np.asarray(stride_sides)

            right_mask = stride_sides == 'right'
            left_mask = stride_sides == 'left'

            if np.any(right_mask):
                stride_ax.plot(
                    stride_times[right_mask],
                    stride_distances[right_mask],
                    'ro',
                    markersize=4,
                    label='Right'
                )

            if np.any(left_mask):
                stride_ax.plot(
                    stride_times[left_mask],
                    stride_distances[left_mask],
                    'bo',
                    markersize=4,
                    label='Left'
                )

            stride_ax.legend(loc='upper right')
        else:
            stride_ax.text(0.5, 0.5, 'No stride data', ha='center', va='center', transform=stride_ax.transAxes)

        # Plot foot contact using fill_between (like visualization.py)
        if foot_right is not None and foot_left is not None:
            foot_right = np.asarray(foot_right, dtype=bool)
            foot_left = np.asarray(foot_left, dtype=bool)
            
            # Align to timestamps length
            n_frames = len(timestamps)
            foot_right = foot_right[:n_frames]
            foot_left = foot_left[:n_frames]
            
            # Left foot: positive (0 to 1)
            if np.any(foot_left):
                contact_ax.fill_between(timestamps, 0, foot_left.astype(int), alpha=0.6, color='blue', label='Left Contact')
            
            # Right foot: negative (0 to -1)
            if np.any(foot_right):
                contact_ax.fill_between(timestamps, 0, -foot_right.astype(int), alpha=0.6, color='red', label='Right Contact')
        else:
            contact_ax.text(0.5, 0.5, 'No foot contact data', ha='center', va='center', transform=contact_ax.transAxes)
    
    def _export_graph(self):
        """Export graph to image file"""
        if not MATPLOTLIB_AVAILABLE or not self.current_data:
            return
        
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        # Determine default filename based on view mode
        if self.view_mode == 'joint':
            joint = self.joint_combo.currentText().lower()
            if joint == 'trunk':
                default_name = "trunk_orientation.png"
            else:
                default_name = f"{joint}_angles.png"
        else:
            default_name = "gait_analysis.png"
        
        # Get save file path
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Graph",
            default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Vector (*.svg);;PDF Document (*.pdf);;All Files (*)"
        )
        
        if file_path:
            try:
                # Save with high DPI for better quality
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
                QMessageBox.information(self, "Export Successful", f"Graph exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Error exporting graph:\n{str(e)}")
    
    def clear(self):
        """Clear graph"""
        if MATPLOTLIB_AVAILABLE:
            for ax in self.axes:
                ax.clear()
            self.canvas.draw()
        self.current_data = None
