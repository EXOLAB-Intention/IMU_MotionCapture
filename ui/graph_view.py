"""
Graph view widget for displaying time-series joint angles
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QCheckBox, QPushButton, QSplitter, QButtonGroup,
    QDialog, QMessageBox, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
import numpy as np
import pandas as pd
from config.settings import app_settings

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
        self.joint_combo.addItems(['Hip', 'Knee', 'Ankle', 'Back'])
        self.update_mode_selection()
        self.joint_combo.currentTextChanged.connect(self._update_graph)
        
        # Side selection
        self.right_check = QCheckBox("Right")
        self.right_check.setChecked(True)
        self.right_check.stateChanged.connect(self._update_graph)
        
        self.left_check = QCheckBox("Left")
        self.left_check.setChecked(True)
        self.left_check.stateChanged.connect(self._update_graph)
        
        # Export buttons
        export_graph_btn = QPushButton("Export Graph")
        export_graph_btn.clicked.connect(self._export_graph)
        
        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.clicked.connect(self._export_csv)
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.joint_view_btn)
        header_layout.addWidget(self.gait_view_btn)
        header_layout.addSpacing(10)
        header_layout.addWidget(QLabel("Joint:"))
        header_layout.addWidget(self.joint_combo)
        header_layout.addWidget(self.right_check)
        header_layout.addWidget(self.left_check)
        header_layout.addWidget(export_graph_btn)
        header_layout.addWidget(export_csv_btn)
        
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

    @pyqtSlot(str)
    def update_mode_selection(self, mode_name: str = None):
        """Update joint selection options based on current mode"""
        # Maintain current selection
        current_text = self.joint_combo.currentText()
        
        # Block signals to avoid triggering updates
        self.joint_combo.blockSignals(True)
        self.joint_combo.clear()

        # Check current mode
        current_mode = app_settings.mode.mode_type

        if current_mode == 'Upper-body':
            # Upper body
            joints = ['Spine', 'Neck', 'Shoulder', 'Elbow']
        else:
            # Lower body
            joints = ['Hip', 'Knee', 'Ankle', 'Back']
        self.joint_combo.addItems(joints)
        
        # Restore previous selection if still valid
        if current_text in joints:
            self.joint_combo.setCurrentText(current_text)
        
        self.joint_combo.blockSignals(False)
        
        # Update graph
        self._update_graph()
    
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

        if joint == 'back':
            self._update_back_view()
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
        
        # Check if graph has been plotted
        plotted_label = False

        # Plot for joints without sides
        if joint in ['spine', 'neck']:
            angles = self.current_data.joint_angles.get_joint_angle(joint, '')
            if angles is not None:
                for i, ax in enumerate(self.axes):
                    ax.plot(timestamps, angles[:, i], 'g-', linewidth=2)
        else:
            # Plot right side
            if self.right_check.isChecked():
                angles_right = self.current_data.joint_angles.get_joint_angle(joint, 'right')
                if angles_right is not None:
                    for i, ax in enumerate(self.axes):
                        ax.plot(timestamps, angles_right[:, i], 'r-', label='Right', linewidth=2)
                    plotted_label = True
            
            # Plot left side
            if self.left_check.isChecked():
                angles_left = self.current_data.joint_angles.get_joint_angle(joint, 'left')
                if angles_left is not None:
                    for i, ax in enumerate(self.axes):
                        ax.plot(timestamps, angles_left[:, i], 'b-', label='Left', linewidth=2)
                    plotted_label = True
        
        # Format axes
        for i, ax in enumerate(self.axes):
            ax.set_ylabel(f'{dof_labels[i]}\n(deg)', fontsize=10)
            ax.grid(True, alpha=0.3)
            if plotted_label:
                ax.legend(loc='upper right', fontsize=8)
            
            if i < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (s)', fontsize=10)

        self.axes[0].set_title(f'{joint.capitalize()} Joint Angles', fontsize=12, fontweight='bold')

    def _update_back_view(self):
        """Render back orientation plots (yaw, roll, pitch)."""
        self.right_check.setEnabled(False)
        self.left_check.setEnabled(False)

        if not self.current_data.kinematics or self.current_data.kinematics.back_angle is None:
            for ax in self.axes:
                ax.text(0.5, 0.5, 'No back orientation data', ha='center', va='center', transform=ax.transAxes)
                ax.grid(True, alpha=0.3)
            return

        back_angles = self.current_data.kinematics.back_angle
        timestamps = self.current_data.kinematics.timestamps

        # Stored order is [roll(x), pitch(y), yaw(z)] in local frame
        # Map local -> global: x -> z, y -> -y, z -> x
        # Display order: Yaw, Roll, Pitch (global frame)
        series = [back_angles[:, 0], back_angles[:, 2], -back_angles[:, 1]]
        labels = ['Yaw', 'Roll', 'Pitch']

        for i, ax in enumerate(self.axes):
            if len(series[i]) > 0:
                # Unwrap to avoid 0/180 deg flips, keep measured values
                series[i] = np.degrees(np.unwrap(np.radians(series[i])))
            ax.plot(timestamps, series[i], 'k-', label='Back')
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

        self.axes[0].set_title('Back Orientation', fontsize=12, fontweight='bold')

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
        
        # Determine default filename based on view mode
        if self.view_mode == 'joint':
            joint = self.joint_combo.currentText().lower()
            if joint == 'back':
                default_name = "back_orientation.png"
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
    
    def _export_csv(self):
        """Export selected data columns to CSV"""
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "No data loaded to export.")
            return
        
        # Show selection dialog
        dialog = ExportCSVDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        
        selected_items = dialog.get_selected_items()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select at least one item to export.")
            return
        
        try:
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export CSV",
                "motion_data.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Build DataFrame
            if self.current_data.kinematics and self.current_data.kinematics.timestamps is not None:
                timestamps = self.current_data.kinematics.timestamps
            else:
                timestamps = self.current_data.joint_angles.timestamps
            data_dict = {'Timestamp': timestamps}
            
            # Add selected items
            for item in selected_items:
                if item == 'R_Hip':
                    data_dict['R_Hip_Flexion'] = self.current_data.joint_angles.hip_right[:, 0]
                    data_dict['R_Hip_Abduction'] = self.current_data.joint_angles.hip_right[:, 1]
                    data_dict['R_Hip_Rotation'] = self.current_data.joint_angles.hip_right[:, 2]
                elif item == 'L_Hip':
                    data_dict['L_Hip_Flexion'] = self.current_data.joint_angles.hip_left[:, 0]
                    data_dict['L_Hip_Abduction'] = self.current_data.joint_angles.hip_left[:, 1]
                    data_dict['L_Hip_Rotation'] = self.current_data.joint_angles.hip_left[:, 2]
                elif item == 'R_Knee':
                    data_dict['R_Knee_Flexion'] = self.current_data.joint_angles.knee_right[:, 0]
                    data_dict['R_Knee_Abduction'] = self.current_data.joint_angles.knee_right[:, 1]
                    data_dict['R_Knee_Rotation'] = self.current_data.joint_angles.knee_right[:, 2]
                elif item == 'L_Knee':
                    data_dict['L_Knee_Flexion'] = self.current_data.joint_angles.knee_left[:, 0]
                    data_dict['L_Knee_Abduction'] = self.current_data.joint_angles.knee_left[:, 1]
                    data_dict['L_Knee_Rotation'] = self.current_data.joint_angles.knee_left[:, 2]
                elif item == 'R_Ankle':
                    data_dict['R_Ankle_Flexion'] = self.current_data.joint_angles.ankle_right[:, 0]
                    data_dict['R_Ankle_Abduction'] = self.current_data.joint_angles.ankle_right[:, 1]
                    data_dict['R_Ankle_Rotation'] = self.current_data.joint_angles.ankle_right[:, 2]
                elif item == 'L_Ankle':
                    data_dict['L_Ankle_Flexion'] = self.current_data.joint_angles.ankle_left[:, 0]
                    data_dict['L_Ankle_Abduction'] = self.current_data.joint_angles.ankle_left[:, 1]
                    data_dict['L_Ankle_Rotation'] = self.current_data.joint_angles.ankle_left[:, 2]
                elif item == 'Back':
                    # Unwrap back angles and map local -> global like in graph view
                    back_angles = self.current_data.kinematics.back_angle
                    roll_global = np.degrees(np.unwrap(np.radians(back_angles[:, 2])))
                    yaw_global = np.degrees(np.unwrap(np.radians(back_angles[:, 0])))
                    pitch_global = -np.degrees(np.unwrap(np.radians(back_angles[:, 1])))
                    data_dict['Back_Roll'] = roll_global
                    data_dict['Back_Yaw'] = yaw_global
                    data_dict['Back_Pitch'] = pitch_global
                elif item == 'Stride':
                    # Stride is sparse - only at transition points
                    stride_col = np.full(len(timestamps), np.nan)
                    if self.current_data.kinematics.stride_times_right:
                        for stride_time in self.current_data.kinematics.stride_times_right:
                            idx = np.argmin(np.abs(timestamps - stride_time))
                            stride_col[idx] = stride_time
                    data_dict['Stride_Right'] = stride_col
                elif item == 'R_FootContact':
                    data_dict['R_FootContact'] = self.current_data.kinematics.foot_contact_right.astype(int)
                elif item == 'L_FootContact':
                    data_dict['L_FootContact'] = self.current_data.kinematics.foot_contact_left.astype(int)
                elif item == 'Foot_Abs_Dorsiflexion':
                    right_thigh_y = self._compute_segment_local_y_rotation_from_initial('thigh_right')
                    left_thigh_y = self._compute_segment_local_y_rotation_from_initial('thigh_left')
                    right_shank_y = self._compute_segment_local_y_rotation_from_initial('shank_right')
                    left_shank_y = self._compute_segment_local_y_rotation_from_initial('shank_left')
                    right_local_y, left_local_y = self._compute_foot_local_y_rotation_from_initial()

                    if right_thigh_y is not None:
                        data_dict['R_Thigh_LocalY_Rotation'] = self._fit_series_length(right_thigh_y, len(timestamps))
                    if left_thigh_y is not None:
                        data_dict['L_Thigh_LocalY_Rotation'] = self._fit_series_length(left_thigh_y, len(timestamps))
                    if right_shank_y is not None:
                        data_dict['R_Shank_LocalY_Rotation'] = self._fit_series_length(right_shank_y, len(timestamps))
                    if left_shank_y is not None:
                        data_dict['L_Shank_LocalY_Rotation'] = self._fit_series_length(left_shank_y, len(timestamps))
                    if right_local_y is not None:
                        data_dict['R_Foot_LocalY_Rotation'] = self._fit_series_length(right_local_y, len(timestamps))
                    if left_local_y is not None:
                        data_dict['L_Foot_LocalY_Rotation'] = self._fit_series_length(left_local_y, len(timestamps))
            
            # Create DataFrame and save
            df = pd.DataFrame(data_dict)
            df.to_csv(file_path, index=False)
            
            QMessageBox.information(self, "Export Successful", f"Data exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting CSV:\n{str(e)}")

    @staticmethod
    def _fit_series_length(series: np.ndarray, target_len: int) -> np.ndarray:
        """Trim or NaN-pad a 1D series to target length."""
        series = np.asarray(series, dtype=float).reshape(-1)
        if len(series) == target_len:
            return series
        if len(series) > target_len:
            return series[:target_len]

        padded = np.full(target_len, np.nan, dtype=float)
        padded[:len(series)] = series
        return padded

    def _compute_foot_local_y_rotation_from_initial(self):
        """Compute foot local-y rotation from initial foot IMU orientation.

        q_rel = conj(q0) * q_t is represented in the initial local frame.
        We extract the twist component around local y-axis directly from q_rel,
        avoiding Euler-axis ambiguity.
        """
        if not self.current_data or not self.current_data.imu_data:
            return None, None

        return (
            self._compute_segment_local_y_rotation_from_initial('foot_right'),
            self._compute_segment_local_y_rotation_from_initial('foot_left')
        )

    def _compute_segment_local_y_rotation_from_initial(self, location: str):
        """Compute segment local-y rotation from initial quaternion for one sensor location."""
        if not self.current_data or not self.current_data.imu_data:
            return None

        from core.kinematics import KinematicsProcessor

        sensor = self.current_data.imu_data.get(location)
        if sensor is None or sensor.quaternions is None or len(sensor.quaternions) == 0:
            return None

        q_series = KinematicsProcessor.quaternion_normalize(np.asarray(sensor.quaternions, dtype=float))
        q0 = q_series[0]
        q_rel = KinematicsProcessor.compute_relative_quaternion(q0, q_series)

        # Twist angle around local y axis from q_rel = [w, x, y, z].
        q_rel = KinematicsProcessor.quaternion_normalize(q_rel)
        w = np.asarray(q_rel[:, 0], dtype=float)
        y = np.asarray(q_rel[:, 2], dtype=float)
        local_y_rad = 2.0 * np.arctan2(y, w)
        local_y_rad = np.unwrap(local_y_rad)
        local_y_deg = np.degrees(local_y_rad)

        if len(local_y_deg) > 0:
            local_y_deg = local_y_deg - local_y_deg[0]
            local_y_deg[np.abs(local_y_deg) < 1e-12] = 0.0

        return local_y_deg
    
    def clear(self):
        """Clear graph"""
        if MATPLOTLIB_AVAILABLE:
            for ax in self.axes:
                ax.clear()
            self.canvas.draw()
        self.current_data = None


class ExportCSVDialog(QDialog):
    """Dialog for selecting CSV export items"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export CSV - Select Items")
        self.setMinimumWidth(400)
        self.check_items = {}
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Joint angles section
        joint_group = QGroupBox("Joint Angles")
        joint_layout = QVBoxLayout()
        
        for joint in ['R_Hip', 'L_Hip', 'R_Knee', 'L_Knee', 'R_Ankle', 'L_Ankle']:
            cb = QCheckBox(joint)
            self.check_items[joint] = cb
            joint_layout.addWidget(cb)
        
        joint_group.setLayout(joint_layout)
        layout.addWidget(joint_group)
        
        # Back section
        back_group = QGroupBox("Back")
        back_layout = QVBoxLayout()
        
        back_cb = QCheckBox("Back (Yaw, Roll, Pitch)")
        self.check_items['Back'] = back_cb
        back_layout.addWidget(back_cb)
        
        back_group.setLayout(back_layout)
        layout.addWidget(back_group)
        
        # Gait section
        gait_group = QGroupBox("Gait")
        gait_layout = QVBoxLayout()
        
        stride_cb = QCheckBox("Stride Distance")
        self.check_items['Stride'] = stride_cb
        gait_layout.addWidget(stride_cb)
        
        for contact in ['R_FootContact', 'L_FootContact']:
            cb = QCheckBox(contact)
            self.check_items[contact] = cb
            gait_layout.addWidget(cb)
        
        gait_group.setLayout(gait_layout)
        layout.addWidget(gait_group)

        # Segment absolute section
        abs_group = QGroupBox("Segment Absolute")
        abs_layout = QVBoxLayout()

        foot_abs_cb = QCheckBox("Thigh/Shank/Foot Local-Y Rotation (green axis, from initial quaternion)")
        self.check_items['Foot_Abs_Dorsiflexion'] = foot_abs_cb
        abs_layout.addWidget(foot_abs_cb)

        abs_group.setLayout(abs_layout)
        layout.addWidget(abs_group)
        
        # Select All / Deselect All buttons
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        button_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        button_layout.addWidget(deselect_all_btn)
        
        layout.addLayout(button_layout)
        
        # Dialog buttons
        dialog_button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.accept)
        dialog_button_layout.addWidget(export_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        dialog_button_layout.addWidget(cancel_btn)
        
        layout.addLayout(dialog_button_layout)
        
        self.setLayout(layout)
    
    def _select_all(self):
        """Check all items"""
        for cb in self.check_items.values():
            cb.setChecked(True)
    
    def _deselect_all(self):
        """Uncheck all items"""
        for cb in self.check_items.values():
            cb.setChecked(False)
    
    def get_selected_items(self):
        """Get list of selected items"""
        return [name for name, cb in self.check_items.items() if cb.isChecked()]
