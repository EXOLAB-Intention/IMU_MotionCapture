"""
Graph view widget for displaying time-series joint angles
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QCheckBox, QPushButton, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
import numpy as np
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
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Header and controls
        header_layout = QHBoxLayout()
        header_label = QLabel("<b>Joint Angles</b>")
        
        # Joint selection
        self.joint_combo = QComboBox()
        self.update_mode_selection()
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
            
            # Create subplots for 3 DOF
            self.axes = []
            for i in range(3):
                ax = self.figure.add_subplot(3, 1, i + 1)
                self.axes.append(ax)
            
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
            joints = ['Hip', 'Knee', 'Ankle']
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
    
    def _update_graph(self):
        """Update graph based on current selections"""
        if not MATPLOTLIB_AVAILABLE or not self.current_data:
            return
        
        if not self.current_data.joint_angles:
            return
        
        # Get selected joint
        joint = self.joint_combo.currentText().lower()
        
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        # Plot data
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
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _export_graph(self):
        """Export graph to image file"""
        # TODO: Implement graph export
        print("Export graph functionality to be implemented")
    
    def clear(self):
        """Clear graph"""
        if MATPLOTLIB_AVAILABLE:
            for ax in self.axes:
                ax.clear()
            self.canvas.draw()
        self.current_data = None
