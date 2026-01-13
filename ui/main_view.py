"""
Main view component combining 3D visualization, graph view, and controls
"""
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QPushButton, QLabel, QToolBar, QAction
)
from PyQt5.QtCore import Qt, pyqtSignal

from ui.visualization_3d import Visualization3D
from ui.graph_view import GraphView
from ui.time_bar import TimeBar


class MainView(QWidget):
    """Main view containing 3D visualization, graphs, and timeline"""
    
    # Signals
    process_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # Main splitter (horizontal)
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: 3D visualization
        self.visualization_3d = Visualization3D()
        self.main_splitter.addWidget(self.visualization_3d)
        
        # Right side: Graph view (initially hidden)
        self.graph_view = GraphView()
        self.graph_view.setVisible(False)
        self.main_splitter.addWidget(self.graph_view)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([600, 400])
        
        layout.addWidget(self.main_splitter, stretch=1)
        
        # Time bar at bottom
        self.time_bar = TimeBar()
        self.time_bar.time_changed.connect(self._on_time_changed)
        layout.addWidget(self.time_bar)
        
        self.setLayout(layout)
        
        # Connect signals
        self.visualization_3d.frame_changed.connect(self._on_frame_changed)
    
    def _create_toolbar(self) -> QToolBar:
        """Create toolbar with view options"""
        toolbar = QToolBar()
        
        # Split view toggle
        self.split_action = QAction("Split View", self)
        self.split_action.setCheckable(True)
        self.split_action.setChecked(False)
        self.split_action.triggered.connect(self._toggle_split_view)
        toolbar.addAction(self.split_action)
        
        toolbar.addSeparator()
        
        # Process button
        process_action = QAction("Process Data", self)
        process_action.triggered.connect(self._on_process)
        toolbar.addAction(process_action)
        
        toolbar.addSeparator()
        
        # Status label
        self.status_label = QLabel("No data loaded")
        toolbar.addWidget(self.status_label)
        
        return toolbar
    
    def _toggle_split_view(self, checked: bool):
        """Toggle split view mode"""
        self.graph_view.setVisible(checked)
        
        if checked:
            # Split view: show both
            self.main_splitter.setSizes([600, 400])
        else:
            # Single view: only 3D
            self.main_splitter.setSizes([1000, 0])
    
    def set_data(self, motion_data):
        """Set motion capture data for visualization"""
        self.current_data = motion_data
        
        # Update 3D visualization
        self.visualization_3d.set_data(motion_data)
        
        # Update graph view
        self.graph_view.set_data(motion_data)
        
        # Update time bar
        if motion_data and motion_data.imu_data:
            start_time, end_time = motion_data.get_time_range()
            duration = end_time - start_time
            
            # Get sampling frequency from first sensor
            first_sensor = next(iter(motion_data.imu_data.values()))
            self.time_bar.set_duration(duration, first_sensor.sampling_frequency)
            
            # Update status
            processed_status = "Processed" if motion_data.is_processed else "Not Processed"
            self.status_label.setText(
                f"Session: {motion_data.session_id} | "
                f"Duration: {duration:.2f}s | "
                f"Status: {processed_status}"
            )
        else:
            self.time_bar.reset()
            self.status_label.setText("No data loaded")
    
    def _on_time_changed(self, time: float):
        """Handle time bar change"""
        if not self.current_data or not self.current_data.imu_data:
            return
        
        # Convert time to frame index
        first_sensor = next(iter(self.current_data.imu_data.values()))
        timestamps = first_sensor.timestamps
        
        # Find closest frame
        frame_index = np.argmin(np.abs(timestamps - time))
        
        # Update 3D view
        self.visualization_3d.set_frame(frame_index)
    
    def _on_frame_changed(self, frame_index: int):
        """Handle frame change from 3D view"""
        if not self.current_data or not self.current_data.imu_data:
            return
        
        # Update time bar
        first_sensor = next(iter(self.current_data.imu_data.values()))
        time = first_sensor.timestamps[frame_index]
        self.time_bar.set_current_time(time)
    
    def _on_process(self):
        """Handle process button click"""
        self.process_requested.emit()
    
    def clear(self):
        """Clear all views"""
        self.visualization_3d.clear()
        self.graph_view.clear()
        self.time_bar.reset()
        self.status_label.setText("No data loaded")
        self.current_data = None
    
    def get_selected_time_range(self) -> tuple:
        """Get selected time range from time bar"""
        return self.time_bar.get_selected_range()
