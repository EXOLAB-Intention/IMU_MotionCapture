"""
Time bar widget for selecting time ranges
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QSlider, QPushButton, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np


class TimeBar(QWidget):
    """Time bar for navigating and selecting time ranges"""
    
    # Signals
    time_changed = pyqtSignal(float)  # current time in seconds
    range_changed = pyqtSignal(float, float)  # start_time, end_time
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_duration = 0.0
        self.sampling_frequency = 1000.0
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Current time slider
        slider_layout = QHBoxLayout()
        
        slider_label = QLabel("Time:")
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(1000)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self._on_slider_changed)
        
        self.time_label = QLabel("0.000 s")
        self.time_label.setMinimumWidth(80)
        
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.time_slider, stretch=1)
        slider_layout.addWidget(self.time_label)
        
        layout.addLayout(slider_layout)
        
        # Range selection
        range_layout = QHBoxLayout()
        
        range_label = QLabel("<b>Selection Range:</b>")
        
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setSuffix(" s")
        self.start_spin.setMinimum(0.0)
        self.start_spin.setMaximum(1000.0)
        self.start_spin.setValue(0.0)
        self.start_spin.valueChanged.connect(self._on_range_changed)
        
        range_to_label = QLabel("to")
        
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setSuffix(" s")
        self.end_spin.setMinimum(0.0)
        self.end_spin.setMaximum(1000.0)
        self.end_spin.setValue(0.0)
        self.end_spin.valueChanged.connect(self._on_range_changed)
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        
        self.duration_label = QLabel("Duration: 0.000 s")
        
        range_layout.addWidget(range_label)
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.start_spin)
        range_layout.addWidget(range_to_label)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.end_spin)
        range_layout.addWidget(self.select_all_btn)
        range_layout.addStretch()
        range_layout.addWidget(self.duration_label)
        
        layout.addLayout(range_layout)
        
        # Playback rate
        playback_layout = QHBoxLayout()
        
        playback_label = QLabel("Playback Speed:")
        self.speed_combo = QSlider(Qt.Horizontal)
        self.speed_combo.setMinimum(1)
        self.speed_combo.setMaximum(10)
        self.speed_combo.setValue(5)
        self.speed_label = QLabel("1.0x")
        
        playback_layout.addWidget(playback_label)
        playback_layout.addWidget(self.speed_combo)
        playback_layout.addWidget(self.speed_label)
        playback_layout.addStretch()
        
        layout.addLayout(playback_layout)
        
        self.setLayout(layout)
        self.setMaximumHeight(150)
    
    def set_duration(self, duration: float, sampling_frequency: float = 1000.0):
        """Set the total duration and sampling frequency"""
        self.total_duration = duration
        self.sampling_frequency = sampling_frequency
        
        # Update slider
        n_samples = int(duration * sampling_frequency)
        self.time_slider.setMaximum(n_samples - 1)
        
        # Update spin boxes
        self.start_spin.setMaximum(duration)
        self.end_spin.setMaximum(duration)
        self.end_spin.setValue(duration)
        
        self._update_time_label()
        self._update_duration_label()
    
    def _on_slider_changed(self, value: int):
        """Handle slider value change"""
        if self.total_duration > 0:
            time = value / self.time_slider.maximum() * self.total_duration
            self._update_time_label()
            self.time_changed.emit(time)
    
    def _update_time_label(self):
        """Update current time label"""
        if self.total_duration > 0:
            value = self.time_slider.value()
            time = value / self.time_slider.maximum() * self.total_duration
            self.time_label.setText(f"{time:.3f} s")
        else:
            self.time_label.setText("0.000 s")
    
    def _on_range_changed(self):
        """Handle range selection change"""
        start = self.start_spin.value()
        end = self.end_spin.value()
        
        # Ensure start < end
        if start >= end:
            if self.sender() == self.start_spin:
                self.end_spin.setValue(start + 0.001)
            else:
                self.start_spin.setValue(end - 0.001)
            return
        
        self._update_duration_label()
        self.range_changed.emit(start, end)
    
    def _update_duration_label(self):
        """Update duration label"""
        duration = self.end_spin.value() - self.start_spin.value()
        self.duration_label.setText(f"Duration: {duration:.3f} s")
    
    def _select_all(self):
        """Select entire time range"""
        self.start_spin.setValue(0.0)
        self.end_spin.setValue(self.total_duration)
    
    def get_current_time(self) -> float:
        """Get current time from slider"""
        if self.total_duration > 0:
            value = self.time_slider.value()
            return value / self.time_slider.maximum() * self.total_duration
        return 0.0
    
    def get_selected_range(self) -> tuple:
        """Get selected time range"""
        return (self.start_spin.value(), self.end_spin.value())
    
    def set_current_time(self, time: float):
        """Set current time position"""
        if self.total_duration > 0:
            value = int(time / self.total_duration * self.time_slider.maximum())
            self.time_slider.setValue(value)
    
    def reset(self):
        """Reset time bar to initial state"""
        self.time_slider.setValue(0)
        self.start_spin.setValue(0.0)
        self.end_spin.setValue(0.0)
        self.total_duration = 0.0
