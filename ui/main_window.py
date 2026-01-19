"""
Main application window integrating all components
"""
import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QMessageBox, QTabWidget
)
from PyQt5.QtCore import Qt, pyqtSlot
from datetime import datetime

from ui.menu_bar import MenuBar
from ui.navigator import NavigatorPanel
from ui.notes import NotesPanel
from ui.main_view import MainView
from file_io.file_handler import FileHandler
from core.data_processor import DataProcessor
from config.settings import app_settings


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.current_file_path = None
        self.file_handler = FileHandler()
        self.data_processor = DataProcessor()
        
        self._init_ui()
        self._connect_signals()
        
        # Set initial directory to current working directory
        initial_dir = os.getcwd()
        self.navigator.set_directory(initial_dir)
    
    def _init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("IMU Motion Capture System")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create menu bar
        menu_bar = self.menuBar()
        self.menu = MenuBar(menu_bar)
        
        # Status bar with calibration indicator
        self.calibration_status_label = QWidget()
        from PyQt5.QtWidgets import QLabel as QStatusLabel
        calib_layout = QHBoxLayout()
        calib_layout.setContentsMargins(0, 0, 10, 0)
        self.calib_indicator = QStatusLabel("⚫ No Calibration")
        self.calib_indicator.setStyleSheet("color: #888; font-weight: bold;")
        calib_layout.addWidget(self.calib_indicator)
        self.calibration_status_label.setLayout(calib_layout)
        self.statusBar().addPermanentWidget(self.calibration_status_label)
        self.statusBar().showMessage("Ready")
        
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left panel: Navigator and Notes in tabs
        left_panel = QTabWidget()
        left_panel.setMaximumWidth(350)
        
        self.navigator = NavigatorPanel()
        self.notes = NotesPanel()
        
        left_panel.addTab(self.navigator, "Navigator")
        left_panel.addTab(self.notes, "Notes")
        
        main_layout.addWidget(left_panel)
        
        # Right panel: Main view
        self.main_view = MainView()
        main_layout.addWidget(self.main_view, stretch=1)
        
        central_widget.setLayout(main_layout)
    
    def _connect_signals(self):
        """Connect signals and slots"""
        # Menu signals
        self.menu.import_file_requested.connect(self.import_file)
        self.menu.open_file_requested.connect(self.open_file)
        self.menu.save_requested.connect(self.save_file)
        self.menu.save_as_requested.connect(self.save_file_as)
        self.menu.exit_requested.connect(self.close)
        
        # Calibration signals
        self.menu.load_calibration_requested.connect(self.load_calibration)
        self.menu.save_calibration_requested.connect(self.save_calibration)
        self.menu.perform_calibration_requested.connect(self.perform_calibration)
        
        # Navigator signals
        self.navigator.file_selected.connect(self.load_selected_file)
        
        # Notes signals
        self.notes.notes_changed.connect(self.update_notes)
        
        # Main view signals
        self.main_view.process_requested.connect(self.process_data)
    
    @pyqtSlot(str)
    def import_file(self, filepath: str):
        """Import raw IMU data"""
        try:
            self.statusBar().showMessage(f"Importing {filepath}...")
            
            # Import data
            data = self.file_handler.import_raw_data(filepath)
            
            self.current_data = data
            self.current_file_path = None  # Not saved yet
            
            # Update UI
            self.main_view.set_data(data)
            self.notes.set_file(os.path.basename(filepath))
            self.notes.set_notes(data.notes)
            
            self.statusBar().showMessage(f"Imported: {filepath}", 3000)
            
            # Check if all sensors are present
            if not data.has_all_sensors:
                QMessageBox.warning(
                    self,
                    "Incomplete Data",
                    "Not all required sensors are present in the data.\n"
                    "Expected 7 sensors for lower body."
                )
        
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import file:\n{str(e)}")
            self.statusBar().showMessage("Import failed", 3000)
    
    @pyqtSlot(str)
    def open_file(self, filepath: str):
        """Open processed motion capture file"""
        try:
            self.statusBar().showMessage(f"Opening {filepath}...")
            
            # Load data
            data = self.file_handler.load_processed_data(filepath)
            
            self.current_data = data
            self.current_file_path = filepath
            
            # Update UI
            self.main_view.set_data(data)
            self.notes.set_file(os.path.basename(filepath))
            self.notes.set_notes(data.notes)
            
            self.statusBar().showMessage(f"Opened: {filepath}", 3000)
        
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to open file:\n{str(e)}")
            self.statusBar().showMessage("Open failed", 3000)
    
    @pyqtSlot(str, bool)
    def load_selected_file(self, filepath: str, is_processed: bool):
        """Load file selected from navigator"""
        if is_processed:
            self.open_file(filepath)
        else:
            self.import_file(filepath)
    
    @pyqtSlot()
    def save_file(self):
        """Save current data"""
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "No data to save")
            return
        
        if self.current_file_path:
            self._save_to_file(self.current_file_path)
        else:
            # No file path yet, trigger save as
            self.menu._on_save_as()
    
    @pyqtSlot(str)
    def save_file_as(self, filepath: str):
        """Save data to new file"""
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "No data to save")
            return
        
        self._save_to_file(filepath)
    
    def _save_to_file(self, filepath: str):
        """Internal save method"""
        try:
            self.statusBar().showMessage(f"Saving to {filepath}...")
            
            # Update notes in data
            self.current_data.notes = self.notes.get_notes()
            
            # Check if saving a time slice
            start_time, end_time = self.main_view.get_selected_time_range()
            data_start, data_end = self.current_data.get_time_range()
            
            # If selection is not full range, save slice
            if abs(start_time - data_start) > 0.01 or abs(end_time - data_end) > 0.01:
                reply = QMessageBox.question(
                    self,
                    "Save Selection",
                    f"Save selected time range ({start_time:.2f}s - {end_time:.2f}s) only?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    save_data = self.current_data.get_time_slice(start_time, end_time)
                else:
                    save_data = self.current_data
            else:
                save_data = self.current_data
            
            # Save to file
            self.file_handler.save_processed_data(save_data, filepath)
            
            self.current_file_path = filepath
            self.statusBar().showMessage(f"Saved: {filepath}", 3000)
            
            # Refresh navigator
            self.navigator.refresh()
        
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{str(e)}")
            self.statusBar().showMessage("Save failed", 3000)
    
    @pyqtSlot()
    def process_data(self):
        """Process current motion capture data using existing calibration"""
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "No data to process.\n\nLoad a motion trial first.")
            return
        
        # Check if calibration is available
        if not self.data_processor.calibration_processor.is_calibrated:
            reply = QMessageBox.question(
                self,
                "No Calibration",
                "No calibration data loaded.\n\n"
                "Do you want to load a calibration file?\n"
                "Or use 'Process > Perform Calibration' if this is a calibration trial.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # Trigger load calibration
                self.menu._on_load_calibration()
                # Check again after load attempt
                if not self.data_processor.calibration_processor.is_calibrated:
                    return
            else:
                return
        
        try:
            self.statusBar().showMessage("Processing data (using existing calibration)...")
            
            # Apply calibration to data first (if not already applied)
            calibrated_data = self.data_processor.calibration_processor.apply_to_data(
                self.current_data
            )
            
            # Compute kinematics only (no calibration step)
            processed_data = self.data_processor.process_kinematics_only(calibrated_data)
            
            self.current_data = processed_data
            
            # Update UI
            self.main_view.set_data(processed_data)
            
            self.statusBar().showMessage("Processing complete", 3000)
            
            QMessageBox.information(
                self,
                "Processing Complete",
                "Motion data has been processed successfully.\n\n"
                f"Calibration: {self.data_processor.calibration_processor.pose_type}\n"
                "Joint angles and kinematics have been computed."
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Failed to process data:\n{str(e)}")
            self.statusBar().showMessage("Processing failed", 3000)
    
    @pyqtSlot(str)
    def update_notes(self, notes: str):
        """Update notes in current data"""
        if self.current_data:
            self.current_data.notes = notes
            self.statusBar().showMessage("Notes updated", 2000)
    
    def _update_calibration_status(self):
        """Update calibration status indicator in status bar"""
        if self.data_processor.calibration_processor.is_calibrated:
            pose_type = self.data_processor.calibration_processor.pose_type or "Unknown"
            n_sensors = len(self.data_processor.calibration_processor.reference_orientations)
            self.calib_indicator.setText(f"🟢 {pose_type} Calibration ({n_sensors} sensors)")
            self.calib_indicator.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.calib_indicator.setText("⚫ No Calibration")
            self.calib_indicator.setStyleSheet("color: #888; font-weight: bold;")
    
    @pyqtSlot(str)
    def load_calibration(self, filepath: str):
        """Load calibration from file and apply to current data"""
        try:
            self.statusBar().showMessage(f"Loading calibration from {filepath}...")
            
            # Load calibration
            self.data_processor.calibration_processor.load_calibration(filepath)
            
            # Apply to current data if available
            if self.current_data:
                reply = QMessageBox.question(
                    self,
                    "Apply Calibration",
                    "Apply loaded calibration to current data?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    calibrated_data = self.data_processor.calibration_processor.apply_to_data(
                        self.current_data
                    )
                    self.current_data = calibrated_data
                    self.main_view.set_data(calibrated_data)
                    self._update_calibration_status()
                    self.statusBar().showMessage("Calibration applied", 3000)
            else:
                self._update_calibration_status()
                self.statusBar().showMessage("Calibration loaded (no data to apply)", 3000)
            
            QMessageBox.information(
                self,
                "Calibration Loaded",
                f"Calibration loaded successfully!\n"
                f"Pose type: {self.data_processor.calibration_processor.pose_type}\n"
                f"Sensors: {len(self.data_processor.calibration_processor.reference_orientations)}"
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Calibration Error", f"Failed to load calibration:\n{str(e)}")
            self.statusBar().showMessage("Calibration load failed", 3000)
    
    @pyqtSlot(str)
    def save_calibration(self, filepath: str):
        """Save current calibration to file"""
        try:
            if not self.data_processor.calibration_processor.is_calibrated:
                QMessageBox.warning(
                    self,
                    "No Calibration",
                    "No calibration data available.\n"
                    "Perform calibration first using 'Perform Calibration'."
                )
                return
            
            self.statusBar().showMessage(f"Saving calibration to {filepath}...")
            self.data_processor.calibration_processor.save_calibration(filepath)
            self.statusBar().showMessage(f"Calibration saved: {filepath}", 3000)
            
            QMessageBox.information(
                self,
                "Calibration Saved",
                f"Calibration saved successfully to:\n{filepath}"
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save calibration:\n{str(e)}")
            self.statusBar().showMessage("Calibration save failed", 3000)
    
    @pyqtSlot()
    def perform_calibration(self):
        """Perform calibration on current data (use entire duration)"""
        if not self.current_data:
            QMessageBox.warning(
                self,
                "No Data",
                "Load calibration trial data first.\n"
                "Use File > Import to load a N-pose or T-pose trial CSV."
            )
            return
        
        try:
            # Use entire time range for calibration
            start_time, end_time = self.current_data.get_time_range()
            
            # Ask for pose type
            from PyQt5.QtWidgets import QInputDialog
            pose_type, ok = QInputDialog.getItem(
                self,
                "Calibration Pose",
                "Select calibration pose type:",
                ["N-pose", "T-pose"],
                0,
                False
            )
            
            if not ok:
                return
            
            self.statusBar().showMessage("Performing calibration...")
            
            # Perform calibration
            self.data_processor.calibration_processor.calibrate(
                self.current_data,
                start_time,
                end_time,
                pose_type
            )
            
            self.statusBar().showMessage("Calibration complete", 3000)
            
            self._update_calibration_status()
            
            QMessageBox.information(
                self,
                "Calibration Complete",
                f"Calibration performed successfully!\n\n"
                f"Pose type: {pose_type}\n"
                f"Duration: {end_time - start_time:.2f} seconds\n"
                f"Sensors calibrated: {len(self.data_processor.calibration_processor.reference_orientations)}\n\n"
                f"Use 'Process > Save Calibration' to save for later use."
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Calibration Error", f"Failed to perform calibration:\n{str(e)}")
            self.statusBar().showMessage("Calibration failed", 3000)
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.current_data and not self.current_data.is_processed:
            reply = QMessageBox.question(
                self,
                "Unsaved Data",
                "You have unsaved/unprocessed data. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        event.accept()
