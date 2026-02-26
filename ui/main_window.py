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
from ui.subject_info_panel import SubjectInfoPanel
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
        
        # ====================================================================
        # LEFT PANEL: Subject Info (top) + Notes (bottom) - Vertical split
        # ====================================================================
        left_panel = QSplitter(Qt.Vertical)
        left_panel.setMaximumWidth(350)
        
        # Subject info panel (top)
        self.subject_info = SubjectInfoPanel()
        left_panel.addWidget(self.subject_info)
        
        # Notes panel (bottom)
        self.notes = NotesPanel()
        left_panel.addWidget(self.notes)
        
        # Set initial sizes (roughly equal)
        left_panel.setSizes([175, 175])
        
        main_layout.addWidget(left_panel)
        
        # ====================================================================
        # RIGHT PANEL: Main View (top) + Navigator (bottom) - Adjustable splitter
        # ====================================================================
        right_panel = QSplitter(Qt.Vertical)
        
        # Main view (top - 3D visualization and graphs)
        self.main_view = MainView()
        right_panel.addWidget(self.main_view)
        
        # Navigator (bottom)
        self.navigator = NavigatorPanel()
        right_panel.addWidget(self.navigator)
        
        # Set initial sizes (5:5 ratio for better balance - 450:450)
        # User can adjust with mouse
        right_panel.setSizes([450, 450])
        right_panel.setStretchFactor(0, 1)  # Main view stretches
        right_panel.setStretchFactor(1, 1)  # Navigator stretches equally
        
        main_layout.addWidget(right_panel, stretch=1)
        
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
        self.navigator.folder_selected.connect(self.on_folder_selected)
        
        # Subject info signals
        self.subject_info.subject_info_changed.connect(self.on_subject_info_changed)
        self.subject_info.subject_info_saved.connect(self.on_subject_info_saved)
        
        # Notes signals
        self.notes.notes_changed.connect(self.update_notes)
        
        # Main view signals
        self.main_view.process_requested.connect(self.process_data)
        self.main_view.mode_changed.connect(self.update_mode)
        self.main_view.mode_changed.connect(self.main_view.visualization_3d.refresh_view_mode)
        self.main_view.mode_changed.connect(self.main_view.graph_view.update_mode_selection)
        self.main_view.mode_changed.connect(self.subject_info.refresh_mode_ui)

    @pyqtSlot(str)
    def update_mode(self, mode_name: str):
        """Update settings when switching mode"""
        formatted_mode = "Upper-body" if mode_name == "Upper-body" else "Lower-body"
        app_settings.mode.mode_type = formatted_mode
        app_settings.refresh_sensor_mapping()
        
        self.statusBar().showMessage(f"Switched to {formatted_mode} mode", 3000)
        self._update_calibration_status()
        
        # Check if current data matches mode
        if self.current_data and not self.current_data.has_all_sensors:
            self.statusBar().showMessage(f"Warning: Data incomplete for {formatted_mode}", 5000)
    
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
            
            # Auto-load subject info from the file's folder
            file_folder = os.path.dirname(filepath)
            self._auto_load_subject_info_for_folder(file_folder)
            
            self.statusBar().showMessage(f"Imported: {filepath}", 3000)
            
            # Check if all sensors are present
            if not data.has_all_sensors:
                current_mode = app_settings.mode.mode_type
                QMessageBox.warning(
                    self,
                    "Incomplete Data",
                    f"Not all required sensors are present for {current_mode} in the data.\n"
                    f"Please check your sensor mapping and mode settings."
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
            
            # Auto-load subject info from the file's folder
            file_folder = os.path.dirname(filepath)
            self._auto_load_subject_info_for_folder(file_folder)
            
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
            if not self.current_data.is_calibrated:
                self.current_data = self.data_processor.calibration_processor.apply_to_data(
                    self.current_data
                )
            
            # Compute kinematics only (no calibration step)
            processed_data = self.data_processor.process_kinematics_only(self.current_data)
            
            self.current_data = processed_data
            
            # Update UI
            self.main_view.set_data(processed_data)
            
            self.statusBar().showMessage("Processing complete", 3000)
            
            current_mode = app_settings.mode.mode_type
            QMessageBox.information(
                self,
                "Processing Complete",
                f"{current_mode} motion data has been processed successfully.\n\n"
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
    
    @pyqtSlot(str)
    def on_folder_selected(self, folder_path: str):
        """Handle folder selection in navigator"""
        # Set current folder in subject info panel
        self.subject_info.set_current_folder(folder_path)
        self.statusBar().showMessage(f"Selected folder: {os.path.basename(folder_path)}", 2000)
    
    @pyqtSlot(dict)
    def on_subject_info_changed(self, subject_info: dict):
        """Handle subject info change - update visualization"""
        try:
            # Update visualization segment lengths
            self.main_view.update_segment_lengths(subject_info)
            
            self.statusBar().showMessage(
                f"Updated segment lengths for {subject_info.get('name', 'subject')}", 
                2000
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Update Error",
                f"Failed to update visualization:\n{str(e)}"
            )
    
    @pyqtSlot(str)
    
    def _auto_load_subject_info_for_folder(self, folder_path: str):
        """
        Auto-load subject info file (.subject) from a folder if it exists.
        This is called when importing/opening files to automatically apply segment lengths.
        """
        if not folder_path or not os.path.isdir(folder_path):
            return
        
        # Set the folder first so Save button works properly
        self.subject_info.set_current_folder(folder_path)
        
        try:
            # Look for .subject files in the folder
            subject_files = [f for f in os.listdir(folder_path) if f.endswith('.subject')]
            
            if subject_files:
                # Load the first .subject file found
                subject_file_path = os.path.join(folder_path, subject_files[0])
                
                # Load subject info into the subject info panel
                self.subject_info.load_subject_file(subject_file_path)
                
                # The signal will automatically update visualization
                self.statusBar().showMessage(
                    f"Auto-loaded subject info: {subject_files[0]}", 
                    2000
                )
        except Exception as e:
            # Silently fail - it's an optional feature
            print(f"Could not auto-load subject info from {folder_path}: {e}")
    def on_subject_info_saved(self, filepath: str):
        """Handle subject info save - refresh navigator"""
        self.navigator.refresh()
        self.statusBar().showMessage(f"Subject info saved: {os.path.basename(filepath)}", 2000)
    
    def _update_calibration_status(self):
        """Update calibration status indicator in status bar"""
        if self.data_processor.calibration_processor.is_calibrated:
            pose_type = self.data_processor.calibration_processor.pose_type or "Unknown"
            n_sensors = len(self.data_processor.calibration_processor.offset_quaternions)
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
                f"mode: {self.data_processor.calibration_processor.mode}\n"
                f"Sensors: {len(self.data_processor.calibration_processor.offset_quaternions)}"
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
        """Perform calibration on current data using selected time range or first 0.5s"""
        if not self.current_data:
            QMessageBox.warning(
                self,
                "No Data",
                "Load calibration trial data first.\n"
                "Use File > Import to load a N-pose or T-pose trial CSV."
            )
            return
        
        try:
            # Get selected time range from time bar
            selected_start, selected_end = self.main_view.get_selected_time_range()
            full_start, full_end = self.current_data.get_time_range()
            
            # Determine calibration time range
            # If user selected a specific range (not full range), use that
            # Otherwise, use first 0.5 seconds (or first frame if file is shorter)
            if abs(selected_start - full_start) > 0.01 or abs(selected_end - full_end) > 0.01:
                # User selected a specific range
                start_time = selected_start
                end_time = selected_end
                range_source = "user selection"
            else:
                # Use first 0.5 seconds for calibration (assuming standing pose at start)
                start_time = full_start
                end_time = min(full_start + 0.5, full_end)
                range_source = "first 0.5s (auto)"
            
            # Ask for pose type
            from PyQt5.QtWidgets import QInputDialog
            pose_type, ok = QInputDialog.getItem(
                self,
                "Calibration Pose",
                f"Calibration range: {start_time:.2f}s - {end_time:.2f}s ({range_source})\n\n"
                "Select calibration pose type:",
                ["N-pose", "T-pose"],
                0,
                False
            )
            
            if not ok:
                return
            
            current_mode = app_settings.mode.mode_type

            self.statusBar().showMessage("Performing calibration...")
            
            # Perform calibration
            self.data_processor.calibration_processor.calibrate(
                self.current_data,
                start_time,
                end_time,
                pose_type,
                current_mode
            )
            
            self.statusBar().showMessage("Calibration complete", 3000)
            
            self._update_calibration_status()
            
            QMessageBox.information(
                self,
                "Calibration Complete",
                f"Calibration performed successfully!\n\n"
                f"Pose type: {pose_type}\n"
                f"mode: {current_mode}\n"
                f"Time range: {start_time:.2f}s - {end_time:.2f}s ({range_source})\n"
                f"Sensors calibrated: {len(self.data_processor.calibration_processor.offset_quaternions)}\n\n"
                f"Now click 'Process > Process Data' to apply calibration."
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
