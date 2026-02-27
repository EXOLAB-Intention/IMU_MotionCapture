"""
Navigator panel for displaying files in current directory with folder tree structure.
Supports both folder browsing and HDF5 file browsing modes.
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QHBoxLayout
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QColor, QBrush
import os
from pathlib import Path

from file_io.file_handler import FileHandler


class NavigatorPanel(QWidget):
    """File navigator panel showing folder tree structure with file status"""

    # Signals
    file_selected = pyqtSignal(str, bool)  # filepath, is_processed
    folder_selected = pyqtSignal(str)  # folder path
    directory_changed = pyqtSignal(str)
    h5_trial_selected = pyqtSignal(str, str)  # h5_filepath, h5_internal_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_directory = None
        self.nav_mode = 'folder'  # 'folder' or 'h5'
        self.current_h5_file = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("<b>Navigator</b>")

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)

        self.back_to_folder_btn = QPushButton("Back to Folder")
        self.back_to_folder_btn.clicked.connect(self._back_to_folder_mode)
        self.back_to_folder_btn.setVisible(False)

        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.back_to_folder_btn)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Directory label
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.dir_label)
        
        # File tree with folder structure
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Name", "Type", "Status"])
        self.file_tree.setColumnWidth(0, 200)
        self.file_tree.setColumnWidth(1, 60)
        self.file_tree.setColumnWidth(2, 80)
        self.file_tree.itemClicked.connect(self._on_item_clicked)
        self.file_tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        layout.addWidget(self.file_tree)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def set_directory(self, directory: str):
        """Set the directory to display in folder mode"""
        if os.path.exists(directory):
            self.current_directory = directory
            self.nav_mode = 'folder'
            self.current_h5_file = None
            self.back_to_folder_btn.setVisible(False)
            self.dir_label.setText(f"Directory: {directory}")
            self.refresh()
            self.directory_changed.emit(directory)
        else:
            self.status_label.setText("Directory does not exist")

    def set_h5_file(self, h5_filepath: str):
        """Switch to H5 browsing mode and display the file's trial hierarchy"""
        if not os.path.exists(h5_filepath):
            self.status_label.setText("H5 file does not exist")
            return
        self.nav_mode = 'h5'
        self.current_h5_file = h5_filepath
        self.back_to_folder_btn.setVisible(True)
        self.dir_label.setText(f"H5: {os.path.basename(h5_filepath)}")
        self._refresh_h5_tree()

    def _back_to_folder_mode(self):
        """Return to folder browsing mode"""
        self.nav_mode = 'folder'
        self.current_h5_file = None
        self.back_to_folder_btn.setVisible(False)
        if self.current_directory:
            self.dir_label.setText(f"Directory: {self.current_directory}")
        self.refresh()

    def _refresh_h5_tree(self):
        """Populate the tree with HDF5 file hierarchy: Subject > Activity > Level > Trial"""
        self.file_tree.clear()
        if not self.current_h5_file:
            return

        try:
            structure = FileHandler.scan_h5_file(self.current_h5_file)
        except Exception as e:
            self.status_label.setText(f"Error scanning H5: {e}")
            return

        trial_count = 0
        # Colors for hierarchy levels
        subject_color = QColor(100, 180, 255)   # Blue
        activity_color = QColor(255, 180, 100)   # Orange
        level_color = QColor(180, 100, 255)      # Purple
        trial_color = QColor(100, 220, 150)      # Green

        for subject_id, activities in sorted(structure.items()):
            subj_item = QTreeWidgetItem(self.file_tree, [subject_id, "Subject", ""])
            subj_item.setData(0, Qt.UserRole, None)  # Not selectable as trial
            subj_item.setData(1, Qt.UserRole, "h5_group")
            subj_item.setForeground(0, subject_color)
            subj_item.setExpanded(True)

            for activity, levels in sorted(activities.items()):
                act_item = QTreeWidgetItem(subj_item, [activity, "Activity", ""])
                act_item.setData(0, Qt.UserRole, None)
                act_item.setData(1, Qt.UserRole, "h5_group")
                act_item.setForeground(0, activity_color)
                act_item.setExpanded(True)

                for level, trials in sorted(levels.items()):
                    lvl_item = QTreeWidgetItem(act_item, [level, "Level", ""])
                    lvl_item.setData(0, Qt.UserRole, None)
                    lvl_item.setData(1, Qt.UserRole, "h5_group")
                    lvl_item.setForeground(0, level_color)
                    lvl_item.setExpanded(True)

                    for trial_id in sorted(trials):
                        h5_path = f"{subject_id}/{activity}/{level}/{trial_id}"
                        trial_item = QTreeWidgetItem(lvl_item, [trial_id, "Trial", "Ready"])
                        trial_item.setData(0, Qt.UserRole, h5_path)
                        trial_item.setData(1, Qt.UserRole, "h5_trial")
                        trial_item.setForeground(0, trial_color)
                        trial_item.setForeground(2, QColor(100, 200, 100))
                        trial_count += 1

        self.status_label.setText(f"H5: {trial_count} trials found")

    def refresh(self):
        """Refresh the file tree"""
        if self.nav_mode == 'h5':
            self._refresh_h5_tree()
            return
        if not self.current_directory:
            return
        
        self.file_tree.clear()
        
        try:
            # Get all items in current directory
            items = os.listdir(self.current_directory)
            
            folders = []
            files = []
            
            for item in items:
                full_path = os.path.join(self.current_directory, item)
                if os.path.isdir(full_path):
                    # Skip hidden folders and __pycache__
                    if not item.startswith('.') and item != '__pycache__':
                        folders.append((item, full_path))
                elif os.path.isfile(full_path):
                    files.append((item, full_path))
            
            # Add folders first
            for folder_name, folder_path in sorted(folders):
                self._add_folder_item(folder_name, folder_path)
            
            # Add files in root directory
            for file_name, file_path in sorted(files):
                self._add_file_item(self.file_tree, file_name, file_path)
            
            # Count totals
            total_items = len(folders) + len(files)
            self.status_label.setText(f"Total: {len(folders)} folders, {len(files)} files")
            
        except Exception as e:
            self.status_label.setText(f"Error scanning directory: {str(e)}")
    
    def _add_folder_item(self, folder_name: str, folder_path: str):
        """Add a folder and its contents to the tree"""
        folder_item = QTreeWidgetItem(self.file_tree, [folder_name, "Folder", ""])
        folder_item.setData(0, Qt.UserRole, folder_path)
        folder_item.setData(1, Qt.UserRole, "folder")  # Mark as folder
        folder_item.setForeground(0, QColor(100, 150, 255))  # Blue for folders
        folder_item.setExpanded(False)  # Start collapsed
        
        try:
            # Add files in this folder
            folder_items = os.listdir(folder_path)
            
            csv_files = []
            mcp_files = []
            cal_files = []
            subject_files = []
            other_files = []
            
            for item in folder_items:
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    ext = os.path.splitext(item)[1].lower()
                    if ext == '.csv':
                        csv_files.append((item, item_path))
                    elif ext == '.mcp':
                        mcp_files.append((item, item_path))
                    elif ext == '.cal':
                        cal_files.append((item, item_path))
                    elif ext == '.subject':
                        subject_files.append((item, item_path))
                    else:
                        other_files.append((item, item_path))
            
            # Add files grouped by type
            for file_name, file_path in sorted(csv_files):
                self._add_file_item(folder_item, file_name, file_path)
            
            for file_name, file_path in sorted(mcp_files):
                self._add_file_item(folder_item, file_name, file_path)
            
            for file_name, file_path in sorted(cal_files):
                self._add_file_item(folder_item, file_name, file_path)
            
            for file_name, file_path in sorted(subject_files):
                self._add_file_item(folder_item, file_name, file_path)
            
            for file_name, file_path in sorted(other_files):
                self._add_file_item(folder_item, file_name, file_path)
            
        except Exception as e:
            error_item = QTreeWidgetItem(folder_item, [f"Error: {str(e)}", "", ""])
            error_item.setForeground(0, QColor(255, 100, 100))
    
    def _add_file_item(self, parent, file_name: str, file_path: str):
        """Add a file item to the tree"""
        ext = os.path.splitext(file_name)[1].lower()
        
        # Determine file type and status
        if ext == '.csv':
            # Check if processed file exists
            processed_path = file_path.replace('.csv', '_processed.csv')
            if os.path.exists(processed_path):
                file_type = "CSV"
                status = "Raw"
                color = QColor(200, 200, 100)  # Yellow for raw with processed version
            else:
                file_type = "CSV"
                status = "Raw Only"
                color = QColor(200, 100, 100)  # Red for raw only
            is_processed = False
            
        elif '_processed.csv' in file_name:
            file_type = "Processed"
            status = "Ready"
            color = QColor(100, 200, 100)  # Green for processed
            is_processed = True
            
        elif ext == '.mcp':
            file_type = "MCP"
            status = "Processed"
            color = QColor(100, 200, 150)  # Light green for MCP
            is_processed = True
            
        elif ext == '.cal':
            file_type = "Calib"
            status = "Config"
            color = QColor(150, 150, 200)  # Purple for calibration
            is_processed = False
            
        elif ext == '.subject':
            file_type = "Subject"
            status = "Config"
            color = QColor(200, 150, 200)  # Pink for subject info
            is_processed = False

        elif ext == '.h5':
            file_type = "H5"
            status = "Browse"
            color = QColor(120, 190, 240)  # Blue-cyan for H5 container
            is_processed = "h5_file"
            
        else:
            file_type = "File"
            status = ""
            color = QColor(150, 150, 150)  # Gray for other files
            is_processed = False
        
        item = QTreeWidgetItem(parent, [file_name, file_type, status])
        item.setData(0, Qt.UserRole, file_path)
        item.setData(1, Qt.UserRole, is_processed)
        item.setForeground(2, color)
    
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click"""
        filepath = item.data(0, Qt.UserRole)
        item_type = item.data(1, Qt.UserRole)

        if item_type == "h5_trial" and filepath:
            self.status_label.setText(f"Trial: {filepath}")
            return
        if item_type == "h5_group":
            self.status_label.setText(f"Group: {item.text(0)}")
            return
        if item_type == "h5_file" and filepath:
            self.status_label.setText(f"H5 file: {os.path.basename(filepath)} (double-click to browse)")
            return

        if filepath:
            if item_type == "folder":
                # Folder clicked
                self.status_label.setText(f"Folder: {os.path.basename(filepath)}")
                self.folder_selected.emit(filepath)
            else:
                # File clicked
                self.status_label.setText(f"Selected: {os.path.basename(filepath)}")
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item double-click to open file or toggle folder"""
        filepath = item.data(0, Qt.UserRole)
        item_type = item.data(1, Qt.UserRole)

        if item_type == "h5_trial" and filepath and self.current_h5_file:
            # H5 trial node - emit h5_trial_selected signal
            self.h5_trial_selected.emit(self.current_h5_file, filepath)
            return

        if item_type == "h5_group":
            # Toggle H5 group expansion
            item.setExpanded(not item.isExpanded())
            return

        if item_type == "h5_file" and filepath:
            # Enter H5 browsing mode instead of raw import
            self.set_h5_file(filepath)
            return

        if filepath:
            if item_type == "folder":
                # Toggle folder expansion
                item.setExpanded(not item.isExpanded())
            elif isinstance(item_type, bool):
                # File - emit signal to open
                is_processed = item_type
                self.file_selected.emit(filepath, is_processed)
    
    def get_selected_file(self):
        """Get currently selected file"""
        current_item = self.file_tree.currentItem()
        if current_item:
            filepath = current_item.data(0, Qt.UserRole)
            is_processed = current_item.data(1, Qt.UserRole)
            return filepath, is_processed
        return None, None
    
    def highlight_file(self, filepath: str):
        """Highlight a specific file in the tree"""
        # TODO: Implement file highlighting
        pass
