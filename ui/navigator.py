"""
Navigator panel for displaying files in current directory
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, 
    QLabel, QPushButton, QHBoxLayout
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QColor
import os
from pathlib import Path

from file_io.file_handler import FileHandler


class NavigatorPanel(QWidget):
    """File navigator panel showing importable and processed files"""
    
    # Signals
    file_selected = pyqtSignal(str, bool)  # filepath, is_processed
    directory_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_directory = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("<b>Navigator</b>")
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Directory label
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.dir_label)
        
        # File tree
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["File", "Type", "Status"])
        self.file_tree.setColumnWidth(0, 250)
        self.file_tree.setColumnWidth(1, 80)
        self.file_tree.itemClicked.connect(self._on_item_clicked)
        self.file_tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        layout.addWidget(self.file_tree)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def set_directory(self, directory: str):
        """Set the directory to display"""
        if os.path.exists(directory):
            self.current_directory = directory
            self.dir_label.setText(f"Directory: {directory}")
            self.refresh()
            self.directory_changed.emit(directory)
        else:
            self.status_label.setText("Directory does not exist")
    
    def refresh(self):
        """Refresh the file list"""
        if not self.current_directory:
            return
        
        self.file_tree.clear()
        
        # Scan directory
        files = FileHandler.scan_directory(self.current_directory)
        
        # Add raw files
        if files['raw']:
            raw_parent = QTreeWidgetItem(self.file_tree, ["Raw Data Files"])
            raw_parent.setExpanded(True)
            
            for filepath in sorted(files['raw']):
                filename = os.path.basename(filepath)
                item = QTreeWidgetItem(raw_parent, [filename, "Raw", "Not Processed"])
                item.setData(0, Qt.UserRole, filepath)
                item.setData(1, Qt.UserRole, False)  # is_processed flag
                item.setForeground(2, QColor(200, 100, 100))
        
        # Add processed files
        if files['processed']:
            processed_parent = QTreeWidgetItem(self.file_tree, ["Processed Files"])
            processed_parent.setExpanded(True)
            
            for filepath in sorted(files['processed']):
                filename = os.path.basename(filepath)
                item = QTreeWidgetItem(processed_parent, [filename, "Processed", "Ready"])
                item.setData(0, Qt.UserRole, filepath)
                item.setData(1, Qt.UserRole, True)  # is_processed flag
                item.setForeground(2, QColor(100, 200, 100))
        
        # Update status
        total_files = len(files['raw']) + len(files['processed'])
        self.status_label.setText(
            f"Total: {total_files} files ({len(files['raw'])} raw, {len(files['processed'])} processed)"
        )
    
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click"""
        filepath = item.data(0, Qt.UserRole)
        if filepath:
            is_processed = item.data(1, Qt.UserRole)
            self.status_label.setText(f"Selected: {os.path.basename(filepath)}")
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item double-click to open file"""
        filepath = item.data(0, Qt.UserRole)
        if filepath:
            is_processed = item.data(1, Qt.UserRole)
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
