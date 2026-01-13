"""
Notes panel for trial annotations
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLabel, 
    QPushButton, QHBoxLayout
)
from PyQt5.QtCore import pyqtSignal


class NotesPanel(QWidget):
    """Notes panel for annotating trials"""
    
    # Signals
    notes_changed = pyqtSignal(str)  # notes text
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("<b>Notes</b>")
        
        self.save_btn = QPushButton("Save Notes")
        self.save_btn.clicked.connect(self._on_save_notes)
        self.save_btn.setEnabled(False)
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.save_btn)
        
        layout.addLayout(header_layout)
        
        # File label
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.file_label)
        
        # Notes text edit
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText(
            "Enter notes about this trial...\n\n"
            "Examples:\n"
            "- Subject condition\n"
            "- Experimental protocol\n"
            "- Data quality issues\n"
            "- Special observations"
        )
        self.notes_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.notes_edit)
        
        # Character count
        self.char_label = QLabel("0 characters")
        self.char_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.char_label)
        
        self.setLayout(layout)
    
    def set_file(self, filename: str):
        """Set the current file"""
        self.current_file = filename
        self.file_label.setText(f"File: {filename}")
        self.save_btn.setEnabled(True)
    
    def set_notes(self, notes: str):
        """Load notes into the editor"""
        self.notes_edit.blockSignals(True)  # Prevent textChanged signal
        self.notes_edit.setPlainText(notes)
        self.notes_edit.blockSignals(False)
        self._update_char_count()
    
    def get_notes(self) -> str:
        """Get current notes text"""
        return self.notes_edit.toPlainText()
    
    def clear(self):
        """Clear notes and reset state"""
        self.notes_edit.clear()
        self.current_file = None
        self.file_label.setText("No file selected")
        self.save_btn.setEnabled(False)
    
    def _on_text_changed(self):
        """Handle text changed event"""
        self._update_char_count()
    
    def _update_char_count(self):
        """Update character count label"""
        text = self.notes_edit.toPlainText()
        char_count = len(text)
        line_count = text.count('\n') + 1 if text else 0
        self.char_label.setText(f"{char_count} characters, {line_count} lines")
    
    def _on_save_notes(self):
        """Handle save notes button"""
        notes = self.get_notes()
        self.notes_changed.emit(notes)
    
    def set_read_only(self, read_only: bool):
        """Set notes editor to read-only mode"""
        self.notes_edit.setReadOnly(read_only)
        self.save_btn.setEnabled(not read_only)
