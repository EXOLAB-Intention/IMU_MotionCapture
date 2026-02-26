"""
Subject information panel for entering and managing subject data
Saves subject info (.subject files) in subject folders
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QGroupBox, QFormLayout,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot
import json
import os
from config.settings import app_settings


class SubjectInfoPanel(QWidget):
    """Panel for subject physical information input"""
    
    # Signals
    subject_info_changed = pyqtSignal(dict)  # Emits subject info dict
    subject_info_saved = pyqtSignal(str)  # Emits saved file path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_subject_file = None
        self.current_folder = None
        
        # Initialize predictor once (cached)
        self.predictor = None
        self._predictor_failed = False  # Flag to avoid repeated attempts
        
        # Initialize UI
        self._init_ui()
        # Switch UI based on current mode
        self.refresh_mode_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header = QLabel("<b>Subject Information</b>")
        layout.addWidget(header)
        
        # Subject info group
        info_group = QGroupBox("Physical Parameters")
        info_layout = QFormLayout()
        
        # Name input
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., KTY, HEB")
        info_layout.addRow("Name:", self.name_input)
        
        # Height input
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("cm")
        self.height_input.setText("170.0")
        info_layout.addRow("Height (cm):", self.height_input)
        
        # Shoe size input
        self.shoe_size_input = QLineEdit()
        self.shoe_size_input.setPlaceholderText("mm")
        self.shoe_size_input.setText("270.0")
        info_layout.addRow("Shoe Size (mm):", self.shoe_size_input)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Segment ratios (calculated automatically, read-only)
        self.ratio_group = QGroupBox("Body Segment Ratios (Auto-calculated)")
        self.ratio_main_layout = QVBoxLayout() # To be refreshed based on mode

        # Upper-body ratios
        self.upper_body_widget = QWidget()
        upper_layout = QFormLayout()
        upper_layout.setContentsMargins(0, 0, 0, 0)
        self.abdomen_ratio_label = QLabel("0.190")
        self.chest_ratio_label = QLabel("0.150")
        self.head_ratio_label = QLabel("0.100")
        self.shoulder_ratio_label = QLabel("0.350")
        self.upperarm_ratio_label = QLabel("0.186")
        self.lowerarm_ratio_label = QLabel("0.146")
        upper_layout.addRow("Abdomen:", self.abdomen_ratio_label)
        upper_layout.addRow("Chest:", self.chest_ratio_label)
        upper_layout.addRow("Head:", self.head_ratio_label)
        upper_layout.addRow("Shoulder:", self.shoulder_ratio_label)
        upper_layout.addRow("Upper Arm:", self.upperarm_ratio_label)
        upper_layout.addRow("Lower Arm:", self.lowerarm_ratio_label)
        self.upper_body_widget.setLayout(upper_layout)

        # Lower-body ratios
        self.lower_body_widget = QWidget()
        lower_layout = QFormLayout()
        lower_layout.setContentsMargins(0, 0, 0, 0)
        self.trunk_ratio_label = QLabel("0.288")
        self.pelvis_ratio_label = QLabel("0.250")
        self.thigh_ratio_label = QLabel("0.232")
        self.shank_ratio_label = QLabel("0.246")
        self.foot_ratio_label = QLabel("0.152")
        lower_layout.addRow("Trunk:", self.trunk_ratio_label)
        lower_layout.addRow("Pelvis:", self.pelvis_ratio_label)
        lower_layout.addRow("Thigh:", self.thigh_ratio_label)
        lower_layout.addRow("Shank:", self.shank_ratio_label)
        lower_layout.addRow("Foot:", self.foot_ratio_label)
        self.lower_body_widget.setLayout(lower_layout)

        self.ratio_main_layout.addWidget(self.upper_body_widget)
        self.ratio_main_layout.addWidget(self.lower_body_widget)
        self.ratio_group.setLayout(self.ratio_main_layout)
        layout.addWidget(self.ratio_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._on_save)
        
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._on_load)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        self.apply_btn.setToolTip("Apply current values to visualization")
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.apply_btn)
        
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Connect signals
        self.height_input.textChanged.connect(self._on_height_changed)
        self.shoe_size_input.textChanged.connect(self._on_shoe_size_changed)

    @pyqtSlot(str)
    def refresh_mode_ui(self, mode: str = None):
        """Refresh UI elements based on current mode (Upper-body or Lower-body)"""
        current_mode = app_settings.mode.mode_type
        
        if current_mode == "Upper-body":
            self.upper_body_widget.show()
            self.lower_body_widget.hide()
        else:
            self.upper_body_widget.hide()
            self.lower_body_widget.show()
        
        # Recalculate ratios
        self._calculate_ratios()
    
    def _on_height_changed(self):
        """Recalculate ratios when height changes"""
        self._calculate_ratios()
    
    def _on_shoe_size_changed(self):
        """Recalculate ratios when shoe size changes"""
        self._calculate_ratios()
    
    def _calculate_ratios(self):
        """
        Calculate body segment ratios using machine learning model.
        Uses BodySegmentRatioPredictor trained on Korean anthropometric data.
        Falls back to simple equations if model loading fails.
        """
        try:
            height = float(self.height_input.text())
            shoe_size = float(self.shoe_size_input.text())
            
            # Convert to mm for the predictor
            height_mm = height * 10.0
            foot_length_mm = shoe_size - 10.0  # Approximate: foot length ≈ shoe size - 10mm
            
            # Try to use ML model if not already failed
            if not self._predictor_failed:
                try:
                    # Lazy initialization of predictor (only once)
                    if self.predictor is None:
                        from core.dev.body_segment_ratio.body_segment_ratio import BodySegmentRatioPredictor
                        self.predictor = BodySegmentRatioPredictor()
                        self.status_label.setText("ML model loaded")
                    
                    # Predict ratios using ML model
                    ratios = self.predictor.predict_segment_ratios(height_mm, foot_length_mm)
                    
                    trunk_ratio = ratios['trunk_ratio']
                    thigh_ratio = ratios['thigh_ratio']
                    shank_ratio = ratios['shank_ratio']
                    foot_ratio = ratios['foot_ratio']
                    chest_ratio = ratios['chest_ratio']
                    abdomen_ratio = ratios['abdomen_ratio']
                    upperarm_ratio = ratios['upperarm_ratio']
                    lowerarm_ratio = ratios['lowerarm_ratio']
                    head_ratio = ratios['head_ratio']
                    shoulder_ratio = ratios['shoulder_width_ratio']
                    pelvis_ratio = ratios['pelvis_ratio']
                    
                except Exception as e:
                    # Model failed - mark as failed and use fallback
                    print(f"ML model failed: {e}")
                    self._predictor_failed = True
                    self.predictor = None
                    raise  # Re-raise to use fallback
            else:
                # Already failed before, skip to fallback
                raise ValueError("Using fallback method")
                
        except Exception:
            # Fallback: Use standard anthropometric equations
            try:
                height = float(self.height_input.text())
                shoe_size = float(self.shoe_size_input.text())
                
                # Base ratios (for average height ~170cm)
                base_trunk_ratio = 0.288
                base_thigh_ratio = 0.232
                base_shank_ratio = 0.246
                base_abdomen_ratio = 0.190
                base_chest_ratio = 0.150
                base_head_ratio = 0.100
                base_upperarm_ratio = 0.186
                base_lowerarm_ratio = 0.146
                base_shoulder_ratio = 0.350
                base_pelvis_ratio = 0.250
                
                # Foot length from shoe size
                foot_length_mm = shoe_size - 10.0
                foot_ratio = foot_length_mm / (height * 10.0)
                
                # Height adjustments
                height_factor = (height - 170.0) / 100.0
                
                trunk_ratio = base_trunk_ratio - (height_factor * 0.005)
                thigh_ratio = base_thigh_ratio + (height_factor * 0.003)
                shank_ratio = base_shank_ratio + (height_factor * 0.003)
                abdomen_ratio = base_abdomen_ratio - (height_factor * 0.002)
                chest_ratio = base_chest_ratio - (height_factor * 0.002)
                head_ratio = base_head_ratio - (height_factor * 0.001)
                upperarm_ratio = base_upperarm_ratio + (height_factor * 0.002)
                lowerarm_ratio = base_lowerarm_ratio + (height_factor * 0.002)
                shoulder_ratio = base_shoulder_ratio + (height_factor * 0.005)
                pelvis_ratio = base_pelvis_ratio + (height_factor * 0.004)
                
                # Ensure ratios are reasonable
                trunk_ratio = max(0.25, min(0.32, trunk_ratio))
                thigh_ratio = max(0.20, min(0.27, thigh_ratio))
                shank_ratio = max(0.21, min(0.28, shank_ratio))
                foot_ratio = max(0.12, min(0.18, foot_ratio))
                abdomen_ratio = max(0.17, min(0.21, abdomen_ratio))
                chest_ratio = max(0.13, min(0.17, chest_ratio))
                head_ratio = max(0.09, min(0.11, head_ratio))
                upperarm_ratio = max(0.17, min(0.20, upperarm_ratio))
                lowerarm_ratio = max(0.13, min(0.16, lowerarm_ratio))
                shoulder_ratio = max(0.30, min(0.40, shoulder_ratio))
                pelvis_ratio = max(0.22, min(0.28, pelvis_ratio))
                
            except ValueError:
                # If even this fails, use defaults
                trunk_ratio = 0.288
                thigh_ratio = 0.232
                shank_ratio = 0.246
                foot_ratio = 0.152
                abdomen_ratio = 0.190
                chest_ratio = 0.150
                head_ratio = 0.100
                upperarm_ratio = 0.186
                lowerarm_ratio = 0.146
                shoulder_ratio = 0.350
                pelvis_ratio = 0.250
        
        current_mode = app_settings.mode.mode_type
        # Update labels
        if current_mode == "Upper-body":
            self.abdomen_ratio_label.setText(f"{abdomen_ratio:.3f}")
            self.chest_ratio_label.setText(f"{chest_ratio:.3f}")
            self.head_ratio_label.setText(f"{head_ratio:.3f}")
            self.shoulder_ratio_label.setText(f"{shoulder_ratio:.3f}")
            self.upperarm_ratio_label.setText(f"{upperarm_ratio:.3f}")
            self.lowerarm_ratio_label.setText(f"{lowerarm_ratio:.3f}")
        else:
            self.trunk_ratio_label.setText(f"{trunk_ratio:.3f}")
            self.pelvis_ratio_label.setText(f"{pelvis_ratio:.3f}")
            self.thigh_ratio_label.setText(f"{thigh_ratio:.3f}")
            self.shank_ratio_label.setText(f"{shank_ratio:.3f}")
            self.foot_ratio_label.setText(f"{foot_ratio:.3f}")
        
    
    def _on_save(self):
        """Save subject info to .subject file in current folder"""
        if not self.current_folder:
            QMessageBox.warning(
                self,
                "No Folder Selected",
                "Please select a subject folder first (e.g., KTY_20260122)"
            )
            return
        
        try:
            # Get subject info
            info = self.get_subject_info()
            
            # Generate filename
            name = info.get('name', 'subject')
            filename = f"{name}.subject"
            filepath = os.path.join(self.current_folder, filename)
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(info, f, indent=2)
            
            self.current_subject_file = filepath
            self.status_label.setText(f"Saved: {filename}")
            self.subject_info_saved.emit(filepath)
            
            QMessageBox.information(
                self,
                "Saved",
                f"Subject information saved to:\n{filepath}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save subject info:\n{str(e)}"
            )
    
    def _on_load(self):
        """Load subject info from .subject file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Subject Information",
            self.current_folder or "",
            "Subject Files (*.subject);;All Files (*.*)"
        )
        
        if filepath:
            self.load_subject_file(filepath)
    
    def _on_apply(self):
        """Apply current values without saving"""
        try:
            info = self.get_subject_info()
            self.subject_info_changed.emit(info)
            self.status_label.setText("Applied to visualization")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Invalid Input",
                f"Please check input values:\n{str(e)}"
            )
    
    def get_subject_info(self) -> dict:
        """Get current subject information as dictionary"""
        try:
            height = float(self.height_input.text())
            shoe_size = float(self.shoe_size_input.text())
        except ValueError:
            raise ValueError("Height and shoe size must be numeric values")
        
        name = self.name_input.text().strip()
        if not name:
            name = "subject"
        
        current_mode = app_settings.mode.mode_type
        if current_mode == "Upper-body":
            # Get ratios from labels
            abdomen_ratio = float(self.abdomen_ratio_label.text())
            chest_ratio = float(self.chest_ratio_label.text())
            head_ratio = float(self.head_ratio_label.text())
            shoulder_ratio = float(self.shoulder_ratio_label.text())
            upperarm_ratio = float(self.upperarm_ratio_label.text())
            lowerarm_ratio = float(self.lowerarm_ratio_label.text())

            return {
                'name': name,
                'height': height,
                'shoe_size': shoe_size,
                'abdomen_ratio': abdomen_ratio,
                'chest_ratio': chest_ratio,
                'head_ratio': head_ratio,
                'shoulder_ratio': shoulder_ratio,
                'upperarm_ratio': upperarm_ratio,
                'lowerarm_ratio': lowerarm_ratio
            }
        else:
            # Get ratios from labels
            trunk_ratio = float(self.trunk_ratio_label.text())
            pelvis_ratio = float(self.pelvis_ratio_label.text())
            thigh_ratio = float(self.thigh_ratio_label.text())
            shank_ratio = float(self.shank_ratio_label.text())
            foot_ratio = float(self.foot_ratio_label.text())
            
            return {
                'name': name,
                'height': height,
                'shoe_size': shoe_size,
                'trunk_ratio': trunk_ratio,
                'pelvis_ratio': pelvis_ratio,
                'thigh_ratio': thigh_ratio,
                'shank_ratio': shank_ratio,
                'foot_ratio': foot_ratio
            }
    
    def set_subject_info(self, info: dict):
        """Set subject information from dictionary"""
        self.name_input.setText(info.get('name', ''))
        self.height_input.setText(str(info.get('height', 170.0)))
        self.shoe_size_input.setText(str(info.get('shoe_size', 270.0)))
        
        current_mode = app_settings.mode.mode_type
        if current_mode == "Upper-body":
            # Update ratio labels
            self.abdomen_ratio_label.setText(f"{info.get('abdomen_ratio', 0.190):.3f}")
            self.chest_ratio_label.setText(f"{info.get('chest_ratio', 0.150):.3f}")
            self.head_ratio_label.setText(f"{info.get('head_ratio', 0.100):.3f}")
            self.shoulder_ratio_label.setText(f"{info.get('shoulder_ratio', 0.350):.3f}")
            self.upperarm_ratio_label.setText(f"{info.get('upperarm_ratio', 0.186):.3f}")
            self.lowerarm_ratio_label.setText(f"{info.get('lowerarm_ratio', 0.146):.3f}")
        else:
            # Update ratio labels
            self.trunk_ratio_label.setText(f"{info.get('trunk_ratio', 0.288):.3f}")
            self.pelvis_ratio_label.setText(f"{info.get('pelvis_ratio', 0.250):.3f}")
            self.thigh_ratio_label.setText(f"{info.get('thigh_ratio', 0.232):.3f}")
            self.shank_ratio_label.setText(f"{info.get('shank_ratio', 0.246):.3f}")
            self.foot_ratio_label.setText(f"{info.get('foot_ratio', 0.152):.3f}")
        
        self.status_label.setText(f"Loaded: {info.get('name', 'subject')}")
    
    def load_subject_file(self, filepath: str):
        """Load subject info from .subject file"""
        try:
            with open(filepath, 'r') as f:
                info = json.load(f)
            
            self.set_subject_info(info)
            self.current_subject_file = filepath
            self.current_folder = os.path.dirname(filepath)
            
            # Emit signal
            self.subject_info_changed.emit(info)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load subject info:\n{str(e)}"
            )
    
    def set_current_folder(self, folder: str):
        """Set the current subject folder"""
        self.current_folder = folder
        
        # Try to auto-load .subject file if it exists
        if folder and os.path.isdir(folder):
            # Look for .subject files in this folder
            subject_files = [f for f in os.listdir(folder) if f.endswith('.subject')]
            
            if subject_files:
                # Load the first .subject file found
                filepath = os.path.join(folder, subject_files[0])
                self.load_subject_file(filepath)
            else:
                # Extract subject name from folder (e.g., KTY_20260122 -> KTY)
                folder_name = os.path.basename(folder)
                subject_name = folder_name.split('_')[0] if '_' in folder_name else folder_name
                self.name_input.setText(subject_name)
                self.status_label.setText(f"Folder: {folder_name} (no .subject file)")
