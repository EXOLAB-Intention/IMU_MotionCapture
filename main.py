"""
IMU Motion Capture System - Main Entry Point

A GUI application for processing and analyzing 3D motion capture data
from wearable IMU sensors (Xsens MTi-630)

Features:
- Import raw IMU data (quaternion, acceleration, gyroscope)
- Calibration with T-pose or N-pose
- 3D joint angle computation (hip, knee, ankle)
- Trunk orientation analysis
- Foot contact detection
- Velocity estimation
- Interactive 3D visualization
- Time-series graph plotting
- Batch processing support

Author: EXO Lab, KAIST
Version: 1.0.0
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui.main_window import MainWindow


def main():
    """Main application entry point"""
    
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("IMU Motion Capture")
    app.setOrganizationName("EXO Lab")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
