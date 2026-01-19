"""
Menu bar component for the main application window
"""
from PyQt5.QtWidgets import QMenuBar, QAction, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal, QObject


class MenuBar(QObject):
    """Menu bar with File, View, and other menus"""
    
    # Signals for menu actions
    import_file_requested = pyqtSignal(str)  # filepath
    open_file_requested = pyqtSignal(str)  # filepath
    save_requested = pyqtSignal()
    save_as_requested = pyqtSignal(str)  # filepath
    exit_requested = pyqtSignal()
    
    # Calibration signals
    load_calibration_requested = pyqtSignal(str)  # calibration filepath
    save_calibration_requested = pyqtSignal(str)  # calibration filepath
    perform_calibration_requested = pyqtSignal()  # perform calibration on current data
    
    def __init__(self, menubar: QMenuBar):
        super().__init__()
        self.menubar = menubar
        self._create_menus()
    
    def _create_menus(self):
        """Create all menu items"""
        self._create_file_menu()
        self._create_view_menu()
        self._create_process_menu()
        self._create_help_menu()
    
    def _create_file_menu(self):
        """Create File menu"""
        file_menu = self.menubar.addMenu('&File')
        
        # Open (load processed file)
        open_action = QAction('&Open...', self.menubar)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open processed motion capture file')
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)
        
        # Import (load raw data)
        import_action = QAction('&Import...', self.menubar)
        import_action.setShortcut('Ctrl+I')
        import_action.setStatusTip('Import raw IMU data')
        import_action.triggered.connect(self._on_import)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        # Save
        save_action = QAction('&Save', self.menubar)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save processed data')
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)
        
        # Save As
        save_as_action = QAction('Save &As...', self.menubar)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.setStatusTip('Save processed data as new file')
        save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # Export
        export_menu = file_menu.addMenu('&Export')
        
        export_csv_action = QAction('Export to CSV...', self.menubar)
        export_csv_action.setStatusTip('Export data to CSV format')
        export_menu.addAction(export_csv_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction('E&xit', self.menubar)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self._on_exit)
        file_menu.addAction(exit_action)
    
    def _create_view_menu(self):
        """Create View menu"""
        view_menu = self.menubar.addMenu('&View')
        
        # Toggle Navigator
        toggle_navigator = QAction('&Navigator', self.menubar)
        toggle_navigator.setCheckable(True)
        toggle_navigator.setChecked(True)
        toggle_navigator.setStatusTip('Show/hide navigator panel')
        view_menu.addAction(toggle_navigator)
        
        # Toggle Notes
        toggle_notes = QAction('N&otes', self.menubar)
        toggle_notes.setCheckable(True)
        toggle_notes.setChecked(True)
        toggle_notes.setStatusTip('Show/hide notes panel')
        view_menu.addAction(toggle_notes)
        
        view_menu.addSeparator()
        
        # Toggle 3D View
        toggle_3d = QAction('&3D Visualization', self.menubar)
        toggle_3d.setCheckable(True)
        toggle_3d.setChecked(True)
        toggle_3d.setStatusTip('Show/hide 3D visualization')
        view_menu.addAction(toggle_3d)
        
        # Toggle Graph View
        toggle_graph = QAction('&Graph View', self.menubar)
        toggle_graph.setCheckable(True)
        toggle_graph.setChecked(True)
        toggle_graph.setStatusTip('Show/hide graph view')
        view_menu.addAction(toggle_graph)
        
        view_menu.addSeparator()
        
        # Reset Layout
        reset_layout = QAction('&Reset Layout', self.menubar)
        reset_layout.setStatusTip('Reset window layout to default')
        view_menu.addAction(reset_layout)
    
    def _create_process_menu(self):
        """Create Process menu"""
        process_menu = self.menubar.addMenu('&Process')
        
        # Load Calibration
        load_calib_action = QAction('&Load Calibration...', self.menubar)
        load_calib_action.setShortcut('Ctrl+L')
        load_calib_action.setStatusTip('Load calibration from .cal file')
        load_calib_action.triggered.connect(self._on_load_calibration)
        process_menu.addAction(load_calib_action)
        
        # Save Calibration
        save_calib_action = QAction('&Save Calibration...', self.menubar)
        save_calib_action.setStatusTip('Save current calibration to .cal file')
        save_calib_action.triggered.connect(self._on_save_calibration)
        process_menu.addAction(save_calib_action)
        
        process_menu.addSeparator()
        
        # Perform Calibration
        perform_calib_action = QAction('&Perform Calibration', self.menubar)
        perform_calib_action.setShortcut('Ctrl+K')
        perform_calib_action.setStatusTip('Perform calibration on current data (use entire duration)')
        perform_calib_action.triggered.connect(self._on_perform_calibration)
        process_menu.addAction(perform_calib_action)
        
        # Set Calibration Period
        calibration_action = QAction('Set Calibration &Period...', self.menubar)
        calibration_action.setStatusTip('Define specific calibration pose time range')
        process_menu.addAction(calibration_action)
        
        process_menu.addSeparator()
        
        # Configure Settings
        settings_action = QAction('&Settings...', self.menubar)
        settings_action.setShortcut('Ctrl+,')
        settings_action.setStatusTip('Configure processing settings')
        process_menu.addAction(settings_action)
        
        process_menu.addSeparator()
        
        # Process Current
        process_current = QAction('Process &Data', self.menubar)
        process_current.setShortcut('F5')
        process_current.setStatusTip('Process current motion capture data')
        process_menu.addAction(process_current)
        
        # Batch Process
        batch_process = QAction('&Batch Process...', self.menubar)
        batch_process.setStatusTip('Process multiple files with same settings')
        process_menu.addAction(batch_process)
    
    def _create_help_menu(self):
        """Create Help menu"""
        help_menu = self.menubar.addMenu('&Help')
        
        # Documentation
        docs_action = QAction('&Documentation', self.menubar)
        docs_action.setShortcut('F1')
        docs_action.setStatusTip('Open documentation')
        help_menu.addAction(docs_action)
        
        help_menu.addSeparator()
        
        # About
        about_action = QAction('&About', self.menubar)
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _on_open(self):
        """Handle Open action"""
        filepath, _ = QFileDialog.getOpenFileName(
            None,
            "Open Processed File",
            "",
            "Motion Capture Files (*.mcp);;All Files (*)"
        )
        if filepath:
            self.open_file_requested.emit(filepath)
    
    def _on_import(self):
        """Handle Import action"""
        filepath, _ = QFileDialog.getOpenFileName(
            None,
            "Import Raw IMU Data",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;Data Files (*.dat);;All Files (*)"
        )
        if filepath:
            self.import_file_requested.emit(filepath)
    
    def _on_save(self):
        """Handle Save action"""
        self.save_requested.emit()
    
    def _on_save_as(self):
        """Handle Save As action"""
        filepath, _ = QFileDialog.getSaveFileName(
            None,
            "Save As",
            "",
            "Motion Capture Files (*.mcp);;All Files (*)"
        )
        if filepath:
            self.save_as_requested.emit(filepath)
    
    def _on_exit(self):
        """Handle Exit action"""
        self.exit_requested.emit()
    
    def _on_load_calibration(self):
        """Handle Load Calibration action"""
        filepath, _ = QFileDialog.getOpenFileName(
            None,
            "Load Calibration File",
            "",
            "Calibration Files (*.cal);;All Files (*)"
        )
        if filepath:
            self.load_calibration_requested.emit(filepath)
    
    def _on_save_calibration(self):
        """Handle Save Calibration action"""
        filepath, _ = QFileDialog.getSaveFileName(
            None,
            "Save Calibration File",
            "",
            "Calibration Files (*.cal);;All Files (*)"
        )
        if filepath:
            self.save_calibration_requested.emit(filepath)
    
    def _on_perform_calibration(self):
        """Handle Perform Calibration action"""
        self.perform_calibration_requested.emit()
    
    def _on_about(self):
        """Handle About action"""
        QMessageBox.about(
            None,
            "About IMU Motion Capture",
            "<h3>IMU Motion Capture System</h3>"
            "<p>Version 1.0</p>"
            "<p>3D motion analysis using wearable IMU sensors</p>"
            "<p>Xsens MTi-630 | Lower body kinematics</p>"
        )
