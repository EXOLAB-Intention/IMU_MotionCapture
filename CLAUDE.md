# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMU Motion Capture: a PyQt5 GUI for processing lower-body and upper-body motion capture data from Xsens MTi-630 IMU sensors. Lower-body mode uses 7 sensors (trunk, thigh/shank/foot x L/R); upper-body mode adds pelvis, chest, head, and arm sensors. Written in Python, developed by EXO Lab at KAIST. The project language is mixed Korean/English.

## Commands

```bash
pip install -r requirements.txt   # Install dependencies
python main.py                    # Run the GUI application
```

No test framework is configured. Manual testing is done via `python main.py` and sample data in `data/` (gitignored). Test scripts: `test_npose_calibration.py`, `test_angle_calculation.py`, `verify_angles.py`.

## Architecture

```
main.py                  # Entry point (PyQt5 app)
config/settings.py       # Global config singleton (AppSettings, ModeConfig)
core/                    # Processing pipeline (no UI dependencies)
  imu_data.py            # Data structures: IMUSensorData, MotionCaptureData, JointAngles
  calibration.py         # N-pose/T-pose calibration, walking-direction-aware desired quats
  kinematics.py          # Quaternion math, joint angle computation, foot contact detection
  data_processor.py      # Pipeline orchestrator (incl. gait data)
  dev/                   # Development/experimental modules
    detect_foot_contact/ # Standalone foot contact detection package
    body_segment_ratio/  # ML body segment ratio prediction from anthropometrics
file_io/file_handler.py  # CSV import (lower+upper body), .mcp/.cal/.subject save/load
ui/                      # PyQt5 GUI (signal-slot pattern)
  main_window.py         # Top-level window, signal routing hub, mode switching
  menu_bar.py            # Menus with signals (import, calibration, process)
  navigator.py           # Folder-tree file browser with color-coded file types
  main_view.py           # Central view (3D + graphs + timeline), mode toggle toolbar
  graph_view.py          # Matplotlib joint angle plots (mode-aware: lower/upper body)
  visualization_3d.py    # 3D skeleton rendering (lower + upper body, moving grid)
  subject_info_panel.py  # Subject info input (height/shoe size, segment ratio prediction)
  time_bar.py            # Timeline slider and range selection
  notes.py               # Session notes editor
data/                    # Sample/test data (gitignored, not tracked)
  PJS_20260119/          # Subject datasets with raw CSV + processed files
  HEB_20260126/          # Each subfolder = one capture session
  HWB_20260122/          #   contains .csv (raw), _processed.csv, .mcp, .cal, .subject
  HWB_20260209/
  JJY_20260119/
  KTY_20260122/
```

### Data Flow

Raw CSV → `FileHandler.import_raw_data()` → `MotionCaptureData` → `CalibrationProcessor.calibrate()` + `.apply_to_data()` → `KinematicsProcessor.compute_joint_angles()` → `JointAngles` → save as `.mcp`

### Signal-Slot Wiring

All inter-component communication uses PyQt5 signals. `MainWindow._connect_signals()` is the central wiring point. UI components emit signals (e.g., `MenuBar.import_file_requested(str)`), MainWindow routes them to core/file_io handlers.

## Critical Conventions

### Quaternion Format
- **[w, x, y, z]** (scalar-first) throughout the codebase
- Hamilton multiplication convention
- Calibration uses RIGHT multiplication: `q_offset = conj(q_calib) * q_desired`, applied as `q_segment = q_measured * q_offset`
- Joint angles: `q_rel = conj(q_proximal) * q_distal`

### Coordinate Frames
- Ground: X=forward, Y=left, Z=up
- Trunk IMU: x-up, y-right, z-forward
- Leg IMUs: x-up, y-left, z-backward
- Y180 rotation (`R_Y180 = [[-1,0,0],[0,1,0],[0,0,-1]]`) applied for sensor-to-body conversion

### Data Structures
- Quaternions: `np.ndarray` shape `(N, 4)`
- Vectors (accel/gyro/angles): `np.ndarray` shape `(N, 3)`
- Joint angles in degrees, ZYX Euler sequence, columns = [flexion, abduction, rotation]
- Timestamps derived from `LoopCnt` CSV column, normalized to start at 0.0

### File Formats
- `.csv`/`.txt`/`.dat`: Raw IMU data (columns: `TrunkIMU_QuatW`, `L_THIGH_IMU_AccX`, etc.)
- `.mcp`: Processed data (JSON with numpy arrays as lists)
- `.cal`: Calibration offsets (JSON with version compatibility)
- `.subject`: Subject anthropometric info (JSON with height, shoe size, segment ratios)

All data files live under `data/` (gitignored). Each subject session is a subfolder (e.g., `data/PJS_20260119/`).

## Branching

Feature branches use `feature/<name>` pattern (e.g., `feature/kinematics`). All merges via pull requests to `main`.

## Unimplemented Features (marked TODO in code)

- Signal filtering and resampling in `data_processor.py`
- CSV export in `file_handler.py`
