import os
import sys
import ctypes
import csv
from pathlib import Path

import h5py
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)


class H5Viewer(QMainWindow):
    def __init__(self, initial_file: str = None):
        super().__init__()
        self.setWindowTitle("H5 Viewer")
        self.resize(1200, 760)

        self.current_file = None
        self._h5 = None
        self._build_ui()

        if initial_file:
            self.load_h5_file(initial_file)

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        top_bar = QHBoxLayout()
        self.open_btn = QPushButton("Open H5")
        self.open_btn.clicked.connect(self.open_file_dialog)

        self.export_btn = QPushButton("Export Selected to CSV")
        self.export_btn.clicked.connect(self.export_selected_to_csv)
        self.export_btn.setEnabled(False)

        self.export_mocap_btn = QPushButton("Export Mocap Angle Data")
        self.export_mocap_btn.clicked.connect(self.export_mocap_angle_data)
        self.export_mocap_btn.setEnabled(False)

        self.export_abs_mocap_foot_btn = QPushButton("Export Absolute Mocap Angle Data (kin_q)")
        self.export_abs_mocap_foot_btn.clicked.connect(self.export_absolute_mocap_foot_angle_data)
        self.export_abs_mocap_foot_btn.setEnabled(False)

        self.export_abs_mocap_marker_btn = QPushButton("Export Absolute Mocap Angle Data (marker)")
        self.export_abs_mocap_marker_btn.clicked.connect(self.export_absolute_mocap_marker_angle_data)
        self.export_abs_mocap_marker_btn.setEnabled(False)

        self.path_label = QLabel("No H5 file loaded")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        top_bar.addWidget(self.open_btn)
        top_bar.addWidget(self.export_btn)
        top_bar.addWidget(self.export_mocap_btn)
        top_bar.addWidget(self.export_abs_mocap_foot_btn)
        top_bar.addWidget(self.export_abs_mocap_marker_btn)
        top_bar.addWidget(self.path_label, 1)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type", "Info"])
        self.tree.setColumnWidth(0, 320)
        self.tree.setColumnWidth(1, 110)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(self.detail)
        splitter.setSizes([520, 680])

        layout.addLayout(top_bar)
        layout.addWidget(splitter, 1)

        self.status = QLabel("")
        layout.addWidget(self.status)

        self.setCentralWidget(root)

    def _close_file(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def closeEvent(self, event):
        self._close_file()
        super().closeEvent(event)

    def _get_open_candidates(self, filepath: str):
        abs_path = os.path.abspath(filepath)
        candidates = [filepath, abs_path]

        if os.name == "nt":
            try:
                rel_path = os.path.relpath(abs_path, os.getcwd())
                if not rel_path.startswith("..") and os.path.exists(rel_path):
                    candidates.append(rel_path)
            except Exception:
                pass

            try:
                short_buffer = ctypes.create_unicode_buffer(32768)
                if ctypes.windll.kernel32.GetShortPathNameW(abs_path, short_buffer, len(short_buffer)):
                    short_path = short_buffer.value
                    if short_path:
                        candidates.append(short_path)
            except Exception:
                pass

        unique = []
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return unique

    def _open_h5(self, filepath: str):
        open_errors = []
        for candidate in self._get_open_candidates(filepath):
            try:
                return h5py.File(candidate, "r"), candidate
            except OSError as e:
                open_errors.append(f"{candidate} -> {e}")
        raise OSError("Unable to open HDF5 file. " + " | ".join(open_errors))

    def open_file_dialog(self):
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 File",
            str(Path.cwd()),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*.*)",
        )
        if selected:
            self.load_h5_file(selected)

    def load_h5_file(self, filepath: str):
        self._close_file()
        self.tree.clear()
        self.detail.clear()
        self.export_btn.setEnabled(False)
        self.export_mocap_btn.setEnabled(False)
        self.export_abs_mocap_foot_btn.setEnabled(False)
        self.export_abs_mocap_marker_btn.setEnabled(False)

        try:
            h5_file, used_path = self._open_h5(filepath)
            self._h5 = h5_file
            self.current_file = filepath
            self.path_label.setText(f"Loaded: {filepath}")
            self._populate_tree()
            self.status.setText(f"Opened with path: {used_path}")
        except Exception as e:
            self.current_file = None
            self.path_label.setText("No H5 file loaded")
            self.status.setText("Open failed")
            QMessageBox.critical(self, "H5 Open Error", str(e))

    def _populate_tree(self):
        if self._h5 is None:
            return

        group_count = 0
        dataset_count = 0

        def add_children(parent_item, group_obj, path_prefix=""):
            nonlocal group_count, dataset_count
            for key in sorted(group_obj.keys()):
                obj = group_obj[key]
                obj_path = f"{path_prefix}/{key}" if path_prefix else key

                if isinstance(obj, h5py.Group):
                    info = f"{len(obj.keys())} children"
                    item = QTreeWidgetItem(parent_item, [key, "Group", info])
                    item.setData(0, Qt.UserRole, obj_path)
                    item.setData(1, Qt.UserRole, "group")
                    group_count += 1
                    add_children(item, obj, obj_path)
                elif isinstance(obj, h5py.Dataset):
                    shape_txt = self._shape_text(obj)
                    item = QTreeWidgetItem(parent_item, [key, "Dataset", shape_txt])
                    item.setData(0, Qt.UserRole, obj_path)
                    item.setData(1, Qt.UserRole, "dataset")
                    dataset_count += 1

        for root_key in sorted(self._h5.keys()):
            root_obj = self._h5[root_key]
            if isinstance(root_obj, h5py.Group):
                root_info = f"{len(root_obj.keys())} children"
                root_item = QTreeWidgetItem(self.tree, [root_key, "Group", root_info])
                root_item.setData(0, Qt.UserRole, root_key)
                root_item.setData(1, Qt.UserRole, "group")
                root_item.setExpanded(True)
                group_count += 1
                add_children(root_item, root_obj, root_key)
            elif isinstance(root_obj, h5py.Dataset):
                shape_txt = self._shape_text(root_obj)
                item = QTreeWidgetItem(self.tree, [root_key, "Dataset", shape_txt])
                item.setData(0, Qt.UserRole, root_key)
                item.setData(1, Qt.UserRole, "dataset")
                dataset_count += 1

        self.status.setText(f"Groups: {group_count}, Datasets: {dataset_count}")

    def _shape_text(self, dataset: h5py.Dataset) -> str:
        if dataset.shape is None or len(dataset.shape) == 0:
            return "scalar"
        return "x".join(str(d) for d in dataset.shape)

    def on_selection_changed(self):
        if self._h5 is None:
            return

        items = self.tree.selectedItems()
        self.export_btn.setEnabled(len(items) > 0)
        has_single_kin_q = self._get_selected_single_kin_q_path() is not None
        has_single_marker = self._get_selected_single_marker_path() is not None
        self.export_mocap_btn.setEnabled(has_single_kin_q)
        self.export_abs_mocap_foot_btn.setEnabled(has_single_kin_q)
        self.export_abs_mocap_marker_btn.setEnabled(has_single_marker)
        if not items:
            return

        item = items[0]
        obj_path = item.data(0, Qt.UserRole)
        item_type = item.data(1, Qt.UserRole)

        if not obj_path:
            return

        try:
            obj = self._h5[obj_path]
            if item_type == "group":
                self.detail.setPlainText(self._describe_group(obj_path, obj))
            elif item_type == "dataset":
                self.detail.setPlainText(self._describe_dataset(obj_path, obj))
        except Exception as e:
            self.detail.setPlainText(f"Failed to read selected node:\n{e}")

    def _sanitize_name(self, text: str) -> str:
        safe = text.replace("/", "__").replace("\\", "__")
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in safe)
        return safe.strip("_") or "dataset"

    def _find_kin_q_groups(self, group_obj, prefix):
        kin_q_paths = []
        for key in sorted(group_obj.keys()):
            child = group_obj[key]
            child_path = f"{prefix}/{key}" if prefix else key
            if isinstance(child, h5py.Group):
                if key == "kin_q":
                    kin_q_paths.append(child_path)
                kin_q_paths.extend(self._find_kin_q_groups(child, child_path))
        return kin_q_paths

    def _resolve_kin_q_path_from_item(self, item):
        if self._h5 is None or item is None:
            return None

        obj_path = item.data(0, Qt.UserRole)
        item_type = item.data(1, Qt.UserRole)
        if not obj_path or item_type != "group" or obj_path not in self._h5:
            return None

        obj = self._h5[obj_path]
        if not isinstance(obj, h5py.Group):
            return None

        if obj_path.split("/")[-1] == "kin_q":
            return obj_path

        kin_q_paths = self._find_kin_q_groups(obj, obj_path)
        if len(kin_q_paths) == 1:
            return kin_q_paths[0]

        return None

    def _get_selected_single_kin_q_path(self):
        selected_items = self.tree.selectedItems()
        if len(selected_items) != 1:
            return None
        return self._resolve_kin_q_path_from_item(selected_items[0])

    def _is_mocap_marker_path(self, path: str) -> bool:
        parts = path.split("/") if path else []
        return len(parts) >= 2 and parts[-2:] == ["mocap", "marker"]

    def _find_marker_groups(self, group_obj, prefix):
        marker_paths = []
        for key in sorted(group_obj.keys()):
            child = group_obj[key]
            child_path = f"{prefix}/{key}" if prefix else key
            if isinstance(child, h5py.Group):
                if self._is_mocap_marker_path(child_path):
                    marker_paths.append(child_path)
                marker_paths.extend(self._find_marker_groups(child, child_path))
        return marker_paths

    def _resolve_marker_path_from_item(self, item):
        if self._h5 is None or item is None:
            return None

        obj_path = item.data(0, Qt.UserRole)
        item_type = item.data(1, Qt.UserRole)
        if not obj_path or item_type != "group" or obj_path not in self._h5:
            return None

        obj = self._h5[obj_path]
        if not isinstance(obj, h5py.Group):
            return None

        if self._is_mocap_marker_path(obj_path):
            return obj_path

        marker_paths = self._find_marker_groups(obj, obj_path)
        if len(marker_paths) == 1:
            return marker_paths[0]

        return None

    def _get_selected_single_marker_path(self):
        selected_items = self.tree.selectedItems()
        if len(selected_items) != 1:
            return None
        return self._resolve_marker_path_from_item(selected_items[0])

    def _resolve_marker_group(self, marker_group: h5py.Group, marker_name: str):
        candidates = [marker_name, marker_name.lower(), marker_name.upper()]
        for candidate in candidates:
            if candidate in marker_group:
                return marker_group[candidate], candidate

        lower_lookup = {str(key).lower(): key for key in marker_group.keys()}
        resolved = lower_lookup.get(marker_name.lower())
        if resolved is not None:
            return marker_group[resolved], resolved

        raise KeyError(f"Missing marker: {marker_name}")

    def _get_marker_xyz(self, marker_group: h5py.Group, marker_name: str, expected_len: int = None):
        marker, resolved_name = self._resolve_marker_group(marker_group, marker_name)

        if not isinstance(marker, h5py.Group):
            raise TypeError(f"{resolved_name} is not a marker group")

        coords = []
        for axis in ("x", "y", "z"):
            if axis not in marker or not isinstance(marker[axis], h5py.Dataset):
                raise KeyError(f"Missing dataset: {resolved_name}/{axis}")

            values = np.asarray(marker[axis][()]).reshape(-1).astype(float)
            if expected_len is not None and len(values) > expected_len:
                values = values[:expected_len]
            elif expected_len is not None and len(values) != expected_len:
                raise ValueError(
                    f"Length mismatch for {resolved_name}/{axis}: "
                    f"expected={expected_len}, data={len(values)}"
                )
            coords.append(values)

        if expected_len is None:
            target_len = min(len(values) for values in coords)
            coords = [values[:target_len] for values in coords]

        return self._marker_xyz_to_imu_view_coordinates(np.column_stack(coords))

    def _marker_xyz_to_imu_view_coordinates(self, xyz):
        """Convert raw mocap xyz coordinates into the IMU/global visualization axes."""
        arr = np.asarray(xyz, dtype=float)
        transformed = np.empty_like(arr, dtype=float)
        transformed[:, 0] = -arr[:, 1]
        transformed[:, 1] = arr[:, 0]
        transformed[:, 2] = arr[:, 2]
        return transformed

    def _normalize_vectors(self, vectors, eps: float = 1e-12):
        arr = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        normalized = np.full(arr.shape, np.nan, dtype=float)
        valid = np.isfinite(arr).all(axis=1) & (norms[:, 0] > eps)
        normalized[valid] = arr[valid] / norms[valid]
        return normalized, valid

    def _segment_frame_from_three_markers(self, anterior, posterior, lateral, side: str):
        local_x, x_valid = self._normalize_vectors(anterior - posterior)
        lateral_raw = lateral - 0.5 * (anterior + posterior)
        if side == "right":
            lateral_raw = -lateral_raw

        local_y_raw, y_raw_valid = self._normalize_vectors(lateral_raw)
        local_z, z_valid = self._normalize_vectors(np.cross(local_x, local_y_raw))
        local_y, y_valid = self._normalize_vectors(np.cross(local_z, local_x))

        rotations = np.stack([local_x, local_y, local_z], axis=2)
        valid = x_valid & y_raw_valid & z_valid & y_valid
        rotations[~valid, :, :] = np.nan
        return rotations

    def _foot_frame_from_markers(self, medial_ankle, lateral_ankle, heel, toe, side: str):
        local_x, x_valid = self._normalize_vectors(toe - heel)
        local_y_raw = lateral_ankle - medial_ankle
        if side == "right":
            local_y_raw = -local_y_raw

        local_y_raw, y_raw_valid = self._normalize_vectors(local_y_raw)
        local_z, z_valid = self._normalize_vectors(np.cross(local_x, local_y_raw))
        local_y, y_valid = self._normalize_vectors(np.cross(local_z, local_x))

        rotations = np.stack([local_x, local_y, local_z], axis=2)
        valid = x_valid & y_raw_valid & z_valid & y_valid
        rotations[~valid, :, :] = np.nan
        return rotations

    def _rotation_matrices_to_euler_xyz(self, rotations):
        rotations = np.asarray(rotations, dtype=float)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError("rotations must have shape (N, 3, 3)")

        angles = np.full((rotations.shape[0], 3), np.nan, dtype=float)
        valid_rows = np.isfinite(rotations).all(axis=(1, 2))
        if not np.any(valid_rows):
            return angles

        idxs = np.where(valid_rows)[0]
        for idx in idxs:
            R = rotations[idx]
            cy = np.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])
            if cy > 1e-8:
                angle_x = np.arctan2(R[2, 1], R[2, 2])
                angle_y = np.arctan2(-R[2, 0], cy)
                angle_z = np.arctan2(R[1, 0], R[0, 0])
            else:
                angle_x = np.arctan2(-R[1, 2], R[1, 1])
                angle_y = np.arctan2(-R[2, 0], cy)
                angle_z = 0.0

            angles[idx] = np.degrees([angle_x, angle_y, angle_z])

        return angles

    def _global_axis_relative_rotations_from_initial(self, rotations):
        """
        Express each marker-derived segment frame change around the global axes.

        rotations[idx] maps a segment's local XYZ frame into the mocap/global
        frame.  R(t) @ R(0).T gives the rotation that carries the initial
        global-frame segment orientation to the current global-frame segment
        orientation, so the resulting Euler XYZ values describe what is seen
        from the global axes rather than from the segment's local axes.
        """
        rotations = np.asarray(rotations, dtype=float)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError("rotations must have shape (N, 3, 3)")

        relative = np.full(rotations.shape, np.nan, dtype=float)
        valid_rows = np.isfinite(rotations).all(axis=(1, 2))
        valid_indices = np.where(valid_rows)[0]
        if len(valid_indices) == 0:
            return relative

        initial_idx = 0 if valid_rows[0] else valid_indices[0]
        initial_rotation = rotations[initial_idx]

        for idx in valid_indices:
            relative[idx] = rotations[idx] @ initial_rotation.T

        return relative

    def _relative_rotations_to_initial_global(self, rotations):
        return self._global_axis_relative_rotations_from_initial(rotations)

    def _build_marker_absolute_angle_columns(self, marker_path: str, marker_group: h5py.Group):
        marker_names = [
            "rthi", "rathi", "rpthi",
            "lthi", "lathi", "lpthi",
            "rtib", "ratib", "rptib",
            "ltib", "latib", "lptib",
            "rmank", "rank", "rhee", "rtoe",
            "lmank", "lank", "lhee", "ltoe",
        ]

        markers = {
            marker_name: self._get_marker_xyz(marker_group, marker_name)
            for marker_name in marker_names
        }
        expected_len = min(len(values) for values in markers.values())
        markers = {
            marker_name: values[:expected_len]
            for marker_name, values in markers.items()
        }

        time_values = self._find_trial_timestamp(marker_path, expected_len)
        if time_values is None:
            time_values = np.arange(expected_len, dtype=float) / 100.0
        time_values = np.asarray(time_values, dtype=float)

        segment_frames = [
            (
                "R Thigh",
                self._segment_frame_from_three_markers(
                    markers["rathi"], markers["rpthi"], markers["rthi"], "right"
                ),
            ),
            (
                "L Thigh",
                self._segment_frame_from_three_markers(
                    markers["lathi"], markers["lpthi"], markers["lthi"], "left"
                ),
            ),
            (
                "R Shank",
                self._segment_frame_from_three_markers(
                    markers["ratib"], markers["rptib"], markers["rtib"], "right"
                ),
            ),
            (
                "L Shank",
                self._segment_frame_from_three_markers(
                    markers["latib"], markers["lptib"], markers["ltib"], "left"
                ),
            ),
            (
                "R Foot",
                self._foot_frame_from_markers(
                    markers["rmank"], markers["rank"], markers["rhee"], markers["rtoe"], "right"
                ),
            ),
            (
                "L Foot",
                self._foot_frame_from_markers(
                    markers["lmank"], markers["lank"], markers["lhee"], markers["ltoe"], "left"
                ),
            ),
        ]

        headers = ["time"]
        columns = [time_values.tolist()]
        invalid_counts = {}

        for segment_name, rotations in segment_frames:
            relative_rotations = self._global_axis_relative_rotations_from_initial(rotations)
            angles = self._rotation_matrices_to_euler_xyz(relative_rotations)
            if len(angles) > 0 and np.isfinite(angles[0]).all():
                angles[0] = 0.0
            invalid_counts[segment_name] = int(np.isnan(angles).any(axis=1).sum())
            for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                headers.append(f"{segment_name} {axis_name}")
                columns.append(angles[:, axis_idx].tolist())

        return headers, columns, invalid_counts

    def _dataset_value_to_columns(self, name: str, value):
        if np.isscalar(value):
            return [name], [[value]]

        arr = np.asarray(value)
        if arr.ndim == 0:
            return [name], [[arr.item()]]

        if arr.ndim == 1:
            return [name], [arr.tolist()]

        rows_2d = arr.reshape(arr.shape[0], -1)
        headers = [f"{name}_{i}" for i in range(rows_2d.shape[1])]
        columns = [rows_2d[:, i].tolist() for i in range(rows_2d.shape[1])]
        return headers, columns

    def _subtract_initial_value(self, values):
        if not values:
            return values

        try:
            arr = np.asarray(values, dtype=float)
            normalized = arr - arr[0]
            return normalized.tolist()
        except Exception:
            return values

    def _get_required_1d_series(self, group: h5py.Group, key: str, expected_len: int):
        if key not in group:
            raise KeyError(f"Missing dataset: {key}")

        ds = group[key]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"{key} is not a dataset")

        values = np.asarray(ds[()]).reshape(-1)
        if len(values) != expected_len:
            raise ValueError(
                f"Length mismatch for {key}: time={expected_len}, data={len(values)}"
            )

        return values.astype(float)

    def _normalize_time_for_imu_compare(self, time_values):
        """Return timestamps starting at 0 seconds, matching FileHandler H5 import."""
        timestamps = np.asarray(time_values, dtype=float).reshape(-1)
        if len(timestamps) == 0:
            return timestamps

        timestamps = timestamps - timestamps[0]
        if len(timestamps) > 1:
            median_dt = np.nanmedian(np.diff(timestamps))
            # H5 common/time is stored in milliseconds; kin_q/time may already be seconds.
            if np.isfinite(median_dt) and median_dt > 1.0:
                timestamps = timestamps / 1000.0

        return timestamps

    def _initial_relative(self, values):
        """Subtract the initial value so frame 0 matches IMU relative-angle export."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        if len(arr) == 0:
            return arr
        return arr - arr[0]

    def export_absolute_mocap_foot_angle_data(self):
        if self._h5 is None:
            QMessageBox.warning(self, "No File", "Open an H5 file first.")
            return

        kin_q_path = self._get_selected_single_kin_q_path()
        if kin_q_path is None:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Select exactly one group: either kin_q itself or a parent(trial) containing exactly one kin_q.",
            )
            return

        kin_q_group = self._h5[kin_q_path]
        if "time" not in kin_q_group or not isinstance(kin_q_group["time"], h5py.Dataset):
            QMessageBox.warning(self, "Missing time", f"time dataset not found in:\n{kin_q_path}")
            return

        try:
            time_values = np.asarray(kin_q_group["time"][()]).reshape(-1).astype(float)
        except Exception as e:
            QMessageBox.critical(self, "Read Error", f"Failed to read time dataset:\n{e}")
            return

        if len(time_values) == 0:
            QMessageBox.warning(self, "Empty time", "time dataset is empty.")
            return

        try:
            pelvis_tilt = self._get_required_1d_series(kin_q_group, "pelvis_tilt", len(time_values))
            hip_flexion_r = self._get_required_1d_series(kin_q_group, "hip_flexion_r", len(time_values))
            knee_angle_r = self._get_required_1d_series(kin_q_group, "knee_angle_r", len(time_values))
            ankle_angle_r = self._get_required_1d_series(kin_q_group, "ankle_angle_r", len(time_values))
            hip_flexion_l = self._get_required_1d_series(kin_q_group, "hip_flexion_l", len(time_values))
            knee_angle_l = self._get_required_1d_series(kin_q_group, "knee_angle_l", len(time_values))
            ankle_angle_l = self._get_required_1d_series(kin_q_group, "ankle_angle_l", len(time_values))
        except Exception as e:
            QMessageBox.critical(self, "Read Error", str(e))
            self.status.setText("Absolute mocap foot export failed")
            return

        # User-defined sagittal chain sign convention.
        # Thigh = pelvis_tilt + hip_flexion
        # Shank = pelvis_tilt + hip_flexion - knee_angle
        # Foot  = pelvis_tilt + hip_flexion - knee_angle + ankle_angle
        right_thigh_abs = pelvis_tilt + hip_flexion_r
        left_thigh_abs = pelvis_tilt + hip_flexion_l
        right_shank_abs = pelvis_tilt + hip_flexion_r - knee_angle_r
        left_shank_abs = pelvis_tilt + hip_flexion_l - knee_angle_l
        right_foot_abs = right_shank_abs + ankle_angle_r
        left_foot_abs = left_shank_abs + ankle_angle_l

        # Match graph_view.py IMU segment export format:
        # Timestamp starts at 0 seconds, frame 0 is zero, and sagittal-plane
        # segment angles are placed in the Y column of the XYZ layout.
        timestamps = self._normalize_time_for_imu_compare(time_values)
        segment_series = [
            ("R_Thigh", self._initial_relative(right_thigh_abs)),
            ("L_Thigh", self._initial_relative(left_thigh_abs)),
            ("R_Shank", self._initial_relative(right_shank_abs)),
            ("L_Shank", self._initial_relative(left_shank_abs)),
            ("R_Foot", self._initial_relative(right_foot_abs)),
            ("L_Foot", self._initial_relative(left_foot_abs)),
        ]

        default_name = f"{self._sanitize_name(kin_q_path)}__imu_compare_segments.csv"
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save IMU-comparable kin_q segment angle CSV",
            str(Path.cwd() / default_name),
            "CSV Files (*.csv)",
        )
        if not output_file:
            return

        if not output_file.lower().endswith('.csv'):
            output_file += '.csv'

        headers = ["Timestamp"]
        for segment_name, _ in segment_series:
            headers.extend([f"{segment_name}_X", f"{segment_name}_Y", f"{segment_name}_Z"])

        rows = []
        for i in range(len(timestamps)):
            row = [timestamps[i]]
            for _, values in segment_series:
                row.extend([0.0, values[i], 0.0])
            rows.append(row)

        try:
            with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            self.status.setText("Absolute mocap foot export failed")
            return

        QMessageBox.information(
            self,
            "Export Completed",
            f"Saved CSV:\n{output_file}\n\nRows: {len(rows)}\nColumns: {len(headers)}",
        )

        self.status.setText(
            f"IMU-comparable kin_q segment export: {kin_q_path} -> {Path(output_file).name} ({len(rows)} rows)"
        )

    def export_absolute_mocap_marker_angle_data(self):
        if self._h5 is None:
            QMessageBox.warning(self, "No File", "Open an H5 file first.")
            return

        marker_path = self._get_selected_single_marker_path()
        if marker_path is None:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Select exactly one group: either mocap/marker itself or a parent(trial) containing exactly one mocap/marker.",
            )
            return

        marker_group = self._h5[marker_path]
        try:
            headers, columns, invalid_counts = self._build_marker_absolute_angle_columns(
                marker_path, marker_group
            )
        except Exception as e:
            QMessageBox.critical(self, "Read Error", str(e))
            self.status.setText("Absolute mocap marker export failed")
            return

        default_name = f"{self._sanitize_name(marker_path)}__absolute_marker_segments.csv"
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save absolute marker segment angle CSV",
            str(Path.cwd() / default_name),
            "CSV Files (*.csv)",
        )
        if not output_file:
            return

        if not output_file.lower().endswith('.csv'):
            output_file += '.csv'

        rows = []
        for i in range(len(columns[0])):
            rows.append([col[i] for col in columns])

        try:
            with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            self.status.setText("Absolute mocap marker export failed")
            return

        invalid_summary = [
            f"{name}: {count}" for name, count in invalid_counts.items() if count > 0
        ]
        if invalid_summary:
            QMessageBox.warning(
                self,
                "Export Completed (with warnings)",
                f"Saved CSV:\n{output_file}\n\n"
                f"Rows: {len(rows)}\n"
                f"Columns: {len(headers)}\n"
                f"Frames with invalid angles:\n" + "\n".join(invalid_summary),
            )
        else:
            QMessageBox.information(
                self,
                "Export Completed",
                f"Saved CSV:\n{output_file}\n\nRows: {len(rows)}\nColumns: {len(headers)}",
            )

        self.status.setText(
            f"Absolute mocap marker export: {marker_path} -> {Path(output_file).name} ({len(rows)} rows, {len(headers)} columns)"
        )

    def export_mocap_angle_data(self):
        if self._h5 is None:
            QMessageBox.warning(self, "No File", "Open an H5 file first.")
            return

        kin_q_path = self._get_selected_single_kin_q_path()
        if kin_q_path is None:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Select exactly one group: either kin_q itself or a parent(trial) containing exactly one kin_q.",
            )
            return

        kin_q_group = self._h5[kin_q_path]
        if "time" not in kin_q_group or not isinstance(kin_q_group["time"], h5py.Dataset):
            QMessageBox.warning(self, "Missing time", f"time dataset not found in:\n{kin_q_path}")
            return

        try:
            time_values = np.asarray(kin_q_group["time"][()]).reshape(-1).tolist()
        except Exception as e:
            QMessageBox.critical(self, "Read Error", f"Failed to read time dataset:\n{e}")
            return

        if len(time_values) == 0:
            QMessageBox.warning(self, "Empty time", "time dataset is empty.")
            return

        headers = ["time"]
        columns = [time_values]
        failures = []

        for key in sorted(kin_q_group.keys()):
            if key == "time":
                continue

            ds = kin_q_group[key]
            if not isinstance(ds, h5py.Dataset):
                continue

            try:
                ds_headers, ds_columns = self._dataset_value_to_columns(key, ds[()])
            except Exception as e:
                failures.append(f"{key}: {e}")
                continue

            for ds_header, ds_column in zip(ds_headers, ds_columns):
                if len(ds_column) != len(time_values):
                    failures.append(
                        f"{ds_header}: length mismatch (time={len(time_values)}, data={len(ds_column)})"
                    )
                    continue
                headers.append(ds_header)
                columns.append(self._subtract_initial_value(ds_column))

        if len(headers) == 1:
            QMessageBox.warning(
                self,
                "Export Failed",
                "No kin_q datasets were exported.\n" + "\n".join(failures[:10]),
            )
            self.status.setText("Mocap angle export failed")
            return

        default_name = f"{self._sanitize_name(kin_q_path)}.csv"
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save mocap angle CSV",
            str(Path.cwd() / default_name),
            "CSV Files (*.csv)",
        )
        if not output_file:
            return

        if not output_file.lower().endswith('.csv'):
            output_file += '.csv'

        rows = []
        for i in range(len(time_values)):
            rows.append([col[i] for col in columns])

        try:
            with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            self.status.setText("Mocap angle export failed")
            return

        if failures:
            QMessageBox.warning(
                self,
                "Export Completed (with warnings)",
                f"Saved CSV:\n{output_file}\n\n"
                f"Rows: {len(rows)}\n"
                f"Columns: {len(headers)}\n"
                f"Skipped datasets: {len(failures)}\n\n"
                + "\n".join(failures[:10]),
            )
        else:
            QMessageBox.information(
                self,
                "Export Completed",
                f"Saved CSV:\n{output_file}\n\nRows: {len(rows)}\nColumns: {len(headers)}",
            )

        self.status.setText(
            f"Mocap angle export: {kin_q_path} -> {Path(output_file).name} ({len(rows)} rows, {len(headers)} columns)"
        )

    def _collect_dataset_paths(self, obj_path: str, item_type: str):
        dataset_paths = []
        if item_type == "dataset":
            dataset_paths.append(obj_path)
            return dataset_paths

        if item_type != "group":
            return dataset_paths

        def walk(group_obj, prefix):
            for key in sorted(group_obj.keys()):
                child = group_obj[key]
                child_path = f"{prefix}/{key}" if prefix else key
                if isinstance(child, h5py.Dataset):
                    dataset_paths.append(child_path)
                elif isinstance(child, h5py.Group):
                    walk(child, child_path)

        walk(self._h5[obj_path], obj_path)
        return dataset_paths

    def _dataset_to_rows(self, dataset_path: str, value):
        dataset_name = self._sanitize_name(dataset_path) if dataset_path else 'value'

        if np.isscalar(value):
            return [[value]], [dataset_name]

        arr = np.asarray(value)
        if arr.ndim == 0:
            return [[arr.item()]], [dataset_name]

        if arr.ndim == 1:
            rows = [[v] for v in arr.tolist()]
            return rows, [dataset_name]

        rows_2d = arr.reshape(arr.shape[0], -1)
        headers = [f"{dataset_name}_{i}" for i in range(rows_2d.shape[1])]
        return rows_2d.tolist(), headers

    def _rows_to_columns(self, rows, n_cols):
        if not rows:
            return [[] for _ in range(n_cols)]

        columns = [[] for _ in range(n_cols)]
        for row in rows:
            for col_idx in range(n_cols):
                columns[col_idx].append(row[col_idx])
        return columns

    def _find_trial_context_series(self, dataset_path: str, expected_rows: int):
        """Find common time/loopcnt series in ancestor trial group and return matching columns."""
        if self._h5 is None or expected_rows <= 0:
            return []

        path_parts = dataset_path.split('/')
        if len(path_parts) < 2:
            return []

        for i in range(len(path_parts) - 1, 0, -1):
            ancestor_path = '/'.join(path_parts[:i])
            if ancestor_path not in self._h5:
                continue

            ancestor_obj = self._h5[ancestor_path]
            if not isinstance(ancestor_obj, h5py.Group):
                continue

            if 'common' not in ancestor_obj:
                continue

            common_obj = ancestor_obj['common']
            if not isinstance(common_obj, h5py.Group):
                continue

            context_columns = []
            for key, col_name in [('time', 'timestamp'), ('loopcnt', 'loopcnt')]:
                if key not in common_obj:
                    continue

                ds = common_obj[key]
                if not isinstance(ds, h5py.Dataset):
                    continue

                try:
                    values = np.asarray(ds[()]).reshape(-1)
                except Exception:
                    continue

                if len(values) == expected_rows:
                    context_columns.append((col_name, values.tolist()))

            if context_columns:
                return context_columns

        return []

    def _find_trial_timestamp(self, dataset_path: str, expected_rows: int):
        """Return timestamp series for dataset trial context when available and row count matches."""
        context_columns = self._find_trial_context_series(dataset_path, expected_rows)
        for name, values in context_columns:
            if name == 'timestamp':
                return values
        return None

    def _timestamp_key(self, value):
        """Normalize timestamp value for stable matching across datasets."""
        return round(float(value), 6)

    def _build_dataset_export_columns(self, dataset_path: str):
        dataset = self._h5[dataset_path]
        value = dataset[()]
        rows, headers = self._dataset_to_rows(dataset_path, value)

        columns = self._rows_to_columns(rows, len(headers))

        return headers, columns, len(rows)

    def export_selected_to_csv(self):
        if self._h5 is None:
            QMessageBox.warning(self, "No File", "Open an H5 file first.")
            return

        selected_items = self.tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Select one or more items to export.")
            return

        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save merged CSV",
            str(Path.cwd() / "h5_export_merged.csv"),
            "CSV Files (*.csv)",
        )
        if not output_file:
            return

        if not output_file.lower().endswith('.csv'):
            output_file += '.csv'

        dataset_paths = []
        for item in selected_items:
            obj_path = item.data(0, Qt.UserRole)
            item_type = item.data(1, Qt.UserRole)
            if not obj_path:
                continue
            dataset_paths.extend(self._collect_dataset_paths(obj_path, item_type))

        dataset_paths = sorted(set(dataset_paths))
        if not dataset_paths:
            QMessageBox.information(
                self,
                "Nothing to Export",
                "Selected items contain no datasets.\nSelect dataset nodes or group nodes with datasets.",
            )
            return

        merged_headers = []
        merged_columns = []
        failures = []
        used_headers = set()
        dataset_blocks = []
        global_ts_keys = set()

        for ds_path in dataset_paths:
            try:
                headers, columns, row_count = self._build_dataset_export_columns(ds_path)
                ts_values = self._find_trial_timestamp(ds_path, row_count)
                if ts_values is None:
                    failures.append(f"{ds_path}: timestamp not found or length mismatch")
                    continue

                ts_keys = [self._timestamp_key(v) for v in ts_values]
                global_ts_keys.update(ts_keys)
                dataset_blocks.append((ds_path, headers, columns, ts_keys))
            except Exception as e:
                failures.append(f"{ds_path}: {e}")

        if not dataset_blocks:
            QMessageBox.warning(
                self,
                "Export Failed",
                "No datasets could be exported with valid timestamps.\n"
                + "\n".join(failures[:10]),
            )
            self.status.setText("CSV export failed")
            return

        sorted_ts = sorted(global_ts_keys)
        ts_index = {key: idx for idx, key in enumerate(sorted_ts)}

        for _, headers, columns, ts_keys in dataset_blocks:
            for header, src_col in zip(headers, columns):
                unique_header = header
                suffix = 1
                while unique_header in used_headers:
                    unique_header = f"{header}_{suffix}"
                    suffix += 1
                used_headers.add(unique_header)

                aligned_col = [""] * len(sorted_ts)
                for local_idx, ts_key in enumerate(ts_keys):
                    if local_idx >= len(src_col):
                        continue
                    aligned_col[ts_index[ts_key]] = src_col[local_idx]

                merged_headers.append(unique_header)
                merged_columns.append(aligned_col)

        merged_headers = ['timestamp'] + merged_headers
        merged_columns = [sorted_ts] + merged_columns

        merged_rows = []
        for row_idx in range(len(sorted_ts)):
            merged_rows.append([column[row_idx] for column in merged_columns])

        with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(merged_headers)
            writer.writerows(merged_rows)

        success = len(dataset_blocks)

        if failures:
            QMessageBox.warning(
                self,
                "Export Completed",
                f"Merged CSV saved to:\n{output_file}\n\n"
                f"Exported datasets: {success}\n"
                f"Failed datasets: {len(failures)}\n\n"
                + "\n".join(failures[:10]),
            )
        else:
            QMessageBox.information(
                self,
                "Export Completed",
                f"Merged CSV saved to:\n{output_file}\n\n"
                f"Exported datasets: {success}",
            )

        self.status.setText(
            f"CSV export: merged {success} dataset(s), {len(failures)} failed"
        )

    def _describe_group(self, path: str, group: h5py.Group) -> str:
        lines = [
            f"Path: {path}",
            "Type: Group",
            f"Children: {len(group.keys())}",
            "",
            "Child names:",
        ]
        for name in sorted(group.keys()):
            child = group[name]
            kind = "Group" if isinstance(child, h5py.Group) else "Dataset"
            lines.append(f"- {name} ({kind})")

        if len(group.attrs) > 0:
            lines += ["", "Attributes:"]
            for k in sorted(group.attrs.keys()):
                val = group.attrs[k]
                lines.append(f"- {k}: {self._short_repr(val)}")

        return "\n".join(lines)

    def _describe_dataset(self, path: str, dataset: h5py.Dataset) -> str:
        shape_txt = self._shape_text(dataset)
        lines = [
            f"Path: {path}",
            "Type: Dataset",
            f"Shape: {shape_txt}",
            f"DType: {dataset.dtype}",
            "",
            "Preview:",
        ]

        try:
            arr = dataset[()]
            preview = self._preview_array(arr)
            lines.append(preview)
        except Exception as e:
            lines.append(f"<failed to read dataset: {e}>")

        if len(dataset.attrs) > 0:
            lines += ["", "Attributes:"]
            for k in sorted(dataset.attrs.keys()):
                val = dataset.attrs[k]
                lines.append(f"- {k}: {self._short_repr(val)}")

        return "\n".join(lines)

    def _preview_array(self, value):
        if np.isscalar(value):
            return str(value)

        arr = np.asarray(value)
        if arr.ndim == 0:
            return str(arr.item())

        max_items = 40
        flat = arr.reshape(-1)
        shown = flat[:max_items]
        txt = np.array2string(shown, threshold=max_items, edgeitems=20)
        if flat.size > max_items:
            txt += f"\n... ({flat.size - max_items} more values)"
        return txt

    def _short_repr(self, value):
        try:
            arr = np.asarray(value)
            if arr.ndim == 0:
                return str(arr.item())
            return f"array(shape={arr.shape}, dtype={arr.dtype})"
        except Exception:
            return str(value)


def main():
    app = QApplication(sys.argv)
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    viewer = H5Viewer(initial_file=initial)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
