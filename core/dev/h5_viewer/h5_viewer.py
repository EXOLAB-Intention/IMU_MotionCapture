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

        self.export_btn = QPushButton("Export Selected CSV")
        self.export_btn.clicked.connect(self.export_selected_to_csv)
        self.export_btn.setEnabled(False)

        self.path_label = QLabel("No H5 file loaded")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        top_bar.addWidget(self.open_btn)
        top_bar.addWidget(self.export_btn)
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
