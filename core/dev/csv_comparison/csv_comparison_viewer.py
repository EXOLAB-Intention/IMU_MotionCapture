import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QWidget,
)


class CsvComparisonViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Column Comparison")
        self.resize(1200, 760)

        self.df1 = None
        self.df2 = None
        self.time1 = None
        self.time2 = None
        self.path1 = None
        self.path2 = None

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        controls = QGroupBox("CSV Selection")
        controls_layout = QGridLayout(controls)

        self.btn_open_csv1 = QPushButton("Open IMU CSV")
        self.btn_open_csv1.clicked.connect(lambda: self._open_csv(1))
        self.lbl_csv1_path = QLabel("IMU CSV: not loaded")

        self.btn_open_csv2 = QPushButton("Open Marker CSV")
        self.btn_open_csv2.clicked.connect(lambda: self._open_csv(2))
        self.lbl_csv2_path = QLabel("Marker CSV: not loaded")

        self.combo_csv1_col = QComboBox()
        self.combo_csv1_col.currentIndexChanged.connect(self._update_plot)
        self.btn_reverse_csv1 = QPushButton("Reverse")
        self.btn_reverse_csv1.setCheckable(True)
        self.btn_reverse_csv1.toggled.connect(self._update_plot)

        self.combo_csv2_col = QComboBox()
        self.combo_csv2_col.currentIndexChanged.connect(self._update_plot)
        self.btn_reverse_csv2 = QPushButton("Reverse")
        self.btn_reverse_csv2.setCheckable(True)
        self.btn_reverse_csv2.toggled.connect(self._update_plot)

        controls_layout.addWidget(self.btn_open_csv1, 0, 0)
        controls_layout.addWidget(self.lbl_csv1_path, 0, 1)
        controls_layout.addWidget(QLabel("IMU column:"), 1, 0)
        controls_layout.addWidget(self.combo_csv1_col, 1, 1)
        controls_layout.addWidget(self.btn_reverse_csv1, 1, 2)

        controls_layout.addWidget(self.btn_open_csv2, 2, 0)
        controls_layout.addWidget(self.lbl_csv2_path, 2, 1)
        controls_layout.addWidget(QLabel("Marker column:"), 3, 0)
        controls_layout.addWidget(self.combo_csv2_col, 3, 1)
        controls_layout.addWidget(self.btn_reverse_csv2, 3, 2)

        layout.addWidget(controls)

        fig_container = QWidget()
        fig_layout = QVBoxLayout(fig_container)
        self.figure = Figure(figsize=(9, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        fig_layout.addWidget(self.toolbar)
        fig_layout.addWidget(self.canvas)
        layout.addWidget(fig_container, 1)

        self.lbl_status = QLabel("Load two CSV files and select a column from each file.")
        layout.addWidget(self.lbl_status)

        self.lbl_metrics = QLabel("RMSE: -, MAE: -")
        layout.addWidget(self.lbl_metrics)

        self.setCentralWidget(root)

    def _open_csv(self, target: int):
        selected, _ = QFileDialog.getOpenFileName(
            self,
            f"Open CSV {target}",
            str(Path.cwd()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not selected:
            return

        try:
            df, time_values, available_cols = self._load_csv_with_time(selected)
        except Exception as exc:
            QMessageBox.critical(self, "CSV Load Error", str(exc))
            return

        if target == 1:
            self.df1 = df
            self.time1 = time_values
            self.path1 = selected
            self.lbl_csv1_path.setText(f"IMU CSV: {selected}")
            self._populate_combo(self.combo_csv1_col, available_cols)
        else:
            self.df2 = df
            self.time2 = time_values
            self.path2 = selected
            self.lbl_csv2_path.setText(f"Marker CSV: {selected}")
            self._populate_combo(self.combo_csv2_col, available_cols)

        self._check_time_consistency()
        self._update_plot()

    def _load_csv_with_time(self, filepath: str):
        df = pd.read_csv(filepath)
        if df.shape[1] < 2:
            raise ValueError(
                "CSV must have at least 2 columns: first column for time and one data column."
            )

        time_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if time_raw.isna().all():
            raise ValueError(
                "The first column (time) is not numeric. Please check CSV format."
            )

        if time_raw.isna().any():
            raise ValueError(
                "The first column (time) has invalid values. Please clean the CSV first."
            )

        data_columns = list(df.columns[1:])
        if not data_columns:
            raise ValueError("No data columns found after the first time column.")

        return df, time_raw.to_numpy(dtype=float), data_columns

    def _populate_combo(self, combo: QComboBox, columns):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(columns)
        combo.blockSignals(False)

    def _check_time_consistency(self):
        if self.time1 is None or self.time2 is None:
            return

        if len(self.time1) != len(self.time2):
            self.lbl_status.setText(
                f"Warning: time length mismatch (CSV1={len(self.time1)}, CSV2={len(self.time2)}). Plot uses CSV 1 time."
            )
            return

        if not np.allclose(self.time1, self.time2, rtol=0, atol=1e-9):
            self.lbl_status.setText("Warning: time values differ between CSV 1 and CSV 2. Plot uses CSV 1 time.")
            return

        self.lbl_status.setText("Time columns are consistent. Select columns to compare.")

    def _get_numeric_column(self, df: pd.DataFrame, column_name: str):
        values = pd.to_numeric(df[column_name], errors="coerce")
        if values.isna().all():
            raise ValueError(f"Selected column '{column_name}' is not numeric.")
        return values.to_numpy(dtype=float)

    def _compute_error_metrics(self, y1: np.ndarray, y2: np.ndarray):
        valid_mask = ~np.isnan(y1) & ~np.isnan(y2)
        if not np.any(valid_mask):
            return None, None, 0

        diff = y1[valid_mask] - y2[valid_mask]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.max(np.abs(diff)))
        return rmse, mae, int(np.sum(valid_mask))

    def _update_plot(self):
        self.ax.clear()

        if self.df1 is None or self.df2 is None:
            self.ax.set_title("Load two CSV files")
            self.ax.set_xlabel("time")
            self.ax.set_ylabel("value")
            self.lbl_metrics.setText("RMSE: -, MAE: -")
            self.canvas.draw_idle()
            return

        col1 = self.combo_csv1_col.currentText()
        col2 = self.combo_csv2_col.currentText()
        if not col1 or not col2:
            self.ax.set_title("Select one column from each CSV")
            self.ax.set_xlabel("time")
            self.ax.set_ylabel("value")
            self.lbl_metrics.setText("RMSE: -, MAE: -")
            self.canvas.draw_idle()
            return

        try:
            y1 = self._get_numeric_column(self.df1, col1)
            y2 = self._get_numeric_column(self.df2, col2)
        except Exception as exc:
            self.lbl_status.setText(f"Plot error: {exc}")
            self.ax.set_title("Column conversion error")
            self.lbl_metrics.setText("RMSE: -, MAE: -")
            self.canvas.draw_idle()
            return

        n1 = min(len(self.time1), len(y1))
        n2 = min(len(self.time1), len(y2))
        n = min(n1, n2)

        if n == 0:
            self.lbl_status.setText("No data to plot.")
            self.ax.set_title("Empty data")
            self.lbl_metrics.setText("RMSE: -, MAE: -")
            self.canvas.draw_idle()
            return

        x = self.time1[:n]
        y1_plot = y1[:n]
        y2_plot = y2[:n]
        imu_reversed = self.btn_reverse_csv1.isChecked()
        marker_reversed = self.btn_reverse_csv2.isChecked()

        if imu_reversed:
            y1_plot = -y1_plot
        if marker_reversed:
            y2_plot = -y2_plot

        imu_label = f"IMU: {col1}" + (" (reversed)" if imu_reversed else "")
        marker_label = f"Marker: {col2}" + (" (reversed)" if marker_reversed else "")
        self.ax.plot(x, y1_plot, label=imu_label, linewidth=1.8)
        self.ax.plot(x, y2_plot, label=marker_label, linewidth=1.8)

        rmse, mae, valid_count = self._compute_error_metrics(y1_plot, y2_plot)
        if valid_count == 0:
            self.lbl_metrics.setText("RMSE: -, MAE: - (no valid paired samples)")
        else:
            self.lbl_metrics.setText(
                f"RMSE: {rmse:.6f}, MAE: {mae:.6f} (valid samples: {valid_count})"
            )

        self.ax.set_xlabel("time")
        self.ax.set_ylabel("value")
        self.ax.set_title("CSV Column Comparison")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    viewer = CsvComparisonViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
