# CSV Comparison Viewer

Simple PyQt5 tool to compare one selected column from each of two CSV files on the same time axis.

## Features

- Import two CSV files independently
- Select one column from each CSV file
- Use the first column of each CSV as `time`
- Plot both selected columns on the same graph with CSV 1 time as x-axis
- Warn if the two time columns are not identical

## Expected CSV format

- First column: time (numeric)
- Remaining columns: data columns to choose from

## Run

```bash
python core/dev/csv_comparison/csv_comparison_viewer.py
```
