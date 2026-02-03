"""
Data I/O utilities for IMU motion capture
"""
import numpy as np
import pandas as pd
from typing import Tuple


def load_csv_data(filepath: str) -> Tuple[np.ndarray, dict, dict]:
    """
    Load IMU data from CSV file for both feet
    
    Supports format with columns like:
    - L_FOOT_IMU_AccX, L_FOOT_IMU_AccY, L_FOOT_IMU_AccZ
    - R_FOOT_IMU_AccX, R_FOOT_IMU_AccY, R_FOOT_IMU_AccZ
    - L_FOOT_IMU_GyrX, L_FOOT_IMU_GyrY, L_FOOT_IMU_GyrZ
    - R_FOOT_IMU_GyrX, R_FOOT_IMU_GyrY, R_FOOT_IMU_GyrZ
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        timestamps: (N,) array
        accelerations: dict with 'L_FOOT' and 'R_FOOT' keys, (N, 3) arrays
        gyroscopes: dict with 'L_FOOT' and 'R_FOOT' keys, (N, 3) arrays
    """
    df = pd.read_csv(filepath)
    
    accelerations = {}
    gyroscopes = {}
    
    # Load both feet
    for foot in ['L_FOOT', 'R_FOOT']:
        foot_accel_cols = [col for col in df.columns if f'{foot}_IMU_Acc' in col][:3]
        foot_gyro_cols = [col for col in df.columns if f'{foot}_IMU_Gyr' in col][:3]
        
        if len(foot_accel_cols) == 3 and len(foot_gyro_cols) == 3:
            print(f"✓ Found {foot} foot IMU data")
            print(f"  Accel columns: {foot_accel_cols}")
            print(f"  Gyro columns: {foot_gyro_cols}")
            accelerations[foot] = df[foot_accel_cols].values
            gyroscopes[foot] = df[foot_gyro_cols].values
        else:
            print(f"⚠ {foot} data not found in CSV")
    
    if not accelerations or not gyroscopes:
        raise ValueError("No foot data found in CSV file")
    
    # Try to get timestamps
    time_col = next((col for col in df.columns if 'time' in col.lower()), None)
    timestamps = df[time_col].values if time_col else np.arange(len(df))
    
    return timestamps, accelerations, gyroscopes
