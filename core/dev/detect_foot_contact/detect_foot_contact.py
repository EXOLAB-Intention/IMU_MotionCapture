"""
Development file for detect_foot_contact function implementation
Simple version with functions only (no classes)

Usage:
    python core/dev/detect_foot_contact.py
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from pathlib import Path
import sys

from data_io import load_csv_data
from visualization import plot_results

# Import for type hints when testing
try:
    from core.imu_data import MotionCaptureData
except ImportError:
    MotionCaptureData = None


def compute_magnitude_mean(magnitude: np.ndarray, frame1: int, frame2: int) -> float:
    """
    Compute average magnitude between frame1 and frame2
    
    Args:
        magnitude: Magnitude array (N,)
        frame1: Start frame index (inclusive)
        frame2: End frame index (inclusive)
        
    Returns:
        Average magnitude value in the range [frame1, frame2]
    """
    return np.mean(magnitude[frame1:frame2+1])


def detect_foot_contact(
    data,  # MotionCaptureData
    accel_threshold: float,
    accel_threshold_weight: float,
    gyro_threshold: float,
    window_size: int = 10,
    min_contact_duration: int = 40
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Detect foot contact events from IMU data
    
    Algorithm:
        1. Calculate acceleration and gyroscope magnitudes
        2. Apply moving average smoothing
        3. Detect contact: low acceleration AND low gyroscope
        4. Remove isolated short events
        5. Find gait_start_frame and gait_end_frame
        6. Apply post-processing rules to ensure valid gait cycle
    
    Args:
        data: MotionCaptureData object with imu_data
        accel_threshold: Center acceleration magnitude threshold (m/s²)
        accel_threshold_weight: Weight for threshold range (0-1)
        gyro_threshold: Gyroscope magnitude threshold (rad/s)
        window_size: Moving average window size
        min_contact_duration: Minimum samples for contact event
        
    Returns:
        Tuple of (gait_start_frame, gait_end_frame, foot_contact_right, foot_contact_left)
        where:
            - gait_start_frame (int): Frame where first foot leaves double support
            - gait_end_frame (int): Frame where both feet return to contact at end
            - foot_contact_right: Modified boolean array (no simultaneous contact/non-contact)
            - foot_contact_left: Modified boolean array (no simultaneous contact/non-contact)
    """
    # Extract accelerations and gyroscopes from data
    accel_right = data.imu_data['foot_right'].accelerations
    accel_left = data.imu_data['foot_left'].accelerations
    gyro_right = data.imu_data['foot_right'].gyroscopes
    gyro_left = data.imu_data['foot_left'].gyroscopes
    
    # Detect for right foot
    foot_contact_right = _detect_contact_single(
        accel_right, gyro_right,
        accel_threshold, accel_threshold_weight, gyro_threshold,
        window_size, min_contact_duration
    )
    
    # Detect for left foot
    foot_contact_left = _detect_contact_single(
        accel_left, gyro_left,
        accel_threshold, accel_threshold_weight, gyro_threshold,
        window_size, min_contact_duration
    )
    
    # Post-processing: find gait frames and enforce constraints
    gait_start_frame, gait_end_frame, foot_contact_right, foot_contact_left = \
        _process_gait_cycle(foot_contact_right, foot_contact_left)
    
    return (gait_start_frame, gait_end_frame, foot_contact_right, foot_contact_left)


def _process_gait_cycle(
    foot_contact_right: np.ndarray,
    foot_contact_left: np.ndarray
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Process foot contact arrays to extract gait cycle and enforce constraints
    
    Rules:
        1. Find gait_start_frame: first frame where one foot leaves initial double support
        2. Find gait_end_frame: first frame where both feet return to contact at the end
        3. Between gait_start and gait_end:
           - Remove simultaneous contact: keep only the foot that contacted earlier
           - Remove simultaneous non-contact: keep the foot that was last in contact
    
    Args:
        foot_contact_right: Right foot contact boolean array
        foot_contact_left: Left foot contact boolean array
        
    Returns:
        Tuple of (gait_start_frame, gait_end_frame, modified_right, modified_left)
    """
    N = len(foot_contact_right)
    right = foot_contact_right.copy().astype(bool)
    left = foot_contact_left.copy().astype(bool)
    
    # Step 1: Find gait_start_frame
    # Find first frame where both are True
    both_contact = right & left
    first_both_idx = np.where(both_contact)[0]
    
    if len(first_both_idx) == 0:
        # No double support at start, use first frame
        gait_start_frame = 0
    else:
        # Find first frame after initial double support where one foot leaves
        start_region = first_both_idx[0]
        # Look for first break in double support after start_region
        gait_start_frame = None
        for i in range(start_region, N):
            if not (right[i] and left[i]):
                gait_start_frame = i
                break
        if gait_start_frame is None:
            gait_start_frame = N - 1
    
    # Step 2: Find gait_end_frame
    # Find last section where both are True continuously
    gait_end_frame = None
    for i in range(N - 1, -1, -1):
        if right[i] and left[i]:
            # Check if this is the start of a final double support region
            # by checking if before this point there was single support
            if i == 0 or not (right[i-1] and left[i-1]):
                gait_end_frame = i
                break
    
    if gait_end_frame is None:
        gait_end_frame = N - 1
    
    # Step 3: Apply post-processing rules between gait_start_frame and gait_end_frame (exclusive)
    if gait_start_frame < gait_end_frame:
        right, left = _enforce_gait_constraints(right, left, gait_start_frame, gait_end_frame - 1)
    
    # Step 4: Ensure both feet are True from gait_end_frame to the end
    right[gait_end_frame:] = True
    left[gait_end_frame:] = True
    
    return gait_start_frame, gait_end_frame, right, left


def _enforce_gait_constraints(
    right: np.ndarray,
    left: np.ndarray,
    start_frame: int,
    end_frame: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforce gait constraints between start_frame and end_frame
    
    Rule 1: Remove simultaneous contact
        If both feet contact at same frame, keep only the foot that contacted earlier
    Rule 2: Remove simultaneous non-contact
        If both feet not in contact at same frame, keep the foot that was last in contact
    
    Args:
        right: Right foot contact array
        left: Left foot contact array
        start_frame: Gait cycle start
        end_frame: Gait cycle end
        
    Returns:
        Modified (right, left) arrays
    """
    right = right.copy()
    left = left.copy()
    
    # Process each frame in the gait cycle
    for i in range(start_frame, end_frame + 1):
        both_contact = right[i] and left[i]
        both_notcontact = (not right[i]) and (not left[i])
        
        if both_contact:
            # Rule 1: Remove simultaneous contact
            # Keep the foot that was already in single support just before this frame.
            last_single = None
            for j in range(i - 1, start_frame - 1, -1):
                if right[j] != left[j]:
                    last_single = 'right' if right[j] else 'left'
                    break

            if last_single == 'right':
                left[i] = False
            elif last_single == 'left':
                right[i] = False
            else:
                # Fallback: keep right if no prior single support found
                left[i] = False
        
        elif both_notcontact:
            # Rule 2: Remove simultaneous non-contact
            # Keep the foot that was in contact most recently
            right_last_contact = -1
            left_last_contact = -1
            
            for j in range(i - 1, -1, -1):
                if right[j] and right_last_contact == -1:
                    right_last_contact = j
                if left[j] and left_last_contact == -1:
                    left_last_contact = j
                if right_last_contact != -1 and left_last_contact != -1:
                    break
            
            # Set the foot that was last in contact to be in contact
            if right_last_contact > left_last_contact:
                right[i] = True
            elif left_last_contact > right_last_contact:
                left[i] = True
            else:
                # Both never contacted or contacted at same frame - keep right
                right[i] = True
    
    return right, left


def _detect_contact_single(
    accelerations: np.ndarray,
    gyroscopes: np.ndarray,
    accel_threshold: float,
    accel_threshold_weight: float,
    gyro_threshold: float,
    window_size: int,
    min_contact_duration: int
) -> np.ndarray:
    """Detect foot contact for a single foot"""
    
    # Calculate magnitudes
    accel_magnitude = np.linalg.norm(accelerations, axis=1)
    gyro_magnitude = np.linalg.norm(gyroscopes, axis=1)
    
    
    # Smoothing method 1
    # Apply smoothing (method 1: direct moving average)
    accel_smooth = _moving_average(accel_magnitude, window_size)
    gyro_smooth = _moving_average(gyro_magnitude, window_size)

    # Calculate min/max thresholds from center threshold and weight
    accel_min_threshold = accel_threshold * (1 - accel_threshold_weight)
    accel_max_threshold = accel_threshold * (1 + accel_threshold_weight)

    # Detect contact - acceleration must be between min and max thresholds
    accel_contact = (accel_smooth >= accel_min_threshold) & (accel_smooth <= accel_max_threshold)
    gyro_contact = gyro_smooth <= gyro_threshold
    foot_contact = accel_contact & gyro_contact
    
    """
    # Smoothing method 2
    # Apply smoothing (method 2: using abs diff from threshold for accel)
    accel_diff_from_threshold = abs(accel_magnitude - accel_threshold)
    accel_smooth = _moving_average(accel_diff_from_threshold, window_size)
    gyro_smooth = _moving_average(gyro_magnitude, window_size)
    
    # Detect contact - acceleration must be between min and max thresholds
    accel_contact = accel_smooth <= accel_threshold * accel_threshold_weight
    gyro_contact = gyro_smooth <= gyro_threshold
    foot_contact = accel_contact & gyro_contact
    """
    
    # Remove noise
    # foot_contact = _remove_noise(foot_contact, min_contact_duration)
    foot_contact = _remove_noise_falsegapFirst(foot_contact, min_contact_duration)
    
    return foot_contact


def _moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average smoothing"""
    if window_size <= 1:
        return data
    
    # Use pandas rolling for better edge handling
    smoothed = pd.Series(data).rolling(window=window_size, center=True, min_periods=1).mean().values
    return smoothed


def _remove_noise(contact: np.ndarray, min_duration: int) -> np.ndarray:
    """Remove noise by 2-step process: remove short True regions, then fill short False gaps"""
    cleaned = contact.copy()

    # Step 1: Remove short isolated True regions
    diff = np.diff(cleaned.astype(int))
    true_starts = np.where(diff == 1)[0] + 1   # 0->1 transition
    true_ends = np.where(diff == -1)[0] + 1    # 1->0 transition

    # Correct boundaries for Step 1
    if len(cleaned) > 0 and cleaned[0]:
        true_starts = np.r_[0, true_starts]
    if len(cleaned) > 0 and cleaned[-1]:
        true_ends = np.r_[true_ends, len(cleaned)]

    # Remove short True regions (set to False)
    for start, end in zip(true_starts, true_ends):
        if end - start < min_duration:
            cleaned[start:end] = False

    # Step 2: Fill short False gaps between True regions
    diff = np.diff(cleaned.astype(int))
    true_starts = np.where(diff == 1)[0] + 1
    true_ends = np.where(diff == -1)[0] + 1

    # Correct boundaries for Step 2
    if len(cleaned) > 0 and cleaned[0]:
        true_starts = np.r_[0, true_starts]
    if len(cleaned) > 0 and cleaned[-1]:
        true_ends = np.r_[true_ends, len(cleaned)]

    # Fill gaps between True regions
    for i in range(len(true_ends) - 1):
        gap_start = true_ends[i]
        gap_end = true_starts[i + 1]
        gap_duration = gap_end - gap_start

        if gap_duration < min_duration:
            cleaned[gap_start:gap_end] = True

    return cleaned

def _remove_noise_falsegapFirst(contact: np.ndarray, min_duration: int) -> np.ndarray:
    """Remove noise by 2-step process: fill short False gaps first, then remove short True regions"""
    cleaned = contact.copy()

    # Step 1: Fill short False gaps between True regions
    diff = np.diff(cleaned.astype(int))
    true_starts = np.where(diff == 1)[0] + 1   # 0->1 transition
    true_ends = np.where(diff == -1)[0] + 1    # 1->0 transition

    # Correct boundaries for Step 1
    if len(cleaned) > 0 and cleaned[0]:
        true_starts = np.r_[0, true_starts]
    if len(cleaned) > 0 and cleaned[-1]:
        true_ends = np.r_[true_ends, len(cleaned)]

    # Fill gaps between True regions
    for i in range(len(true_ends) - 1):
        gap_start = true_ends[i]
        gap_end = true_starts[i + 1]
        gap_duration = gap_end - gap_start

        if gap_duration < min_duration:
            cleaned[gap_start:gap_end] = True

    # Step 2: Remove short isolated True regions
    diff = np.diff(cleaned.astype(int))
    true_starts = np.where(diff == 1)[0] + 1   # 0->1 transition
    true_ends = np.where(diff == -1)[0] + 1    # 1->0 transition

    # Correct boundaries for Step 2
    if len(cleaned) > 0 and cleaned[0]:
        true_starts = np.r_[0, true_starts]
    if len(cleaned) > 0 and cleaned[-1]:
        true_ends = np.r_[true_ends, len(cleaned)]

    # Remove short True regions (set to False)
    for start, end in zip(true_starts, true_ends):
        if end - start < min_duration:
            cleaned[start:end] = False

    return cleaned


def compute_reference_values(
    accelerations: dict,
    timestamps: np.ndarray,
    frame_start: int = 0,
    frame_end: Optional[int] = None
) -> Tuple[Dict[str, float], float, float]:
    """Compute acceleration and gyroscope thresholds from IMU data per foot

    Args:
        accelerations: dict with 'L_FOOT' and/or 'R_FOOT' keys containing (N, 3) arrays
        timestamps: time array for reference
        frame_start: start frame index (inclusive) for threshold calculation
        frame_end: end frame index (inclusive) for threshold calculation; None uses last frame
        
    Returns:
        Tuple of (accel_thresholds_per_foot, accel_threshold_weight, gyro_threshold)
    """

    # Clamp frame range to available samples
    total_frames = len(timestamps)
    end_idx_global = total_frames - 1 if frame_end is None else min(frame_end, total_frames - 1)
    start_idx_global = max(0, min(frame_start, end_idx_global))

    accel_reference_value: Dict[str, float] = {}
    for foot in ['L_FOOT', 'R_FOOT']:
        if foot in accelerations:
            accel_mag = np.linalg.norm(accelerations[foot], axis=1)
            if accel_mag.size == 0:
                accel_reference_value[foot] = 10.0
            else:
                local_end = min(end_idx_global, accel_mag.shape[0] - 1)
                local_start = max(0, min(start_idx_global, local_end))
                accel_reference_value[foot] = compute_magnitude_mean(accel_mag, local_start, local_end)

    # Fallback if no foot data found
    if not accel_reference_value:
        accel_reference_value['L_FOOT'] = 10.0
        accel_reference_value['R_FOOT'] = 10.0
    return (accel_reference_value)


def main():
    """Test with CSV file or sample data"""
    csv_file = str(Path(__file__).parents[3] / "data" / "JJY_20260119" / "JJY_20260119_095340_walk_01_processed.csv")
    
    # Frame range for contact ratio calculation
    calculation_frame_start = 2000
    calculation_frame_end = 14000  # None for end of data
    
    # Frame range for calibration (threshold calculation)
    calibration_frame_start = 0
    calibration_frame_end = 1500  # None for end of data

    try:
        print(f"\nLoading: {csv_file}\n")
        timestamps, accelerations, gyroscopes = load_csv_data(csv_file)
        print(f"\nLoaded {len(timestamps)} samples\n")
        
        # Use default thresholds
        accel_thresholds = {
            'L_FOOT': 10.0,
            'R_FOOT': 10.0,
        }
        val_accel_threshold_weight = 0.2
        val_gyro_threshold = 0.5
        val_window_size = 10
        val_min_contact_duration = 40
        
        # Detect foot contact for both feet using modified detect_foot_contact logic
        # First, detect raw contact for each foot
        foot_contact_raw = {}
        for foot in accelerations.keys():
            a_thr = accel_thresholds.get(foot)
            foot_contact_raw[foot] = _detect_contact_single(
                accelerations=accelerations[foot],
                gyroscopes=gyroscopes[foot],
                accel_threshold=a_thr,
                accel_threshold_weight=val_accel_threshold_weight,
                gyro_threshold=val_gyro_threshold,
                window_size=val_window_size,
                min_contact_duration=val_min_contact_duration
            )
        
        # Process gait cycle to get gait frames and modified contact arrays
        gait_start_frame, gait_end_frame, foot_contact_right, foot_contact_left = \
            _process_gait_cycle(foot_contact_raw['R_FOOT'], foot_contact_raw['L_FOOT'])
        
        # Prepare foot_contact dict for visualization
        foot_contact = {
            'R_FOOT': foot_contact_right,
            'L_FOOT': foot_contact_left
        }
        
        print(f"Gait Start Frame: {gait_start_frame}")
        print(f"Gait End Frame: {gait_end_frame}")
        
        # Visualize
        if calculation_frame_end is None:
            calculation_frame_end = len(timestamps) - 1
        
        plot_results(timestamps, accelerations, gyroscopes, foot_contact,
                    accel_threshold=accel_thresholds,
                    accel_threshold_weight=val_accel_threshold_weight,
                    gyro_threshold=val_gyro_threshold,
                    frame_start=calculation_frame_start,
                    frame_end=calculation_frame_end)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
