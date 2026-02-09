"""
Visualization utilities for foot contact detection
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from typing import Optional, Dict, Union


def plot_results(
    timestamps: np.ndarray,
    accelerations: dict,
    gyroscopes: dict,
    foot_contact: dict,
    accel_threshold: Union[float, Dict[str, float]] = 10.0,
    accel_threshold_weight: float = 0.1,
    gyro_threshold: float = 0.1,
    frame_start: int = 0,
    frame_end: Optional[int] = None):
    """Plot IMU data with interactive checkboxes for foot selection
    Args:
        timestamps: (N,) array
        accelerations: dict with 'L_FOOT' and/or 'R_FOOT' keys
        gyroscopes: dict with 'L_FOOT' and/or 'R_FOOT' keys
        foot_contact: dict with 'L_FOOT' and/or 'R_FOOT' keys
        accel_threshold: Center acceleration threshold
        accel_threshold_weight: Weight for threshold range
        gyro_threshold: Gyroscope threshold
        frame_start: Start frame for contact ratio calculation
        frame_end: End frame for contact ratio calculation (None = end of data)
    """
    
    if frame_end is None:
        frame_end = len(timestamps) - 1
    
    def _get_accel_band(foot: str) -> tuple:
        base = accel_threshold
        if isinstance(accel_threshold, dict):
            base = accel_threshold.get(foot, next(iter(accel_threshold.values()), 10.0))
        return (base * (1 - accel_threshold_weight), base * (1 + accel_threshold_weight))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Foot Contact Detection Analysis', fontsize=16, fontweight='bold')
    
    # Create 3 subplots
    ax_accel = plt.subplot(3, 1, 1)
    ax_gyro = plt.subplot(3, 1, 2)
    ax_contact = plt.subplot(3, 1, 3)
    
    # Foot selection checkbox at the top
    available_feet = list(accelerations.keys())
    initial_status = [True if foot in available_feet else False for foot in ['L_FOOT', 'R_FOOT']]
    rax_foot = plt.axes([0.055, 0.93, 0.1, 0.05])
    check_foot = CheckButtons(rax_foot, ['L_FOOT', 'R_FOOT'], initial_status)
    
    def update_all_plots():
        """Update all plots based on checkbox states"""
        foot_status = check_foot.get_status()
        selected_feet = [foot for i, foot in enumerate(['L_FOOT', 'R_FOOT']) if foot_status[i] and foot in accelerations]
        
        # Update acceleration plot
        ax_accel.clear()
        if selected_feet:
            for foot in selected_feet:
                accel_mag = np.linalg.norm(accelerations[foot], axis=1)
                color = 'blue' if foot == 'L_FOOT' else 'red'
                ax_accel.plot(timestamps, accel_mag, linewidth=1.5, color=color, label=f'{foot} Magnitude', alpha=0.7)

                # Draw per-foot threshold band
                band_min, band_max = _get_accel_band(foot)
                ax_accel.axhline(y=band_min, color=color, linestyle='--', alpha=0.4, label=f'{foot} Min Th', zorder=1)
                ax_accel.axhline(y=band_max, color=color, linestyle='-.', alpha=0.4, label=f'{foot} Max Th', zorder=1)
            
            # Add vertical lines for calculation frame range
            ax_accel.axvline(x=timestamps[frame_start], color='green', linestyle=':', linewidth=2, alpha=0.6, label='Calc Start')
            ax_accel.axvline(x=timestamps[frame_end], color='purple', linestyle=':', linewidth=2, alpha=0.6, label='Calc End')
            
            ax_accel.set_ylabel('Acceleration (m/s²)', fontsize=11)
            ax_accel.set_title('Acceleration Magnitude', fontsize=12)
            ax_accel.legend(loc='upper right', fontsize=10)
            ax_accel.grid(True, alpha=0.3)
        else:
            ax_accel.text(0.5, 0.5, 'No foot selected', ha='center', va='center', 
                         fontsize=12, transform=ax_accel.transAxes)
        
        # Update gyroscope plot
        ax_gyro.clear()
        if selected_feet:
            for foot in selected_feet:
                gyro_mag = np.linalg.norm(gyroscopes[foot], axis=1)
                color = 'blue' if foot == 'L_FOOT' else 'red'
                ax_gyro.plot(timestamps, gyro_mag, linewidth=1.5, color=color, label=f'{foot} Magnitude', alpha=0.7)
            
            ax_gyro.axhline(y=gyro_threshold, color='orange', linestyle='--', alpha=0.5, label='Threshold', zorder=1)
            
            # Add vertical lines for calculation frame range
            ax_gyro.axvline(x=timestamps[frame_start], color='green', linestyle=':', linewidth=2, alpha=0.6, label='Calc Start')
            ax_gyro.axvline(x=timestamps[frame_end], color='purple', linestyle=':', linewidth=2, alpha=0.6, label='Calc End')
            
            ax_gyro.set_ylabel('Gyroscope (rad/s)', fontsize=11)
            ax_gyro.set_title('Gyroscope Magnitude', fontsize=12)
            ax_gyro.legend(loc='upper right', fontsize=10)
            ax_gyro.grid(True, alpha=0.3)
        else:
            ax_gyro.text(0.5, 0.5, 'No foot selected', ha='center', va='center', 
                        fontsize=12, transform=ax_gyro.transAxes)
        
        # Update contact plot
        ax_contact.clear()
        
        # Calculate contact ratios within the specified frame range
        frame_range = slice(frame_start, frame_end + 1)
        stats_text = f'Calculation frame range: {frame_start} - {frame_end}\n'
        
        total_frames = frame_end - frame_start + 1
        
        l_contact_in_range = None
        r_contact_in_range = None
        
        if 'L_FOOT' in foot_contact and 'L_FOOT' in selected_feet:
            l_contact = foot_contact['L_FOOT']
            l_contact_in_range = l_contact[frame_range]
            ax_contact.fill_between(timestamps, 0, l_contact.astype(int), alpha=0.6, color='blue', label='L_FOOT Contact')
            l_pct = 100.0 * np.sum(l_contact_in_range) / total_frames
            stats_text += f'L_FOOT: {l_pct:.1f}%\n'
        
        if 'R_FOOT' in foot_contact and 'R_FOOT' in selected_feet:
            r_contact = foot_contact['R_FOOT']
            r_contact_in_range = r_contact[frame_range]
            ax_contact.fill_between(timestamps, 0, -r_contact.astype(int), alpha=0.6, color='red', label='R_FOOT Contact')
            r_pct = 100.0 * np.sum(r_contact_in_range) / total_frames
            stats_text += f'R_FOOT: {r_pct:.1f}%\n'
        
        # Calculate any-foot contact ratio (union of selected feet)
        if l_contact_in_range is not None or r_contact_in_range is not None:
            if l_contact_in_range is not None and r_contact_in_range is not None:
                union_contact = l_contact_in_range | r_contact_in_range
            elif l_contact_in_range is not None:
                union_contact = l_contact_in_range
            else:
                union_contact = r_contact_in_range
            any_pct = 100.0 * np.sum(union_contact) / total_frames
            stats_text += f'Any foot: {any_pct:.1f}%\n'

        # Calculate both feet contact ratio only if both are selected and available
        if l_contact_in_range is not None and r_contact_in_range is not None:
            both_contact = l_contact_in_range & r_contact_in_range
            both_contact_count = np.sum(both_contact)
            both_pct = 100.0 * both_contact_count / total_frames
            stats_text += f'Double support: {both_pct:.1f}%'
        
        # Add vertical lines for calculation frame range
        ax_contact.axvline(x=timestamps[frame_start], color='green', linestyle=':', linewidth=2, alpha=0.6, label='Calc Start')
        ax_contact.axvline(x=timestamps[frame_end], color='purple', linestyle=':', linewidth=2, alpha=0.6, label='Calc End')
        
        ax_contact.set_ylabel('Contact', fontsize=11)
        ax_contact.set_xlabel('Time (s)', fontsize=11)
        ax_contact.set_ylim([-1.1, 1.1])
        ax_contact.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax_contact.set_title('Detected Foot Contact', fontsize=12)
        ax_contact.legend(loc='upper right', fontsize=10)
        ax_contact.text(0.005, 0.970, stats_text, transform=ax_contact.transAxes, 
                       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_contact.grid(True, alpha=0.3)
        
        fig.canvas.draw_idle()
    
    # Connect checkboxes to update function
    check_foot.on_clicked(lambda label: update_all_plots())
    
    # Initial plot
    update_all_plots()
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
