"""
N-pose Calibration 간단 테스트
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.calibration import CalibrationProcessor
from core.kinematics import KinematicsProcessor
from file_io.file_handler import FileHandler

print("="*70)
print("N-pose Calibration 테스트")
print("="*70)

# Load N-pose data
npose_file = Path("data/legacy/PJS_20260119/PJS_20260119_Npose_processed.csv")
if not npose_file.exists():
    print(f"❌ N-pose file not found: {npose_file}")
    sys.exit(1)

print(f"\n📂 N-pose 데이터 로드: {npose_file}")
file_handler = FileHandler()
npose_data = file_handler.load_processed_file(str(npose_file))

if npose_data is None:
    print("❌ Failed to load N-pose data")
    sys.exit(1)

print(f"✓ 로드 완료: {npose_data.imu_data['back'].duration:.2f}초")

# Show raw quaternions at start of N-pose
print(f"\n📊 N-pose 시작 시점의 Raw Quaternions (t=0.5s):")
for location, sensor_data in npose_data.imu_data.items():
    idx = int(0.5 * sensor_data.sampling_frequency)
    if idx < len(sensor_data.quaternions):
        q = sensor_data.quaternions[idx]
        print(f"  {location:12s}: [{q[0]:7.4f}, {q[1]:7.4f}, {q[2]:7.4f}, {q[3]:7.4f}]")

# Perform calibration
print(f"\n⏳ Calibration 수행 중 (0.5s ~ 2.5s)...")
calibration = CalibrationProcessor()
calibration.calibrate(npose_data, start_time=0.5, end_time=2.5, pose_type="N-pose")

print(f"\n📊 계산된 Correction Quaternions:")
for location, q_corr in calibration.correction_quaternions.items():
    print(f"  {location:12s}: [{q_corr[0]:7.4f}, {q_corr[1]:7.4f}, {q_corr[2]:7.4f}, {q_corr[3]:7.4f}]")

# Apply calibration to N-pose data itself to verify
print(f"\n⏳ Calibration을 N-pose 데이터에 적용하여 검증...")
calibrated_npose = calibration.apply_to_data(npose_data)

# Check if N-pose becomes identity after calibration
print(f"\n✅ Calibration 검증: N-pose에 적용한 후 quaternion (t=0.5s):")
print(f"   (모든 센서가 identity [1, 0, 0, 0]에 가까워야 함)")
for location, sensor_data in calibrated_npose.imu_data.items():
    idx = int(0.5 * sensor_data.sampling_frequency)
    if idx < len(sensor_data.quaternions):
        q = sensor_data.quaternions[idx]
        identity = np.array([1, 0, 0, 0])
        error = np.linalg.norm(q - identity)
        status = "✓" if error < 0.01 else "⚠️"
        print(f"  {status} {location:12s}: [{q[0]:7.4f}, {q[1]:7.4f}, {q[2]:7.4f}, {q[3]:7.4f}] (error: {error:.4f})")

# Compute joint angles on calibrated N-pose
print(f"\n⏳ N-pose 관절 각도 계산...")
kinematics = KinematicsProcessor()
joint_angles = kinematics.compute_joint_angles(calibrated_npose)

print(f"\n✅ N-pose 관절 각도 (t=0.5s ~ 2.5s 평균):")
print(f"   (N-pose에서는 모든 각도가 0도에 가까워야 함)")

fs = calibrated_npose.imu_data['back'].sampling_frequency
start_idx = int(0.5 * fs)
end_idx = int(2.5 * fs)

for joint_name in ['hip_right', 'knee_right', 'ankle_right', 'hip_left', 'knee_left', 'ankle_left']:
    angles = getattr(joint_angles, joint_name)[start_idx:end_idx]
    mean_angles = np.mean(angles, axis=0)
    std_angles = np.std(angles, axis=0)
    max_abs = np.max(np.abs(mean_angles))
    status = "✓" if max_abs < 10 else "⚠️"
    print(f"  {status} {joint_name:12s}: yaw={mean_angles[0]:6.1f}° pitch={mean_angles[1]:6.1f}° roll={mean_angles[2]:6.1f}° "
          f"(std: {std_angles[0]:.1f}° {std_angles[1]:.1f}° {std_angles[2]:.1f}°)")

print(f"\n" + "="*70)
print("테스트 완료!")
print("="*70)
