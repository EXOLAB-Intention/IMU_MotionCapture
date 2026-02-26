"""
실제 데이터로 각도 계산 검증
"""
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.imu_data import MotionCaptureData
from core.calibration import CalibrationProcessor
from core.kinematics import KinematicsProcessor
from file_io.file_handler import FileHandler

print("="*70)
print("실제 IMU 데이터로 각도 계산 검증")
print("="*70)

# File paths
npose_file = Path("data/legacy/PJS_20260119/PJS_20260119_Npose_processed.csv")
motion_file = Path("data/legacy/PJS_20260119/PJS_20260119_walk_01_processed.csv")

if not npose_file.exists():
    print(f"❌ N-pose file not found: {npose_file}")
    sys.exit(1)
    
if not motion_file.exists():
    print(f"❌ Motion file not found: {motion_file}")
    sys.exit(1)

print(f"\n📂 파일 로드:")
print(f"  - N-pose: {npose_file}")
print(f"  - Motion: {motion_file}")

# Load N-pose data for calibration
print(f"\n⏳ N-pose 데이터 로드 중...")
file_handler = FileHandler()
npose_data = file_handler.load_processed_file(str(npose_file))

if npose_data is None:
    print("❌ Failed to load N-pose data")
    sys.exit(1)

print(f"✓ N-pose 데이터 로드 완료")
print(f"  - 샘플 수: {len(npose_data.imu_data['back'].timestamps)}")
print(f"  - 지속 시간: {npose_data.imu_data['back'].duration:.2f}초")

# Perform calibration on N-pose data
print(f"\n⏳ N-pose Calibration 수행 중...")
calibration = CalibrationProcessor()
# Use first 2 seconds of N-pose data for calibration
start_time = 0.5  # Skip first 0.5s to allow subject to stabilize
end_time = 2.5    # Use 2 seconds of stable pose
calibration.calibrate(npose_data, start_time, end_time, pose_type="N-pose")
print(f"✓ Calibration 완료")

# Load motion data
print(f"\n⏳ 보행 데이터 로드 중...")
motion_data = file_handler.load_processed_file(str(motion_file))

if motion_data is None:
    print("❌ Failed to load motion data")
    sys.exit(1)

print(f"✓ 보행 데이터 로드 완료")
print(f"  - 샘플 수: {len(motion_data.imu_data['back'].timestamps)}")
print(f"  - 지속 시간: {motion_data.imu_data['back'].duration:.2f}초")
print(f"  - IMU 센서 개수: {len(motion_data.imu_data)}")
print(f"  - 센서 위치: {list(motion_data.imu_data.keys())}")

# Apply calibration to motion data
print(f"\n⏳ Calibration 적용 중...")
calibrated_data = calibration.apply_to_data(motion_data)
print(f"✓ Calibration 적용 완료")

# Compute joint angles
print(f"\n⏳ 관절 각도 계산 중...")
kinematics = KinematicsProcessor()
joint_angles = kinematics.compute_joint_angles(calibrated_data)

if joint_angles is None:
    print("❌ Failed to compute joint angles")
    sys.exit(1)

print(f"✓ 관절 각도 계산 완료")

# Display statistics
print(f"\n📊 관절 각도 통계 (처음 5초 데이터):")
print(f"\n  Right Hip [yaw, pitch, roll] (degrees):")

# Get first 5 seconds of data
fs = motion_data.imu_data['back'].sampling_frequency
n_samples = min(int(5 * fs), len(joint_angles.timestamps))

hip_right_5s = joint_angles.hip_right[:n_samples]
knee_right_5s = joint_angles.knee_right[:n_samples]
ankle_right_5s = joint_angles.ankle_right[:n_samples]

print(f"    - Yaw:   mean={np.mean(hip_right_5s[:,0]):7.2f}°, std={np.std(hip_right_5s[:,0]):6.2f}°, "
      f"range=[{np.min(hip_right_5s[:,0]):7.2f}°, {np.max(hip_right_5s[:,0]):7.2f}°]")
print(f"    - Pitch: mean={np.mean(hip_right_5s[:,1]):7.2f}°, std={np.std(hip_right_5s[:,1]):6.2f}°, "
      f"range=[{np.min(hip_right_5s[:,1]):7.2f}°, {np.max(hip_right_5s[:,1]):7.2f}°]")
print(f"    - Roll:  mean={np.mean(hip_right_5s[:,2]):7.2f}°, std={np.std(hip_right_5s[:,2]):6.2f}°, "
      f"range=[{np.min(hip_right_5s[:,2]):7.2f}°, {np.max(hip_right_5s[:,2]):7.2f}°]")

print(f"\n  Right Knee [yaw, pitch, roll] (degrees):")
print(f"    - Yaw:   mean={np.mean(knee_right_5s[:,0]):7.2f}°, std={np.std(knee_right_5s[:,0]):6.2f}°, "
      f"range=[{np.min(knee_right_5s[:,0]):7.2f}°, {np.max(knee_right_5s[:,0]):7.2f}°]")
print(f"    - Pitch: mean={np.mean(knee_right_5s[:,1]):7.2f}°, std={np.std(knee_right_5s[:,1]):6.2f}°, "
      f"range=[{np.min(knee_right_5s[:,1]):7.2f}°, {np.max(knee_right_5s[:,1]):7.2f}°]")
print(f"    - Roll:  mean={np.mean(knee_right_5s[:,2]):7.2f}°, std={np.std(knee_right_5s[:,2]):6.2f}°, "
      f"range=[{np.min(knee_right_5s[:,2]):7.2f}°, {np.max(knee_right_5s[:,2]):7.2f}°]")

print(f"\n  Right Ankle [yaw, pitch, roll] (degrees):")
print(f"    - Yaw:   mean={np.mean(ankle_right_5s[:,0]):7.2f}°, std={np.std(ankle_right_5s[:,0]):6.2f}°, "
      f"range=[{np.min(ankle_right_5s[:,0]):7.2f}°, {np.max(ankle_right_5s[:,0]):7.2f}°]")
print(f"    - Pitch: mean={np.mean(ankle_right_5s[:,1]):7.2f}°, std={np.std(ankle_right_5s[:,1]):6.2f}°, "
      f"range=[{np.min(ankle_right_5s[:,1]):7.2f}°, {np.max(ankle_right_5s[:,1]):7.2f}°]")
print(f"    - Roll:  mean={np.mean(ankle_right_5s[:,2]):7.2f}°, std={np.std(ankle_right_5s[:,2]):6.2f}°, "
      f"range=[{np.min(ankle_right_5s[:,2]):7.2f}°, {np.max(ankle_right_5s[:,2]):7.2f}°]")

# Check for reasonable values
print(f"\n🔍 합리성 검사:")
reasonable = True

# Hip flexion/extension (pitch) should be reasonable during walking (-30 to 50 degrees typically)
hip_pitch_mean = np.mean(hip_right_5s[:,1])
if -50 < hip_pitch_mean < 70:
    print(f"  ✓ Hip pitch mean ({hip_pitch_mean:.1f}°) is reasonable")
else:
    print(f"  ⚠️  Hip pitch mean ({hip_pitch_mean:.1f}°) might be unusual")
    reasonable = False

# Knee should flex during walking (0 to 70 degrees typically)
knee_pitch_range = np.max(knee_right_5s[:,1]) - np.min(knee_right_5s[:,1])
if knee_pitch_range > 10:
    print(f"  ✓ Knee pitch range ({knee_pitch_range:.1f}°) shows movement")
else:
    print(f"  ⚠️  Knee pitch range ({knee_pitch_range:.1f}°) is very small")
    reasonable = False

# Ankle should have some movement
ankle_pitch_range = np.max(ankle_right_5s[:,1]) - np.min(ankle_right_5s[:,1])
if ankle_pitch_range > 5:
    print(f"  ✓ Ankle pitch range ({ankle_pitch_range:.1f}°) shows movement")
else:
    print(f"  ⚠️  Ankle pitch range ({ankle_pitch_range:.1f}°) is very small")
    reasonable = False

print(f"\n" + "="*70)
if reasonable:
    print("✅ 각도 계산이 정상적으로 보입니다!")
else:
    print("⚠️  각도 값이 예상 범위를 벗어났습니다. 추가 확인이 필요합니다.")
print("="*70)
