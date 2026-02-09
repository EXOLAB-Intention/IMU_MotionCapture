"""
Test script to verify IMU angle calculation
"""
import numpy as np
from core.kinematics import KinematicsProcessor

# Create test processor
processor = KinematicsProcessor()

print("="*60)
print("IMU 각도 계산 검증 테스트")
print("="*60)

# Test 1: Identity quaternion should give zero angles
print("\n테스트 1: Identity quaternion [1, 0, 0, 0]")
q_identity = np.array([1.0, 0.0, 0.0, 0.0])
angles = processor.quaternion_to_euler(q_identity)
print(f"  Euler angles (yaw, pitch, roll): [{angles[0]:.2f}°, {angles[1]:.2f}°, {angles[2]:.2f}°]")
print(f"  Expected: [0.00°, 0.00°, 0.00°]")
print(f"  ✓ PASS" if np.allclose(angles, [0, 0, 0], atol=1e-10) else f"  ✗ FAIL")

# Test 2: 90 degree rotation about Z-axis (yaw)
print("\n테스트 2: Z축 90도 회전 (yaw)")
q_z90 = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])  # 90 deg about Z
angles = processor.quaternion_to_euler(q_z90)
print(f"  Quaternion: {q_z90}")
print(f"  Euler angles (yaw, pitch, roll): [{angles[0]:.2f}°, {angles[1]:.2f}°, {angles[2]:.2f}°]")
print(f"  Expected: [90.00°, 0.00°, 0.00°]")
print(f"  ✓ PASS" if np.allclose(angles, [90, 0, 0], atol=1e-6) else f"  ✗ FAIL")

# Test 3: 90 degree rotation about Y-axis (pitch)
print("\n테스트 3: Y축 90도 회전 (pitch)")
q_y90 = np.array([np.cos(np.pi/4), 0.0, np.sin(np.pi/4), 0.0])  # 90 deg about Y
angles = processor.quaternion_to_euler(q_y90)
print(f"  Quaternion: {q_y90}")
print(f"  Euler angles (yaw, pitch, roll): [{angles[0]:.2f}°, {angles[1]:.2f}°, {angles[2]:.2f}°]")
print(f"  Expected: [0.00°, 90.00°, 0.00°]")
print(f"  ✓ PASS" if np.allclose(angles, [0, 90, 0], atol=1e-6) else f"  ✗ FAIL")

# Test 4: 90 degree rotation about X-axis (roll)
print("\n테스트 4: X축 90도 회전 (roll)")
q_x90 = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 0.0])  # 90 deg about X
angles = processor.quaternion_to_euler(q_x90)
print(f"  Quaternion: {q_x90}")
print(f"  Euler angles (yaw, pitch, roll): [{angles[0]:.2f}°, {angles[1]:.2f}°, {angles[2]:.2f}°]")
print(f"  Expected: [0.00°, 0.00°, 90.00°]")
print(f"  ✓ PASS" if np.allclose(angles, [0, 0, 90], atol=1e-6) else f"  ✗ FAIL")

# Test 5: Relative orientation calculation
print("\n테스트 5: 상대 회전 계산")
print("  시나리오: 근위부 세그먼트가 Identity, 원위부가 Y축으로 30도 회전")
q_proximal = np.array([1.0, 0.0, 0.0, 0.0])
q_distal = np.array([np.cos(np.pi/12), 0.0, np.sin(np.pi/12), 0.0])  # 30 deg about Y
q_rel = processor.compute_relative_orientation(q_proximal, q_distal)
angles_rel = processor.quaternion_to_euler(q_rel)
print(f"  Relative quaternion: {q_rel}")
print(f"  Relative Euler angles (yaw, pitch, roll): [{angles_rel[0]:.2f}°, {angles_rel[1]:.2f}°, {angles_rel[2]:.2f}°]")
print(f"  Expected: [0.00°, 30.00°, 0.00°]")
print(f"  ✓ PASS" if np.allclose(angles_rel, [0, 30, 0], atol=1e-6) else f"  ✗ FAIL")

# Test 6: Batch processing
print("\n테스트 6: 배치 처리 (여러 quaternion 동시 처리)")
q_batch = np.array([
    [1.0, 0.0, 0.0, 0.0],  # Identity
    [np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)],  # 90 deg Z
    [np.cos(np.pi/4), 0.0, np.sin(np.pi/4), 0.0],  # 90 deg Y
    [np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 0.0],  # 90 deg X
])
angles_batch = processor.quaternion_to_euler(q_batch)
print(f"  입력: 4개의 quaternions")
print(f"  출력 shape: {angles_batch.shape}")
expected_batch = np.array([
    [0, 0, 0],
    [90, 0, 0],
    [0, 90, 0],
    [0, 0, 90]
])
print(f"  결과:")
for i, (angles, expected) in enumerate(zip(angles_batch, expected_batch)):
    match = "✓" if np.allclose(angles, expected, atol=1e-6) else "✗"
    print(f"    {match} Sample {i}: [{angles[0]:6.2f}°, {angles[1]:6.2f}°, {angles[2]:6.2f}°] "
          f"(expected: [{expected[0]:6.2f}°, {expected[1]:6.2f}°, {expected[2]:6.2f}°])")

print("\n" + "="*60)
print("테스트 완료!")
print("="*60)
