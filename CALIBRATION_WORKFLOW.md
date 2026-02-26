# Calibration Workflow Guide

## 개요

IMU 센서는 각 시행마다 초기 자세(N-pose 또는 T-pose)에서 calibration이 필요합니다.
이 시스템은 **calibration 전용 trial**과 **동작 trial**을 분리하여 처리합니다.

## Calibration 프로세스

### 방법 1: Calibration Trial 사용 (권장)

#### 1단계: Calibration Trial 수행
- N-pose 또는 T-pose를 4~5초간 유지한 trial 수행
- CSV 파일로 저장 (예: `Subject01_Npose.csv`)

#### 2단계: Calibration 생성
1. GUI에서 **File > Import**로 calibration trial CSV 불러오기
2. **Process > Perform Calibration** (단축키: `Ctrl+K`)
3. Pose 타입 선택 (N-pose or T-pose)
4. 전체 duration이 calibration으로 사용됨
5. **Process > Save Calibration** (`.cal` 파일로 저장)

#### 3단계: 동작 Trial에 Calibration 적용
1. **File > Import**로 동작 trial CSV 불러오기
2. **Process > Load Calibration** (단축키: `Ctrl+L`)로 저장된 `.cal` 파일 로드
3. "Apply to current data?" 팝업에서 **Yes** 선택
4. Calibration이 적용된 데이터로 처리 진행
5. **Process > Process Data** (단축키: `F5`)로 관절각 계산

### 방법 2: 동일 Trial에서 Calibration 구간 사용

#### 준비: Trial 시작 시 N-pose 포함
- Trial 시작 시 N-pose를 4~5초 유지
- 이후 동작 수행

#### 처리:
1. **File > Import**로 trial CSV 불러오기
2. Timeline에서 calibration 구간 확인
3. **Process > Set Calibration Period**로 시작/종료 시간 설정
4. **Process > Process Data**로 한 번에 처리

## 파일 형식

### Calibration 파일 (`.cal`)
```json
{
  "version": "1.0",
  "pose_type": "N-pose",
  "calibration_time": "2026-01-19T10:30:00",
  "subject_id": "Subject01",
  "reference_orientations": {
    "thigh_left": [w, x, y, z],
    "shank_left": [w, x, y, z],
    ...
  }
}
```

### 장점
- 한 번의 calibration으로 여러 trial에 재사용 가능
- Subject별, Session별로 calibration 관리 용이
- 처리 시간 단축

## 권장 워크플로우

### 실험 설계
```
Session/
  ├── Subject01_Npose.csv          # Calibration trial
  ├── Subject01_Walking.csv        # 동작 trial 1
  ├── Subject01_Running.csv        # 동작 trial 2
  └── Subject01_Npose.cal          # 생성된 calibration
```

### 처리 순서
1. **Npose trial** → Perform Calibration → **Save Calibration**
2. **Walking trial** → Load Calibration → Apply → Process Data → Save
3. **Running trial** → Load Calibration → Apply → Process Data → Save

## 단축키
- `Ctrl+I`: Import raw data
- `Ctrl+L`: Load calibration
- `Ctrl+K`: Perform calibration
- `Ctrl+S`: Save processed data
- `F5`: Process data

## 주의사항

1. **Calibration 유효성**
   - 센서 위치가 변경되면 새로운 calibration 필요
   - Subject가 다르면 새로운 calibration 필요
   - 하루 중 여러 session은 같은 calibration 사용 가능

2. **Back IMU = 0**
   - 현재 시스템에서 back IMU 값이 0인 것은 정상입니다
   - 다리 센서(thigh, shank, foot)만 사용하여 처리

3. **Gyroscope = 0**
   - 현재 CSV에서 gyroscope 값이 0으로 기록됨
   - Quaternion과 Acceleration 데이터만 사용

## 문제 해결

### "No calibration data available"
→ Calibration trial을 import하고 **Perform Calibration** 먼저 실행

### "No calibration reference for back"
→ Back 센서가 0값이므로 정상 (다리 센서만 사용)

### Calibration 적용 후 결과 확인
→ Graph View에서 joint angle 그래프 확인
→ 3D View에서 시각화 (구현 예정)
