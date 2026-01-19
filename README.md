# IMU Motion Capture

Xsens MTi-630 기반 하체 운동학 분석 GUI

## 센서 구성

7개 센서: trunk, thigh_right/left, shank_right/left, foot_right/left
- Quaternion (w,x,y,z), 3축 가속도/자이로
- 샘플링: 500~1000Hz

## 설치 및 실행

```bash
pip install -r requirements.txt
python main.py
```

## CSV 데이터 형식

지원되는 CSV 컬럼 구조:
- **Trunk**: `TrunkIMU_QuatW/X/Y/Z`, `TrunkIMU_LocalAccX/Y/Z`, `TrunkIMU_LocalGyrX/Y/Z`
- **Left Thigh**: `L_THIGH_IMU_QuatW/X/Y/Z`, `L_THIGH_IMU_AccX/Y/Z`, `L_THIGH_IMU_GyrX/Y/Z`
- **Left Shank**: `L_SHANK_IMU_QuatW/X/Y/Z`, `L_SHANK_IMU_AccX/Y/Z`, `L_SHANK_IMU_GyrX/Y/Z`
- **Left Foot**: `L_FOOT_IMU_QuatW/X/Y/Z`, `L_FOOT_IMU_AccX/Y/Z`, `L_FOOT_IMU_GyrX/Y/Z`
- **Right Thigh**: `R_THIGH_IMU_QuatW/X/Y/Z`, `R_THIGH_IMU_AccX/Y/Z`, `R_THIGH_IMU_GyrX/Y/Z`
- **Right Shank**: `R_SHANK_IMU_QuatW/X/Y/Z`, `R_SHANK_IMU_AccX/Y/Z`, `R_SHANK_IMU_GyrX/Y/Z`
- **Right Foot**: `R_FOOT_IMU_QuatW/X/Y/Z`, `R_FOOT_IMU_AccX/Y/Z`, `R_FOOT_IMU_GyrX/Y/Z`

타임스탬프는 `LoopCnt` 컬럼에서 자동 계산 (기본 100Hz)

## 프로젝트 구조

```
├── main.py
├── config/settings.py       # 설정
├── core/                     # 데이터 처리
│   ├── imu_data.py          # ✓ 완료
│   ├── data_processor.py    # ✓ 완료
│   ├── calibration.py       # ✓ 완료 (프레임워크)
│   └── kinematics.py        # TODO: 관절각 계산 구현 필요
├── file_io/file_handler.py  # ✓ 완료 (CSV 파싱)
└── ui/                      # GUI 컴포넌트
    ├── main_window.py       # ✓ 완료
    ├── menu_bar.py          # ✓ 완료
    ├── navigator.py         # ✓ 완료
    ├── notes.py             # ✓ 완료
    ├── main_view.py         # ✓ 완료
    ├── visualization_3d.py  # TODO: 3D 렌더링 구현 필요
    ├── graph_view.py        # ✓ 완료
    └── time_bar.py          # ✓ 완료
```

## 기본 사용법

1. **File > Import**: CSV 파일 불러오기
   - 자동으로 센서 위치별 데이터 파싱
   - 누락된 센서나 0값 데이터는 자동 건너뜀
   
2. **캘리브레이션 구간 선택** (T-pose/N-pose)
   - 타임라인에서 시작/종료 시간 선택
   
3. **Process > Process Data**: 처리 실행
   - 관절각 계산 (구현 예정)
   - 발 접지 감지 (구현 예정)
   
4. **그래프/3D로 결과 확인**
   - 그래프: 시계열 데이터 표시
   - 3D: 인체 모델 시각화 (구현 예정)
   
5. **File > Save**: 처리 결과 저장 (.mcp 형식)

## 주요 기능

✓ **완료된 기능**:
- GUI 프레임워크 (PyQt5)
- CSV 파일 Import (센서별 자동 파싱)
- 파일 관리 (Import/Open/Save)
- 배치 처리 지원
- 시간 구간 선택 및 부분 저장
- IMU 데이터 구조 (Quaternion, Acc, Gyro)
- JSON 기반 처리 데이터 저장/로드

⏳ **구현 필요**:
- `core/kinematics.py`: Quaternion 기반 관절각 계산 알고리즘
- `ui/visualization_3d.py`: OpenGL/PyQtGraph 3D 렌더링
- 발 접지 감지 및 보행 분석

## 테스트

```bash
# CSV import 테스트
python test_import.py

# GUI 실행
python main.py
```
