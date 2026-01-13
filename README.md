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

## 프로젝트 구조

```
├── main.py
├── config/settings.py       # 설정
├── core/                     # 데이터 처리
│   ├── imu_data.py
│   ├── data_processor.py
│   ├── calibration.py
│   └── kinematics.py        # TODO: 관절각 계산 구현 필요
├── file_io/file_handler.py  # 파일 입출력
└── ui/                      # GUI 컴포넌트
    ├── main_window.py
    ├── menu_bar.py
    ├── navigator.py
    ├── notes.py
    ├── main_view.py
    ├── visualization_3d.py  # TODO: 3D 렌더링 구현 필요
    ├── graph_view.py
    └── time_bar.py
```

## 기본 사용법

1. File > Import: 원시 데이터 불러오기
2. 캘리브레이션 구간 선택 (T-pose/N-pose)
3. Process 실행
4. 그래프/3D로 결과 확인
5. Save: 처리 결과 저장 (.mcp)

## 주요 구현 사항

- GUI 프레임워크 (PyQt5)
- 파일 관리 (Import/Open/Save)
- 배치 처리 지원
- 시간 구간 선택 및 부분 저장

## TODO

- `core/kinematics.py`: Quaternion 기반 관절각 계산
- `ui/visualization_3d.py`: 3D 렌더링
- `file_io/file_handler.py`: 실제 데이터 형식 파싱
