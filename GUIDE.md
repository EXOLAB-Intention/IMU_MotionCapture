## 1. main.py (프로그램 시작점)

**역할**: 프로그램을 실행하는 진입점

```python
def main():
    # PyQt5 GUI 애플리케이션 생성
    app = QApplication(sys.argv)
    
    # 메인 윈도우 생성
    window = MainWindow()
    window.show()
    
    # 프로그램 실행 (종료할 때까지 계속 돌아감)
    sys.exit(app.exec_())
```

**설명**:
- 실행하면 GUI 창이 뜨고, 창을 닫으면 프로그램이 종료됨
- `if __name__ == "__main__":` → 이 파일을 직접 실행했을 때만 작동

---

## 2. config/settings.py (설정 파일)

**역할**: 프로그램에서 사용하는 모든 설정값을 관리

### 주요 클래스들

#### `IMUConfig` - IMU 센서 설정
```python
model: str = "Xsens MTi-630"       # 센서 모델명
sampling_frequency: int = 1000      # 샘플링 주파수 (Hz)
num_sensors: int = 7                # 센서 개수
sensor_locations = ["back", ...]    # 센서 위치 목록
```

#### `SubjectConfig` - 피험자 정보
```python
height: float = 170.0        # 키 (cm)
shoe_size: float = 270.0     # 신발 사이즈 (mm)
back_ratio: float = 0.288    # 몸통 길이 비율
```

#### `AppSettings` - 전체 설정 통합
```python
def __init__(self):
    self.imu = IMUConfig()           # IMU 설정
    self.subject = SubjectConfig()    # 피험자 설정
    self.sensor_mapping: dict = {}    # 센서 번호 매핑
```

**특징**:
- `@dataclass`: 데이터를 담는 클래스를 쉽게 만들어주는 도구
- `app_settings = AppSettings()`: 프로그램 전체에서 공유하는 설정 객체

---

## 3. core/imu_data.py (데이터 구조)

**역할**: IMU 데이터를 저장하는 그릇들을 정의

### 주요 클래스들

#### `IMUSample` - 한 순간의 센서 데이터
```python
timestamp: float              # 시간 (초)
quaternion: np.ndarray        # 방향 데이터 [w,x,y,z]
acceleration: np.ndarray      # 가속도 [ax,ay,az]
gyroscope: np.ndarray         # 자이로 [gx,gy,gz]
```

#### `IMUSensorData` - 한 센서의 시계열 데이터
```python
sensor_id: int                # 센서 번호
location: str                 # 위치 (예: "back")
timestamps: np.ndarray        # 시간 배열 (N개)
quaternions: np.ndarray       # 방향 배열 (N x 4)
```

**주요 메서드**:
```python
def get_sample(self, index):
    # index 번째 샘플 하나 가져오기
    
def get_time_slice(self, start_time, end_time):
    # 특정 시간 구간만 잘라내기
```

#### `MotionCaptureData` - 전체 데이터 묶음
```python
session_id: str                           # 세션 ID
imu_data: Dict[str, IMUSensorData]       # 센서별 데이터
joint_angles: Optional[JointAngles]       # 관절각 (처리 후)
kinematics: Optional[KinematicsData]      # 운동학 데이터 (처리 후)
notes: str                                # 메모
```

**설명**:
- `Dict[str, IMUSensorData]`: 위치 이름(str)을 키로, 센서 데이터를 값으로 하는 딕셔너리
- `Optional[...]`: 있을 수도, 없을 수도 있는 데이터

---

## 4. io/file_handler.py (파일 입출력)

**역할**: 파일을 읽고 쓰는 기능

### 주요 메서드들

#### `import_raw_data(filepath)` - 원시 데이터 불러오기
```python
def import_raw_data(filepath: str) -> MotionCaptureData:
    # CSV/TXT 파일 읽어서 MotionCaptureData 객체로 변환
    # TODO: 실제 파일 형식에 맞춰 구현 필요
```

#### `save_processed_data(data, filepath)` - 처리된 데이터 저장
```python
def save_processed_data(data, filepath):
    # MotionCaptureData를 JSON 형식으로 저장 (.mcp 파일)
    save_dict = {
        'imu_data': {...},
        'joint_angles': {...},
        'kinematics': {...}
    }
    json.dump(save_dict, f)  # 파일로 저장
```

#### `load_processed_data(filepath)` - 저장된 데이터 불러오기
```python
def load_processed_data(filepath: str) -> MotionCaptureData:
    # .mcp 파일 읽어서 다시 MotionCaptureData 객체로 복원
```

#### `scan_directory(directory)` - 디렉토리 스캔
```python
def scan_directory(directory: str) -> Dict[str, List[str]]:
    # 디렉토리 안의 파일 목록 분류
    files = {
        'raw': [],        # 원시 데이터 파일들
        'processed': []   # 처리된 파일들
    }
    return files
```

**특징**:
- 모든 메서드가 `@staticmethod` → 객체 없이 바로 사용 가능
- 예: `FileHandler.scan_directory("/path/to/dir")`

---

## 5. core/calibration.py (캘리브레이션)

**역할**: 초기 자세(T-pose/N-pose)로 센서 기준점 설정

### 주요 메서드

#### `calibrate()` - 캘리브레이션 수행
```python
def calibrate(self, data, start_time, end_time, pose_type):
    # 지정된 시간 구간의 데이터로 기준 방향 계산
    for location, sensor_data in data.imu_data.items():
        # 해당 시간대 데이터만 추출
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        calib_quaternions = sensor_data.quaternions[mask]
        
        # 평균 내서 기준 방향 저장
        reference_quat = self._average_quaternions(calib_quaternions)
        self.reference_orientations[location] = reference_quat
```

**설명**:
- T-pose나 N-pose를 취한 1-5초 구간의 데이터를 평균내서
- 각 센서의 "기준 방향"을 결정
- 이후 모든 데이터는 이 기준을 바탕으로 계산됨

---

## 6. core/kinematics.py (운동학 계산)

**역할**: 관절각, 발 접촉, 속도 등을 계산 (아직 구현 안 됨)

### 구현해야 할 함수들

```python
def compute_joint_angles(data):
    # IMU 방향 데이터 → 고관절/무릎/발목 각도 계산
    # TODO: Quaternion 수학 구현 필요
    
def detect_foot_contact(data):
    # 가속도 패턴으로 발이 땅에 닿았는지 판단
    # TODO: 알고리즘 구현 필요
    
def compute_velocity(data, foot_contacts):
    # 발 접촉 정보와 IMU로 이동 속도 추정
    # TODO: 알고리즘 구현 필요
```

**현재 상태**: 빈 껍데기만 있음, 실제 알고리즘 구현 필요

---

## 7. core/data_processor.py (데이터 처리 파이프라인)

**역할**: 전체 처리 과정을 순서대로 실행

```python
def process_motion_data(self, data, calib_start, calib_end):
    # Step 1: 캘리브레이션
    self.calibration_processor.calibrate(data, calib_start, calib_end)
    
    # Step 2: 관절각 계산
    joint_angles = self.kinematics_processor.compute_joint_angles(data)
    
    # Step 3: 몸통 각도 계산
    trunk_angle = self.kinematics_processor.compute_trunk_angle(data)
    
    # Step 4: 발 접촉 감지
    foot_contacts = self.kinematics_processor.detect_foot_contact(data)
    
    # Step 5: 속도 계산
    velocity = self.kinematics_processor.compute_velocity(data, foot_contacts)
    
    # 모든 결과를 data 객체에 저장
    data.joint_angles = joint_angles
    data.kinematics = kinematics_data
    data.is_processed = True
    
    return data
```

**설명**: 여러 처리 단계를 조립해서 실행하는 "매니저" 역할

---

## 8. UI 파일들 (화면 구성)

### ui/main_window.py (메인 윈도우)

**역할**: 전체 GUI 창을 관리

```python
class MainWindow(QMainWindow):
    def __init__(self):
        # 화면 구성요소들 생성
        self.navigator = NavigatorPanel()  # 왼쪽 파일 목록
        self.notes = NotesPanel()          # 왼쪽 메모
        self.main_view = MainView()        # 오른쪽 메인 화면
        
        # 데이터 관리 객체들
        self.file_handler = FileHandler()
        self.data_processor = DataProcessor()
```

**주요 메서드**:
- `import_file()`: 원시 데이터 불러오기
- `save_file()`: 처리된 데이터 저장
- `process_data()`: 데이터 처리 실행

**Signal-Slot 연결**:
```python
# 메뉴에서 Import 클릭 → import_file() 실행
self.menu.import_file_requested.connect(self.import_file)

# 네비게이터에서 파일 선택 → load_selected_file() 실행
self.navigator.file_selected.connect(self.load_selected_file)
```

**설명**: 
- PyQt의 Signal-Slot은 이벤트 처리 방식
- "버튼 클릭" 같은 이벤트가 발생하면 연결된 함수가 자동 실행

---

### ui/menu_bar.py (메뉴바)

**역할**: 상단 메뉴 (File, View, Process, Help) 생성

```python
class MenuBar(QObject):
    # Signal 정의 (이벤트 알림용)
    import_file_requested = pyqtSignal(str)  # 파일 경로 전달
    save_requested = pyqtSignal()
    
    def _on_import(self):
        # 파일 선택 대화상자 띄우기
        filepath, _ = QFileDialog.getOpenFileName(...)
        if filepath:
            # Signal 발송 (연결된 함수가 실행됨)
            self.import_file_requested.emit(filepath)
```

**설명**: 메뉴 클릭 → Signal 발송 → MainWindow에서 처리

---

### ui/navigator.py (파일 네비게이터)

**역할**: 왼쪽 패널에 파일 목록 표시

```python
class NavigatorPanel(QWidget):
    file_selected = pyqtSignal(str, bool)  # 파일경로, 처리여부
    
    def set_directory(self, directory):
        # 디렉토리 설정
        self.current_directory = directory
        self.refresh()  # 파일 목록 갱신
    
    def refresh(self):
        # FileHandler로 파일 스캔
        files = FileHandler.scan_directory(self.current_directory)
        
        # 트리 위젯에 표시
        for filepath in files['raw']:
            item = QTreeWidgetItem(["파일명", "Raw", "미처리"])
            # 빨간색으로 표시
```

**설명**: 
- QTreeWidget: 폴더 트리 같은 계층 구조 표시
- 파일 더블클릭 시 `file_selected` Signal 발송

---

### ui/notes.py (노트 패널)

**역할**: 각 데이터 파일에 메모 작성

```python
class NotesPanel(QWidget):
    notes_changed = pyqtSignal(str)  # 메모 내용 전달
    
    def set_notes(self, notes):
        # 기존 메모 불러와서 표시
        self.notes_edit.setPlainText(notes)
    
    def get_notes(self):
        # 현재 작성된 메모 반환
        return self.notes_edit.toPlainText()
```

**설명**: 간단한 텍스트 에디터 역할

---

### ui/main_view.py (메인 화면)

**역할**: 3D 시각화 + 그래프를 통합 관리

```python
class MainView(QWidget):
    def __init__(self):
        # 화면 분할 (Splitter)
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # 왼쪽: 3D 시각화
        self.visualization_3d = Visualization3D()
        self.main_splitter.addWidget(self.visualization_3d)
        
        # 오른쪽: 그래프 (토글 가능)
        self.graph_view = GraphView()
        self.main_splitter.addWidget(self.graph_view)
        
        # 아래: 타임바
        self.time_bar = TimeBar()
    
    def set_data(self, motion_data):
        # 데이터를 각 하위 컴포넌트에 전달
        self.visualization_3d.set_data(motion_data)
        self.graph_view.set_data(motion_data)
        self.time_bar.set_duration(motion_data.duration)
```

**설명**: 여러 화면 요소를 묶어서 배치

---

### ui/visualization_3d.py (3D 시각화)

**역할**: 움직이는 사람을 3D로 표시 (아직 구현 안 됨)

```python
class Visualization3D(QWidget):
    def set_data(self, motion_data):
        # 데이터 저장
        self.current_data = motion_data
        self._render_frame(0)  # 첫 프레임 렌더링
    
    def _render_frame(self, frame_index):
        # TODO: 실제 3D 렌더링 구현 필요
        # - 각 센서의 quaternion 읽기
        # - 분절(몸통, 허벅지 등)을 3D 객체로 그리기
        # - 회전/위치 적용
        pass
    
    def _start_playback(self):
        # 재생 시작 (타이머로 프레임 진행)
        self.timer.start(1000 // 60)  # 60 FPS
```

**설명**: 플레이스홀더만 있음, OpenGL이나 PyQtGraph로 구현 필요

---

### ui/graph_view.py (그래프)

**역할**: 관절각을 시계열 그래프로 표시

```python
class GraphView(QWidget):
    def _update_graph(self):
        # 선택된 관절 가져오기
        joint = self.joint_combo.currentText()  # "Hip", "Knee", "Ankle"
        
        # 데이터 읽기
        timestamps = self.current_data.joint_angles.timestamps
        angles_right = self.current_data.joint_angles.get_joint_angle(joint, 'right')
        
        # Matplotlib으로 그래프 그리기
        for i in range(3):  # 3개 자유도
            self.axes[i].plot(timestamps, angles_right[:, i], 'r-')
```

**설명**: 
- Matplotlib: 그래프 그리는 라이브러리
- 3개 subplot: 굴곡/외전/회전 각도 따로 표시

---

### ui/time_bar.py (타임바)

**역할**: 시간 탐색 및 구간 선택

```python
class TimeBar(QWidget):
    time_changed = pyqtSignal(float)       # 현재 시간
    range_changed = pyqtSignal(float, float)  # 선택 구간
    
    def set_duration(self, duration, sampling_freq):
        # 전체 시간 설정
        n_samples = int(duration * sampling_freq)
        self.time_slider.setMaximum(n_samples - 1)
    
    def _on_slider_changed(self, value):
        # 슬라이더 움직이면 시간 계산 후 Signal 발송
        time = value / self.time_slider.maximum() * self.total_duration
        self.time_changed.emit(time)
```

**설명**: 
- 슬라이더로 현재 시간 선택
- SpinBox로 시작/끝 구간 선택
- Save 시 선택 구간만 저장 가능

---

## 데이터 흐름 요약

```
1. 사용자가 File > Import 클릭
   ↓
2. MenuBar가 import_file_requested Signal 발송
   ↓
3. MainWindow.import_file() 실행
   ↓
4. FileHandler.import_raw_data() 호출
   ↓
5. MotionCaptureData 객체 생성
   ↓
6. MainView.set_data()로 화면에 표시
   ↓
7. 사용자가 Process 버튼 클릭
   ↓
8. DataProcessor.process_motion_data() 실행
   - Calibration
   - 관절각 계산
   - 발 접촉 감지
   - 속도 계산
   ↓
9. 처리된 데이터를 다시 화면에 표시
   ↓
10. 사용자가 Save 클릭
    ↓
11. FileHandler.save_processed_data() 호출
    ↓
12. .mcp 파일로 저장 완료
```

---

## 파일 간 관계도

```
main.py (시작)
  ↓
MainWindow (총괄)
  ├─ MenuBar (메뉴)
  ├─ NavigatorPanel (파일 목록)
  ├─ NotesPanel (메모)
  ├─ MainView (메인 화면)
  │   ├─ Visualization3D (3D)
  │   ├─ GraphView (그래프)
  │   └─ TimeBar (타임라인)
  ├─ FileHandler (파일 처리)
  └─ DataProcessor (데이터 처리)
      ├─ CalibrationProcessor (캘리브레이션)
      └─ KinematicsProcessor (운동학 계산)

데이터 구조:
  ├─ MotionCaptureData (전체 데이터)
  │   ├─ IMUSensorData (센서 데이터)
  │   ├─ JointAngles (관절각)
  │   └─ KinematicsData (운동학)
  └─ AppSettings (설정)
```

---

## 초보자를 위한 팁

### 1. 파일 수정 시작할 때
```python
# 먼저 어떤 클래스가 있는지 확인
# 클래스 안에 어떤 메서드(함수)가 있는지 확인
# 메서드의 입력/출력이 무엇인지 확인
```

### 2. Signal-Slot 이해하기
```python
# Signal 정의 (보내는 쪽)
my_signal = pyqtSignal(str)

# Signal 발송
self.my_signal.emit("Hello")

# Slot 연결 (받는 쪽)
self.my_signal.connect(self.my_function)

# 결과: my_function("Hello") 자동 실행
```

### 3. 데이터 확인하기
```python
# 중간에 print 찍어서 확인
print(f"현재 데이터: {self.current_data}")
print(f"타입: {type(self.current_data)}")
print(f"속성들: {dir(self.current_data)}")
```

### 4. 에러 났을 때
```python
try:
    # 실행할 코드
    result = some_function()
except Exception as e:
    # 에러 메시지 출력
    print(f"에러 발생: {e}")
    import traceback
    traceback.print_exc()  # 상세 에러 위치 출력
```

---

## 우선 구현해야 할 것들

1. **io/file_handler.py**: 실제 CSV 파일 읽기
   - CSV 형식 확인 후 파싱 로직 작성

2. **core/kinematics.py**: 관절각 계산
   - Quaternion 수학 함수 구현
   - 관절각 추출 알고리즘

3. **ui/visualization_3d.py**: 3D 렌더링
   - PyQtGraph 3D 또는 OpenGL 사용
   - 분절 그리기

이 순서대로 하나씩 구현하면 됩니다!
