# Quick Start: Calibration + Processing Workflow

## ✅ 완성된 워크플로우

### 1️⃣ Calibration Trial 처리
```
1. File > Import → "JJY_Npose.csv" 로드
2. Process > Perform Calibration (Ctrl+K)
   - Pose type 선택: "N-pose" 또는 "T-pose"
   - 전체 duration 사용
3. Process > Save Calibration → "JJY_Npose.cal" 저장
```

**결과**: 
- 상태바 우측에 "🟢 N-pose Calibration (7 sensors)" 표시됨
- Calibration이 메모리에 로드된 상태 유지

---

### 2️⃣ 동작 Trial 처리
```
1. File > Import → "JJY_Walking.csv" 로드
2. Process Data 버튼 클릭 (또는 F5)
   - 자동으로 기존 calibration 사용
   - Kinematics 계산 수행
3. File > Save → "JJY_Walking.mcp" 저장
```

**자동 동작**:
- Calibration이 이미 로드되어 있으므로
- 다시 calibration 하지 않고
- 바로 kinematics 계산 진행

---

### 3️⃣ 다른 Trial 추가 처리
```
1. File > Import → "JJY_Running.csv"
2. Process Data (F5) - 동일한 calibration 재사용
3. File > Save → "JJY_Running.mcp"
```

---

## 📊 Calibration 상태 확인

### Status Bar 표시
- **⚫ No Calibration** (회색): Calibration 없음
- **🟢 N-pose Calibration (7 sensors)** (녹색): N-pose로 calibration 완료
- **🟢 T-pose Calibration (6 sensors)** (녹색): T-pose로 calibration 완료

---

## 🔄 재시작 후 사용

프로그램을 다시 실행한 경우:

```
1. Process > Load Calibration (Ctrl+L)
   → "JJY_Npose.cal" 선택
   → 상태바에 calibration 표시 확인

2. File > Import → 동작 trial 로드

3. Process Data (F5)
```

---

## ⚠️ 중요 사항

### Process Data 동작 방식
1. **Calibration 있음** → Kinematics만 계산
2. **Calibration 없음** → 경고 메시지 + Load Calibration 유도

### Perform Calibration vs Process Data
- **Perform Calibration**: Calibration trial에만 사용
- **Process Data**: 동작 trial에 사용 (기존 calibration 적용)

### 여러 Session 처리
- Subject나 센서 위치가 바뀌면 새로운 calibration 필요
- 같은 session 내 여러 trial은 하나의 calibration 공유

---

## 🎯 전체 흐름 예시

```
세션 시작:
├─ 1. Npose.csv Import + Perform Calibration + Save Calibration
│    → 상태: 🟢 N-pose Calibration (7 sensors)
│
├─ 2. Walking.csv Import + Process Data + Save
│    → Npose calibration 재사용
│
├─ 3. Running.csv Import + Process Data + Save
│    → Npose calibration 재사용
│
└─ 4. Stairs.csv Import + Process Data + Save
     → Npose calibration 재사용
```

모든 trial이 동일한 calibration을 사용하여 일관성 보장!
