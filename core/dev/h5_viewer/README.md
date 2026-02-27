# H5 Viewer

간단한 HDF5 탐색기입니다. 그룹/데이터셋 전체 트리를 보고, 선택한 노드의 상세 정보와 데이터 프리뷰를 확인할 수 있습니다.

## 실행

워크스페이스 루트(`IMU_MotionCapture`)에서:

```bash
python core/dev/h5_viewer/h5_viewer.py
```

파일을 바로 지정해서 실행:

```bash
python core/dev/h5_viewer/h5_viewer.py data/combined_data_S009.h5
```

## 기능

- H5 전체 계층 트리 표시 (Group/Dataset)
- Dataset shape/dtype 표시
- Dataset 값 일부 미리보기
- Group/Dataset attributes 표시
- 선택 항목(단일/다중, 그룹 포함) 병합 CSV(단일 파일) 내보내기
- 센서 데이터 export 시 같은 trial의 `common/time`(timestamp), `common/loopcnt` 자동 병합(길이 일치 시)
- Windows 한글 경로 대응용 fallback 경로 열기 지원

## CSV Export 사용법

1. 트리에서 dataset 또는 group 항목을 선택합니다. (`Ctrl`/`Shift`로 다중 선택 가능)
2. `Export Selected CSV` 버튼을 클릭합니다.
3. 저장할 CSV 파일명을 지정하면, 선택된 dataset들이 한 개 CSV 파일로 병합 저장됩니다.

병합 규칙:
- 첫 열은 `timestamp`이며, 선택된 dataset들의 timestamp를 합집합(outer join)으로 정렬해 생성합니다.
- 각 dataset 값은 동일 timestamp 행에만 채워지고, 해당 timestamp가 없으면 빈칸으로 남습니다.
- timestamp를 찾을 수 없는 dataset은 정확도 보장을 위해 export에서 제외됩니다.
