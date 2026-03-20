# handpose3d 워크스페이스 개요

## 1) 목적
이 프로젝트는 **2대의 보정(calibrated) 카메라**와 **MediaPipe Hands**를 사용해 손의 2D 키포인트를 검출하고, 이를 삼각측량(DLT)으로 3D 좌표로 복원하는 데모입니다.

핵심 결과물은 프레임별 손 관절 21개에 대한 3D 좌표입니다.

## 2) 주요 파일 구성
- `handpose3d.py`: 메인 파이프라인 (영상 입력, 2D 키포인트 검출, 3D 복원, 화면 표시)
- `utils.py`: 수학/입출력 유틸 (DLT, 카메라 파라미터 로딩, 투영행렬 생성, 키포인트 파일 저장)
- `show_3d_hands.py`: 저장된 3D 키포인트(`kpts_3d.dat`)를 matplotlib 3D로 렌더링
- `camera_parameters/c0.dat`, `camera_parameters/c1.dat`: 카메라 내부 파라미터(intrinsic) + distortion
- `camera_parameters/rot_trans_c0.dat`, `camera_parameters/rot_trans_c1.dat`: 외부 파라미터(rotation, translation)
- `media/`: 샘플 입력 영상 및 결과 gif

## 3) 전체 파이프라인
### A. 입력 준비
1. `handpose3d.py` 실행 시 기본 입력은
   - `media/cam0_test.mp4`
   - `media/cam1_test.mp4`
2. CLI 인자로 두 개를 주면 웹캠 ID로 사용
   - 예: `python handpose3d.py 0 1`

### B. 카메라 모델 준비
1. `get_projection_matrix(camera_id)`가 아래를 조합:
   - 내부행렬 `K` (`c{camera_id}.dat`)
   - 외부행렬 `R`, `t` (`rot_trans_c{camera_id}.dat`)
2. 투영행렬 계산:
   - `P = K * [R|t]` (3x4)

### C. 프레임별 처리 (`run_mp`)
1. 두 영상 스트림에서 프레임 동시 읽기
2. 해상도/크롭 처리
   - 코드 기준 `frame_shape = [720, 1280]`
   - 너비가 720이 아니면 중앙 720x720으로 크롭
   - 주의: 보정 파라미터 해상도와 일치해야 정확한 3D 복원이 가능
3. BGR -> RGB 변환 후 MediaPipe Hands 수행
4. 각 카메라에서 손 랜드마크 21개를 픽셀 좌표 `(x, y)`로 변환
5. 검출 실패 시 예외값 사용
   - 2D: `[-1, -1]`
6. 같은 인덱스의 양안 2D 점 쌍을 DLT로 삼각측량
   - 둘 중 하나라도 `-1`이면 3D는 `[-1, -1, -1]`
7. 프레임의 3D 결과를 `(21, 3)`으로 구성
8. 시각화
   - 각 카메라 프레임에 손 랜드마크를 그려 `cv.imshow`
   - ESC(27) 입력 시 종료

### D. 종료 및 반환
- `run_mp`는 다음 배열을 반환:
  - `kpts_cam0`: `(num_frames, 21, 2)`
  - `kpts_cam1`: `(num_frames, 21, 2)`
  - `kpts_3d`: `(num_frames, 21, 3)`

## 4) 데이터 흐름(요약)
1. `VideoCapture` 입력 프레임
2. MediaPipe로 2D 키포인트(카메라0/1)
3. `DLT(P0, P1, uv0, uv1)`로 3D 포인트
4. 프레임별 21개 포인트를 `frame_p3ds`로 구성
5. 누적 버퍼 `kpts_3d`에 append
6. (선택) 파일 저장 `write_keypoints_to_disk`
7. (선택) `show_3d_hands.py`로 3D 시각화

## 5) 파일 포맷
### 카메라 내부 파라미터 파일 (`camera_parameters/c*.dat`)
- `intrinsic:` 아래 3줄: 3x3 카메라 행렬 `K`
- `distortion:` 아래 1줄: 왜곡 계수

### 외부 파라미터 파일 (`camera_parameters/rot_trans_c*.dat`)
- `R:` 아래 3줄: 회전행렬(3x3)
- `T:` 아래 3줄: 평행이동벡터(3x1)

### 키포인트 저장 파일 (`*.dat`)
- 한 줄 = 한 프레임
- 2D 파일은 각 키포인트마다 `x y`
- 3D 파일은 각 키포인트마다 `x y z`
- 프레임당 21개 포인트가 공백으로 이어짐

## 6) 시각화 파이프라인 (`show_3d_hands.py`)
1. `read_keypoints('kpts_3d.dat')`로 프레임 배열 로드
2. 좌표축 정렬을 위해 회전 적용 (`Rz`, `Rx`)
3. 손가락 연결 topology(thumb/index/middle/ring/pinkie)를 선으로 그림
4. 프레임별 이미지를 `figs/fig_*.png`로 저장

## 7) 운영상 주의사항
- 메모리: 현재 구현은 모든 프레임의 `kpts_cam0`, `kpts_cam1`, `kpts_3d`를 누적합니다. 장시간 실행 시 메모리 사용량이 증가합니다.
- 저장 코드: `handpose3d.py`의 `write_keypoints_to_disk(...)` 호출은 기본적으로 주석 처리되어 있어, 필요 시 주석 해제해야 파일이 생성됩니다.
- 정확도: 카메라 보정값과 실제 입력 해상도/크롭 정책이 다르면 3D 복원 오차가 커집니다.

## 8) 실행 예시
```bash
# 샘플 영상으로 실행
python handpose3d.py

# 웹캠(예: ID 0, 1)으로 실행
python handpose3d.py 0 1

# 저장된 3D 키포인트 시각화
python show_3d_hands.py
```
