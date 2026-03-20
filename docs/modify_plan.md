# Depth-Based 3D Hand Pose Extension Plan

## 1. 현재 코드베이스 요약

### 1.1 핵심 파이프라인
- [`handpose3d.py`](/home/ur5/handpose3d/handpose3d.py): 두 입력 스트림을 읽고 MediaPipe Hands로 2D landmark 21개를 검출한 뒤, 두 카메라의 동일 landmark를 [`utils.py`](/home/ur5/handpose3d/utils.py)의 `DLT()`로 삼각측량해서 3D로 복원한다.
- [`utils.py`](/home/ur5/handpose3d/utils.py): DLT, 카메라 intrinsic/extrinsic 로딩, projection matrix 생성, keypoint 저장을 담당한다.
- [`show_3d_hands.py`](/home/ur5/handpose3d/show_3d_hands.py): 저장된 3D 결과를 matplotlib로 시각화한다.

### 1.2 현재 구조의 장점
- 프레임 루프가 단순해서 새로운 3D 추정 경로를 추가하기 쉽다.
- MediaPipe 기반 2D landmark 추출은 이미 동작 중이므로, depth 기반 3D lifting은 그 뒤 단계만 교체하거나 병렬 추가하면 된다.
- 두 카메라 extrinsic 파일 구조가 이미 있으므로, 추후 각 카메라의 depth 기반 3D를 공통 좌표계로 옮겨 fusion하는 토대가 있다.

### 1.3 현재 구조의 제약
- 입력이 `cv.VideoCapture(...)` 기반이라 depth 스트림을 직접 다루지 못한다.
- 현재 `run_mp()`는 `2D 검출`, `3D 복원`, `시각화`, `로그 저장`이 한 함수에 강하게 결합돼 있다.
- 현재 파이프라인은 중앙 `720x720` 크롭을 전제로 한다. depth 기반 deprojection에서는 크롭 이후의 intrinsics 보정이 필요하므로, 이 부분은 그대로 쓰면 위험하다.
- invalid point 표현이 `[-1, -1, -1]`이라 수치 계산과 필터링 단계에서 실수를 유발할 수 있다. depth 경로에서는 내부적으로 `NaN + validity mask`가 더 안전하다.

## 2. 목표 상태

### 2.1 최종 목표
- RealSense D435i 2대에서 각 카메라별 `color + aligned depth`를 사용해 2D hand keypoint를 각 카메라 좌표계의 3D point로 직접 복원한다.
- 이후 두 카메라의 3D hand pose를 공통 좌표계로 정렬한다.
- landmark별 Kalman filter 기반 fusion으로 최종 3D hand pose를 만든다.

### 2.2 이번 단계의 범위
- 먼저 각 카메라 단독으로 `2D hand keypoint + depth -> 3D hand pose`를 만드는 경로를 구현한다.
- 아직 fusion과 Kalman filter는 구현하지 않고, 이후를 고려한 구조 설계까지만 이번 문서에 포함한다.

## 3. 전체 구현 그림

### Phase 1. 입력 파이프라인 분리
- `RGB stereo triangulation` 경로와 `RealSense depth lifting` 경로를 동시에 수용할 수 있도록 입력 abstraction을 분리한다.
- 권장 구조:
  - `handpose3d.py`: 엔트리포인트와 실행 옵션만 담당
  - `realsense_stream.py`: RealSense color/depth 프레임 획득, 정렬, intrinsics 제공
  - `hand_detector.py` 또는 기존 함수 분리: MediaPipe Hands 실행
  - `depth_lifter.py`: depth 기반 2D->3D 복원
  - `fusion.py`: 추후 Kalman fusion
- 꼭 파일을 나눌 필요는 없지만, `run_mp()` 하나에 계속 기능을 쌓는 구조는 fusion 단계에서 유지보수 비용이 커진다.

### Phase 2. 단일 카메라 depth 기반 3D 복원
- 각 카메라에서 color frame으로 2D hand keypoint를 검출한다.
- 같은 시점의 aligned depth frame에서 각 keypoint 위치의 깊이를 robust하게 읽는다.
- color intrinsics 또는 aligned depth intrinsics를 이용해 픽셀 `(u, v, z)`를 3D `(x, y, z)`로 deprojection한다.
- 결과는 카메라별 21x3 pose로 저장한다.

### Phase 3. 공통 좌표계 정렬
- 카메라 0을 world 기준으로 둔다.
- 카메라 1에서 얻은 3D pose는 기존 extrinsic(`R`, `t`) 또는 RealSense 기반 재보정 결과를 사용해 camera0/world 좌표계로 변환한다.
- 이렇게 하면 `cam0_pose_world`, `cam1_pose_world`를 동일 기준에서 비교 가능하다.

### Phase 4. Landmark 단위 Kalman fusion
- 각 landmark마다 독립적인 3D constant-velocity Kalman filter를 둔다.
- 측정치는 최대 2개(`cam0`, `cam1`)가 들어오며, 유효한 측정만 업데이트에 사용한다.
- depth quality가 낮은 점은 measurement covariance를 크게 두고, 좋은 점은 더 강하게 반영한다.
- 초기 버전은 landmark별 독립 필터로 시작하고, 이후 skeleton constraint는 후속 단계로 미룬다.

### Phase 5. 검증 및 운영
- 정적 손 자세에서 깊이 기반 단일 카메라 3D가 안정적인지 먼저 확인한다.
- 이후 두 카메라 좌표 정렬과 fusion을 확인한다.
- 마지막으로 triangulation 결과와 depth 기반 결과를 동시에 로그해 비교 검증한다.

## 4. 아키텍처 제안

### 4.1 추천 데이터 흐름
1. RealSense 2대 초기화
2. 각 카메라에서 `color frame`, `depth frame`, `intrinsics`, `timestamp` 획득
3. color frame으로 MediaPipe Hands 수행
4. 각 landmark에 대해 depth sampling
5. 각 landmark를 3D로 deprojection
6. 카메라별 pose 품질 점수 계산
7. 추후 world transform 및 Kalman fusion
8. 시각화 / 로그 저장

### 4.2 추천 내부 데이터 구조
- `FrameBundle`
  - `color_image`
  - `depth_image`
  - `depth_scale`
  - `intrinsics`
  - `timestamp_ms`
  - `camera_id`
- `HandPose2D`
  - `points_uv`: `(21, 2)`
  - `valid_mask`: `(21,)`
- `HandPose3D`
  - `points_xyz`: `(21, 3)`
  - `valid_mask`: `(21,)`
  - `depth_values_m`: `(21,)`
  - `quality_score`: `(21,)`

### 4.3 좌표계 원칙
- 단일 카메라 depth lifting 결과는 우선 각 카메라의 local camera frame으로 유지한다.
- 일반적으로 RealSense deprojection 결과는 `x: right`, `y: down`, `z: forward` 기준이므로, 기존 삼각측량 결과와 축 방향이 다를 수 있다.
- 시각화와 fusion 전에 좌표계 정의를 문서로 고정해야 한다.

## 5. 구현 우선순위

### 5.1 1차 목표
- RealSense 한 대에서 color + depth를 읽는다.
- color 기준 2D hand keypoint를 얻는다.
- landmark 21개를 depth로 3D화한다.
- 결과를 수치와 시각화로 검증한다.

### 5.2 2차 목표
- 같은 로직을 두 카메라에 적용한다.
- 카메라별 3D pose를 동시에 출력한다.

### 5.3 3차 목표
- camera1 pose를 camera0/world 기준으로 변환한다.
- landmark별 Kalman fusion을 붙인다.

## 6. 2D Hand Keypoint를 Depth로 3D로 올리는 상세 구현 계획

### 6.1 핵심 설계 결정
- `cv.VideoCapture` 대신 `pyrealsense2`를 사용한다.
  - 이유: aligned depth, depth scale, intrinsics, timestamp를 안정적으로 얻어야 하기 때문이다.
- v1에서는 `중앙 720x720 크롭을 제거`하거나 `크롭 시 intrinsics를 함께 보정`한다.
  - 권장: v1에서는 크롭 없이 전체 해상도에서 먼저 동작시킨다.
- v1에서는 `한 프레임에 한 손`, `21개 landmark`, `카메라 local frame`까지만 확실히 만든다.

### 6.2 세부 작업 항목

#### Step 1. RealSense 입력 모듈 추가
- 새 모듈 예시: `realsense_stream.py`
- 기능:
  - 장치 serial로 카메라 2대 고정 식별
  - color/depth stream enable
  - depth를 color에 align
  - `depth_scale` 읽기
  - 현재 프레임의 color/depth numpy array 반환
  - 현재 프레임에 대응하는 intrinsics 반환
- 출력 예시:
  - `color_image: HxWx3 uint8`
  - `depth_image_m: HxW float32`
  - `intrinsics: fx, fy, cx, cy`

#### Step 2. MediaPipe 2D 검출부 분리
- 현재 `run_mp()` 내부 landmark 추출 부분을 별도 함수로 뺀다.
- 예시 함수:
  - `detect_hand_2d(color_image) -> points_uv, valid_mask, mp_result`
- 이 함수는 입력 종류와 무관하게 재사용 가능해야 한다.
- pixel 좌표는 `int`로 저장하되, 내부 계산용으로는 `float`도 유지할 수 있게 한다.

#### Step 3. Depth sampling 함수 구현
- 새 함수 예시:
  - `sample_depth_at_keypoint(depth_image_m, u, v, patch_radius=2) -> depth_m, quality`
- 단순히 `depth[v, u]` 한 점만 읽으면 노이즈와 hole에 취약하므로 패치 기반 읽기를 권장한다.
- 권장 방식:
  - `(u, v)` 주변 `5x5` 패치 추출
  - 0 또는 비정상 depth 제거
  - valid depth의 median 사용
  - valid sample 개수와 depth 분산으로 quality 산출
- 실패 규칙:
  - 유효 depth가 하나도 없으면 invalid
  - depth가 설정 범위 밖이면 invalid
- 권장 depth 범위 초기값:
  - `0.1m ~ 1.2m`
  - 실제 작업 거리 맞춰 파라미터화

#### Step 4. Depth 기반 deprojection 함수 구현
- 새 함수 예시:
  - `deproject_pixel_to_point(u, v, depth_m, intrinsics) -> xyz`
- 수식:
  - `x = (u - cx) * depth / fx`
  - `y = (v - cy) * depth / fy`
  - `z = depth`
- 또는 `pyrealsense2.rs2_deproject_pixel_to_point(...)` 사용 가능
- 이 단계의 출력은 각 카메라 기준 3D 좌표다.

#### Step 5. Landmark 21개 전체 lifting 함수 구현
- 새 함수 예시:
  - `lift_hand_pose_3d(points_uv, valid_mask_2d, depth_image_m, intrinsics) -> points_xyz, valid_mask_3d, depth_values_m, quality`
- 처리 순서:
  1. 각 landmark의 2D valid 여부 확인
  2. valid면 depth sampling
  3. valid depth면 deprojection
  4. 실패 시 `NaN` 저장 및 invalid mask 설정
- 내부 표현 권장:
  - `points_xyz`: `(21, 3)` float32
  - invalid point: `np.nan`
  - 별도 `valid_mask_3d` 유지

#### Step 6. 결과 검증용 디버그 출력 추가
- color image 위에 각 landmark의 depth(m) 텍스트를 오버레이
- 일부 landmark에 대해 `(x, y, z)`를 콘솔/파일로 출력
- 손목(landmark 0)과 MCP 관절 몇 개만 먼저 확인해도 초반 디버깅에 충분하다

#### Step 7. 저장 포맷 확장
- 현재 `write_keypoints_to_disk()`는 invalid를 `-1` 방식으로 다룬다.
- depth 경로에서는 아래 중 하나를 권장:
  - 내부는 `NaN`, 저장 시만 `-1`
  - 또는 `xyz + valid + quality`를 별도 파일로 저장
- 추천 파일:
  - `kpts_3d_cam0_depth.dat`
  - `kpts_3d_cam1_depth.dat`
  - 추후 `kpts_3d_fused.dat`

### 6.3 코드 구조상 실제 수정 포인트

#### `handpose3d.py`
- 현재의 `run_mp(input_stream1, input_stream2, P0, P1)`는 triangulation 전용이다.
- 권장 수정:
  - `run_triangulation_pipeline(...)`
  - `run_depth_pipeline(...)`
  - 공통 2D detector 함수 재사용
- 이렇게 분리하면 기존 기능을 깨지 않고 depth 경로를 추가할 수 있다.

#### `utils.py`
- 추가 후보 함수:
  - `deproject_pixel_to_point(...)`
  - `transform_points(...)`
  - `write_pose_with_mask(...)`
- 기존 DLT 관련 함수는 그대로 유지한다.

#### 신규 파일 권장
- `realsense_stream.py`
- `depth_lifter.py`
- 필요 시 `fusion.py`

## 7. 구현 시 주의사항

### 7.1 depth와 color의 정렬
- MediaPipe는 color image 기준으로 landmark를 내놓는다.
- 따라서 depth는 반드시 color 기준으로 align된 depth를 사용해야 한다.
- align 없이 raw depth 좌표계에서 바로 샘플링하면 landmark 픽셀과 depth 픽셀이 어긋난다.

### 7.2 crop과 intrinsics
- 현재 코드는 중앙 crop을 수행한다.
- depth deprojection에서 crop 후 intrinsics를 그대로 쓰면 `cx`, `cy`가 틀어져 3D가 왜곡된다.
- 해결책:
  - v1에서는 crop 제거
  - 또는 crop offset만큼 `cx`, `cy`를 수정한 cropped intrinsics 사용

### 7.3 hole / invalid depth
- 손가락 끝은 depth hole이 잘 생긴다.
- 따라서 center pixel 1개만 읽는 방식은 피한다.
- 패치 median + validity mask는 사실상 필수다.

### 7.4 temporal consistency
- depth는 프레임별로 튈 수 있다.
- fusion 이전에도 landmark별 간단한 EMA나 median smoothing이 유용할 수 있다.
- 다만 v1에서는 측정 원본을 먼저 확보하고, smoothing은 선택 옵션으로 두는 것이 낫다.

### 7.5 MediaPipe 좌표의 안정성
- 손이 프레임 끝에 걸리면 landmark가 depth 유효 영역 밖으로 나갈 수 있다.
- 픽셀 변환 시 반드시 image bounds clipping 또는 invalid 처리 규칙이 필요하다.

## 8. 테스트 계획

### 8.1 단일 카메라 기능 테스트
- 정지한 손을 카메라 앞에 두고 손목 landmark의 `z`가 실제 거리와 대략 일치하는지 확인
- 손을 앞뒤로 움직일 때 `z` 값이 자연스럽게 변하는지 확인
- 한 프레임에서 손가락 뼈 길이가 비정상적으로 튀지 않는지 확인

### 8.2 정량 sanity check
- 같은 손 자세에서 인접 프레임 landmark 거리 변화량을 계산
- `invalid ratio`, `depth variance`, `bone length variance`를 로그로 남긴다

### 8.3 triangulation과 교차 검증
- 동일 시퀀스에서 기존 triangulation 결과와 depth 기반 결과를 동시에 비교
- 절대값이 완전히 같을 필요는 없지만, 손의 상대 구조와 시간 변화는 유사해야 한다

## 9. 이후 Kalman Fusion 준비 메모

### 9.1 상태 정의
- landmark별 상태:
  - `[x, y, z, vx, vy, vz]`
- 측정:
  - `cam0_world_xyz`
  - `cam1_world_xyz`

### 9.2 측정 품질 반영
- depth patch quality가 낮으면 measurement covariance `R`를 크게 둔다.
- invalid 측정은 업데이트에서 제외한다.

### 9.3 초기 구현 권장
- 21개 landmark에 대해 독립 Kalman filter 21개
- 먼저 `position only update + constant velocity prediction`
- skeleton constraint는 후속 단계

## 10. 추천 구현 순서 요약
1. `pyrealsense2` 기반 단일 카메라 color/depth 획득
2. MediaPipe 2D 검출 함수 분리
3. depth sampling + deprojection으로 단일 카메라 3D lifting 구현
4. 디버그 오버레이와 파일 저장 추가
5. 동일 구조를 2대 카메라에 확장
6. camera1 -> world transform 추가
7. landmark별 Kalman fusion 추가

## 11. 바로 다음 작업 제안
- 다음 작업은 `Phase 1 + Phase 2`를 실제 코드로 옮기는 것이다.
- 가장 먼저 손댈 포인트는 아래 세 가지다.
  - `realsense_stream.py` 추가
  - `handpose3d.py`에서 2D detector 분리
  - `depth_lifter.py`에서 `sample_depth_at_keypoint()`와 `lift_hand_pose_3d()` 구현
