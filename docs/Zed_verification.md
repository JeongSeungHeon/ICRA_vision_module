# ZED Mini Single-Camera Verification Plan

## 목표
- ZED Mini 1대로 `object point cloud + 3D hand pose + grasp target`까지 계산되는지 먼저 검증한다.
- 이번 단계는 `dual camera` 완성본이 아니라, 이후 ZED Mini 2대 구성으로 확장하기 위한 단일 카메라 PoC를 만드는 것이 목적이다.
- 검증 대상은 다음 4가지다.
  - ZED depth를 기존 perception 파이프라인에 안정적으로 넣을 수 있는지
  - YOLO 기반 object mask에서 point cloud가 정상적으로 생성되는지
  - MediaPipe 2D hand keypoint를 depth로 3D hand pose로 올릴 수 있는지
  - object와 hand를 바탕으로 grasp point가 계산되고 시각화되는지

## 범위
- 이번 구현은 `cam0-only` 흐름으로 제한한다.
- robot RTDE 제어는 제외한다.
- dual-camera synchronization, hand selection, object merge는 이번 단계에서 구현하지 않는다.
- 출력은 Open3D 3D viewer와 2D overlay debug window를 기준으로 한다.

## 구현 방향
- 베이스 스크립트는 [`archive_non_rtde_main/tools/visualize_cam0_target_debug.py`](/home/ur5/ICRA_vision_module/archive_non_rtde_main/tools/visualize_cam0_target_debug.py) 로 잡는다.
- 기존 `ObjectWorkerCam0`, `HandWorkerCam0`, `GraspTargetPlanner`는 최대한 재사용한다.
- 핵심 신규 작업은 RealSense 대신 ZED Mini에서 `FrameBundle` 호환 입력을 만드는 것이다.

## 사전 확인 사항

### 1. SDK / 런타임
- ZED Python SDK (`pyzed.sl`) 사용 가능 여부를 먼저 확인한다.
- 카메라 연결 상태, 해상도, depth mode, fps가 원하는 운영 조건에서 안정적인지 확인한다.
- OpenCV, MediaPipe, Open3D와 함께 런타임 충돌이 없는지 확인한다.

### 2. 좌표계 / 단위
- ZED depth 결과 단위를 meter로 통일한다.
- left image 기준 intrinsics를 가져오는지 확인한다.
- point cloud와 hand pose가 같은 카메라 좌표계를 쓰는지 확인한다.

### 3. 캘리브레이션
- 최종 목표가 robot base visualization이라면 `cam0_to_robot` extrinsic이 필요하다.
- 아직 ZED용 extrinsic이 없으면 1차 구현은 camera frame 기준 시각화로 먼저 검증한다.
- extrinsic이 준비되면 base frame visualization으로 전환한다.

## Step 1. 단일 카메라 검증 경로를 고정한다

### 작업
- 이번 PoC의 입력 카메라를 `ZED Mini 1대`로 고정한다.
- 논리 카메라 ID는 `cam0`으로 본다.
- dual-camera 전용 흐름은 사용하지 않는다.

### 이유
- 현재 dual 구조를 억지로 유지하면 sync, selection, merge까지 같이 건드려야 해서 디버깅 비용이 커진다.
- 먼저 `cam0-only`에서 전체 perception chain이 살아나는지 확인하는 것이 더 빠르다.

### 완료 기준
- 문서/코드 상에서 이번 PoC가 `single-camera cam0-only verification`임이 명확하다.

## Step 2. ZED 입력 모듈을 만든다

### 작업
- 신규 파일 [`utils/zed_stream.py`](/home/ur5/ICRA_vision_module/utils/zed_stream.py) 를 만든다.
- ZED에서 아래 정보를 읽어 반환한다.
  - `color_image`
  - `depth_image_m`
  - `intrinsics`
  - `timestamp_ms`
  - `serial`
- 반환 형식은 기존 [`utils/realsense_stream.py`](/home/ur5/ICRA_vision_module/utils/realsense_stream.py) 의 `FrameBundle`과 호환되도록 맞춘다.

### 세부 구현
- `sl.Camera`, `sl.InitParameters`, `sl.RuntimeParameters` 초기화
- left image를 BGR numpy로 변환
- depth map 또는 `XYZRGBA` measure에서 depth를 meter 기준 `HxW float32`로 정리
- left camera intrinsics에서 `fx`, `fy`, `cx`, `cy` 구성
- timestamp를 ms로 저장

### 리스크
- depth retrieval 방식이 `VIEW.DEPTH`인지 `MEASURE.DEPTH`인지 혼동하지 않도록 주의한다.
- display용 depth image를 실제 계산에 쓰면 안 된다.

### 완료 기준
- ZED에서 읽은 1프레임이 기존 `FrameBundle` 형태로 정상 반환된다.

## Step 3. 단일 센서 허브를 만든다

### 작업
- 신규 파일 [`system/single_sensor_hub.py`](/home/ur5/ICRA_vision_module/system/single_sensor_hub.py) 를 추가한다.
- 이 허브는 dual sync 없이 최신 단일 프레임만 제공한다.

### 세부 구현
- `start()`
- `read()`
- `stop()`
- 필요 시 간단한 `SensorState` 변환

### 이유
- [`system/dual_sensor_hub.py`](/home/ur5/ICRA_vision_module/system/dual_sensor_hub.py) 는 RealSense 2대 전제를 너무 많이 갖고 있다.
- 이번 PoC에서 dual abstraction을 유지하려고 하면 오히려 불필요한 복잡도가 늘어난다.

### 완료 기준
- 메인 visualization 스크립트가 `sensor_hub.read()` 한 번으로 최신 ZED 프레임을 받을 수 있다.

## Step 4. ZED 설정 파일을 분리한다

### 작업
- 신규 파일 [`configs/handover_zed_single.yaml`](/home/ur5/ICRA_vision_module/configs/handover_zed_single.yaml) 를 만든다.
- 기본값은 기존 [`configs/handover.yaml`](/home/ur5/ICRA_vision_module/configs/handover.yaml) 를 참고하되, single-camera 검증에 필요한 값만 남긴다.

### 포함할 항목
- camera backend 종류
- width / height / fps
- depth min/max
- YOLO prompt / confidence
- hand depth lifting 파라미터
- grasp planner 파라미터
- 선택적으로 cam0-to-robot extrinsic 경로

### 완료 기준
- RealSense 설정과 섞이지 않는 ZED 전용 실행 설정이 생긴다.

## Step 5. object point cloud 생성 경로를 ZED 입력으로 검증한다

### 작업
- [`perception/object_worker.py`](/home/ur5/ICRA_vision_module/perception/object_worker.py) 의 `ObjectWorkerCam0` 를 그대로 재사용해 본다.
- 입력 프레임만 ZED `FrameBundle`로 교체한다.

### 세부 검증 항목
- YOLO segmentation이 정상 동작하는지
- mask 영역에서 depth 기반 point cloud가 생성되는지
- point 개수와 centroid가 말이 되는지
- 2D overlay에서 mask와 실제 물체 위치가 잘 맞는지

### 디버그 출력
- `object_detected`
- `point_count`
- `centroid_base` 또는 `centroid_camera`
- segmentation confidence

### 완료 기준
- 물체 1종에 대해 포인트클라우드가 안정적으로 생성된다.

## Step 6. hand 3D lifting 경로를 ZED 입력으로 검증한다

### 작업
- [`perception/hand_worker.py`](/home/ur5/ICRA_vision_module/perception/hand_worker.py) 의 `HandWorkerCam0` 를 재사용해 본다.
- ZED color와 depth를 넣어서 MediaPipe 2D keypoint -> depth lifting -> 3D hand pose까지 확인한다.

### 세부 검증 항목
- 2D landmark가 이미지 위에 안정적으로 잡히는지
- landmark depth sampling이 정상인지
- 손바닥 중심과 normal이 계산되는지
- 손을 움직였을 때 3D 위치가 과도하게 튀지 않는지

### 리스크
- depth hole이 많으면 fingertip이 잘 안 올라올 수 있다.
- 손 가까이에서 depth 노이즈가 심하면 palm pose 품질이 떨어질 수 있다.

### 완료 기준
- `HandState.valid == True` 인 프레임이 반복적으로 나오고, `palm_center_base` 또는 `palm_center_camera`가 안정적으로 계산된다.

## Step 7. cam0-only merged-like state를 만든다

### 작업
- [`archive_non_rtde_main/tools/visualize_cam0_target_debug.py`](/home/ur5/ICRA_vision_module/archive_non_rtde_main/tools/visualize_cam0_target_debug.py#L225) 의 `to_cam0_only_merged_state()`와 [`archive_non_rtde_main/tools/visualize_cam0_target_debug.py`](/home/ur5/ICRA_vision_module/archive_non_rtde_main/tools/visualize_cam0_target_debug.py#L243) 의 `to_cam0_selected_hand()` 패턴을 그대로 사용한다.
- 단일 object state와 단일 hand state를 `GraspTargetPlanner`가 받을 수 있는 형태로 맞춘다.

### 이유
- 이번 단계에서는 dual merge나 hand selection이 필요 없다.
- planner 앞단만 single-camera adapter로 만들면 grasp target 검증이 가능하다.

### 완료 기준
- object/hand 단일 상태에서 planner 입력 구조가 만들어진다.

## Step 8. grasp target 계산을 붙인다

### 작업
- [`perception/grasp_target.py`](/home/ur5/ICRA_vision_module/perception/grasp_target.py) 를 그대로 사용해 grasp point를 계산한다.
- 입력은 Step 7에서 만든 `merged_like`, `selected_hand`를 넣는다.

### 세부 검증 항목
- grasp target이 object centroid 근처에 생기는지
- hand clearance 조건을 만족할 때만 valid가 되는지
- 손이 너무 가까우면 invalid가 되는지

### 로그
- `grasp_target.valid`
- `grasp_target.target_position_base`
- `grasp_target.distance_to_centroid_m`
- `grasp_target.hand_height_clearance_m`

### 완료 기준
- 의미 있는 grasp point가 계산되어 3D scene에 표시된다.

## Step 9. ZED 전용 visualization 스크립트를 만든다

### 작업
- 신규 파일 [`archive_non_rtde_main/tools/visualize_zed_target_debug.py`](/home/ur5/ICRA_vision_module/archive_non_rtde_main/tools/visualize_zed_target_debug.py) 를 추가한다.
- 구조는 [`archive_non_rtde_main/tools/visualize_cam0_target_debug.py`](/home/ur5/ICRA_vision_module/archive_non_rtde_main/tools/visualize_cam0_target_debug.py) 를 최대한 유지한다.

### 메인 루프
1. ZED 프레임 읽기
2. object state 계산
3. hand state 계산
4. cam0-only merged-like state 생성
5. selected hand 생성
6. grasp target 계산
7. Open3D 갱신
8. 2D overlay 갱신
9. 콘솔 status 출력

### 완료 기준
- `python archive_non_rtde_main/tools/visualize_zed_target_debug.py --config configs/handover_zed_single.yaml --show-2d`
  형태로 실행 가능하다.

## Step 10. 2D debug overlay를 정리한다

### 작업
- color frame 위에 다음 정보를 겹쳐 보여준다.
  - object mask
  - hand skeleton
  - object centroid
  - grasp target 상태
  - point count / hand valid / confidence

### 이유
- 3D viewer만 보면 mask 오차와 depth misalignment를 빨리 찾기 어렵다.
- 2D overlay가 있어야 segmentation과 hand lifting 실패 원인을 바로 볼 수 있다.

### 완료 기준
- 한 화면에서 object/hand/grasp 상태를 빠르게 디버깅할 수 있다.

## Step 11. 3D scene 구성을 정리한다

### 작업
- Open3D scene에 아래 geometry를 넣는다.
  - coordinate frame
  - object point cloud
  - object centroid sphere
  - hand keypoint cloud / hand bone line set
  - selected palm center / normal
  - grasp target sphere
  - centroid-to-grasp line

### 재사용 대상
- [`archive_non_rtde_main/tools/visualize_cam0_target_debug.py`](/home/ur5/ICRA_vision_module/archive_non_rtde_main/tools/visualize_cam0_target_debug.py#L47)
- [`archive_non_rtde_main/tools/visualize_cam0_target_debug.py`](/home/ur5/ICRA_vision_module/archive_non_rtde_main/tools/visualize_cam0_target_debug.py#L115)

### 완료 기준
- scene만 봐도 object, hand, grasp 관계를 바로 이해할 수 있다.

## Step 12. 좌표계 검증을 수행한다

### 작업
- camera frame 기준으로 먼저 point cloud와 hand pose가 일관된지 확인한다.
- extrinsic이 있는 경우 base frame에서도 같은 검증을 한다.

### 체크 포인트
- object point cloud가 실제 물체 위치에 있는지
- hand skeleton이 object와 상대적으로 말이 되는 위치에 있는지
- hand를 위로 들면 height axis 방향으로 값이 일관되게 바뀌는지
- grasp target이 object 내부/주변의 적절한 위치인지

### 완료 기준
- 시각적으로도 수치적으로도 좌표계가 뒤집히지 않았다.

## Step 13. 실패 케이스를 수집한다

### 작업
- 다음 실패 케이스를 의도적으로 확인한다.
  - reflective object
  - object edge leakage
  - hand fingertip depth dropout
  - object와 hand가 가까워졌을 때 grasp invalid

### 로그로 남길 것
- segmentation confidence
- point count
- valid hand keypoint 수
- grasp invalid reason이 있으면 함께 출력

### 완료 기준
- 무엇이 잘 되고 무엇이 아직 불안한지 정리된 상태가 된다.

## Step 14. 2대 ZED 확장 전 체크리스트를 만든다

### 작업
- 단일 카메라 PoC가 끝나면 아래 항목이 준비됐는지 확인한다.
  - ZED 입력 abstraction이 분리돼 있는지
  - single-camera script가 안정적으로 동작하는지
  - config가 ZED 기준으로 분리돼 있는지
  - extrinsic 파일 포맷을 확장할 준비가 됐는지

### 다음 단계
- ZED Mini 2대용 `dual_sensor_hub_zed`
- cam0/cam1 각각의 ZED frame 처리
- object merge
- hand selection
- robot base dual-camera grasp visualization

## 권장 구현 순서
1. `utils/zed_stream.py`
2. `system/single_sensor_hub.py`
3. `configs/handover_zed_single.yaml`
4. `archive_non_rtde_main/tools/visualize_zed_target_debug.py`
5. 2D overlay / 3D scene polishing
6. extrinsic 반영

## 예상 산출물
- [`docs/Zed_verification.md`](/home/ur5/ICRA_vision_module/docs/Zed_verification.md)
- [`utils/zed_stream.py`](/home/ur5/ICRA_vision_module/utils/zed_stream.py)
- [`system/single_sensor_hub.py`](/home/ur5/ICRA_vision_module/system/single_sensor_hub.py)
- [`configs/handover_zed_single.yaml`](/home/ur5/ICRA_vision_module/configs/handover_zed_single.yaml)
- [`archive_non_rtde_main/tools/visualize_zed_target_debug.py`](/home/ur5/ICRA_vision_module/archive_non_rtde_main/tools/visualize_zed_target_debug.py)

## 완료 판단 기준
- ZED Mini 1대로 object point cloud가 생성된다.
- ZED Mini 1대로 3D hand pose가 계산된다.
- 두 결과를 바탕으로 grasp target이 계산된다.
- Open3D와 2D overlay에서 전체 결과를 동시에 볼 수 있다.
- 다음 단계로 ZED Mini 2대 확장을 진행할 만큼 입력 계층과 설정이 정리돼 있다.
