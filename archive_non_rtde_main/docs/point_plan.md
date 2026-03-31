# Point Cloud Merge Implementation Plan

## 목표
- RealSense 2대의 color/depth 입력으로부터 YOLOE segmentation mask를 얻는다.
- 각 카메라에서 mask 영역의 depth를 3D point cloud로 deprojection한다.
- 두 카메라의 point cloud를 extrinsic parameter로 같은 좌표계에 정렬한다.
- 정렬된 point cloud를 merge하고 저장/시각화한다.

## 범위
- 우선 대상 object 1종 또는 prompt로 지정한 소수 class를 안정적으로 point cloud로 만드는 것을 1차 목표로 둔다.
- 실시간 merge까지 포함하되, 고급 정합 알고리즘(ICP, TSDF, SLAM)은 이번 범위에서 제외한다.
- 먼저 `cam0` 좌표계를 기준 좌표계로 사용한다.

## 사전 확인 사항
- [`camera_parameters/c0.dat`](/home/ur5/handpose3d/camera_parameters/c0.dat), [`camera_parameters/c1.dat`](/home/ur5/handpose3d/camera_parameters/c1.dat)의 intrinsic이 현재 설치 상태와 맞는지 확인한다.
- [`camera_parameters/rot_trans_c0.dat`](/home/ur5/handpose3d/camera_parameters/rot_trans_c0.dat), [`camera_parameters/rot_trans_c1.dat`](/home/ur5/handpose3d/camera_parameters/rot_trans_c1.dat)의 `R`, `T` 방향과 단위를 검증한다.
- 중요: depth는 meter 단위이고, 현재 extrinsic translation은 값 크기상 mm일 가능성이 높으므로 단위 통일이 필요하다.

## Step 1. 공통 데이터 구조 정리
### 작업
- segmentation 결과와 point cloud 결과를 담을 공통 구조를 정의한다.
- 예시:
  - `SegmentationInstance`: `mask`, `score`, `class_id`, `class_name`, `bbox`
  - `PointCloudFrame`: `points_xyz`, `colors_rgb`, `mask_area`, `timestamp_ms`, `serial`
- RealSense 프레임 bundle과 segmentation 결과를 연결하는 상위 흐름을 정리한다.

### 대상 파일
- [`realsense_stream.py`](/home/ur5/handpose3d/realsense_stream.py)
- 신규 파일 후보: `segmentation_types.py` 또는 `pointcloud_utils.py`

### 완료 기준
- 카메라 입력, segmentation, point cloud 단계가 같은 자료구조를 공유한다.
- 이후 단계에서 임시 tuple/list 남발 없이 확장 가능한 형태가 된다.

## Step 2. YOLOE segmentation 모듈 분리
### 작업
- 현재 [`realsense_yoloe_seg_demo.py`](/home/ur5/handpose3d/realsense_yoloe_seg_demo.py)의 모델 로딩/프롬프트/추론 부분을 재사용 가능한 모듈로 분리한다.
- 입력은 `BGR frame`, 출력은 `SegmentationInstance` 리스트로 통일한다.
- `model.set_classes(prompt_classes)`를 초기화 시 1회만 수행한다.
- 다중 object가 검출될 때의 선택 정책을 옵션화한다.
  - `highest_score`
  - `all_instances`
  - `class_filter`

### 대상 파일
- 신규 파일 권장: `segmentation_engine.py`
- [`realsense_yoloe_seg_demo.py`](/home/ur5/handpose3d/realsense_yoloe_seg_demo.py)

### 완료 기준
- 이미지 1장을 넣으면 mask/bbox/class 정보가 구조화되어 반환된다.
- 실시간 데모와 후속 point cloud 파이프라인이 같은 추론 모듈을 쓴다.

## Step 3. mask 기반 depth 샘플링 함수 구현
### 작업
- hand landmark 전용이던 [`depth_lifter.py`](/home/ur5/handpose3d/depth_lifter.py)를 일반 object mask에도 쓰도록 확장한다.
- 아래 함수들을 추가한다.
  - `mask_to_pixel_indices(mask, stride=1)`
  - `filter_depth_pixels(depth_image_m, pixel_indices, min_depth_m, max_depth_m)`
  - `deproject_pixels_to_points(pixel_indices, depth_values_m, intrinsics)`
- 너무 많은 점이 생기지 않도록 `stride` 또는 `max_points` 옵션을 둔다.

### 대상 파일
- [`depth_lifter.py`](/home/ur5/handpose3d/depth_lifter.py)

### 완료 기준
- binary mask와 aligned depth를 넣으면 카메라 local frame 기준 `Nx3` point cloud가 반환된다.
- invalid depth, NaN, 거리 범위 밖 픽셀은 제거된다.

## Step 4. mask 후처리와 노이즈 억제
### 작업
- segmentation mask 품질이 낮을 때를 대비해 간단한 후처리를 추가한다.
- 우선순위:
  - 작은 connected component 제거
  - morphology erosion/dilation 옵션
  - confidence threshold 강화
  - bounding box 내부에서 mask만 사용
- depth 쪽에서는 아래 필터를 적용한다.
  - `min_depth_m`, `max_depth_m`
  - outlier depth 제거
  - downsample 전 point count 제한

### 대상 파일
- 신규 파일 후보: `mask_utils.py`
- [`depth_lifter.py`](/home/ur5/handpose3d/depth_lifter.py)
- `segmentation_engine.py`

### 완료 기준
- 배경 누수나 점 구멍이 심한 경우에도 point cloud가 과도하게 번지지 않는다.
- 테스트 object 1개 기준으로 눈으로 봐도 shape가 유지된다.

## Step 5. 단일 카메라 point cloud 생성 파이프라인 완성
### 작업
- 한 카메라에 대해 아래 전체 흐름을 연결한다.
  - RealSense frame 획득
  - YOLOE segmentation
  - target instance 선택
  - mask 기반 point cloud 생성
  - 시각화 및 저장
- point cloud는 우선 numpy로 유지하고, 필요 시 `.ply` 저장 함수를 추가한다.
- overlay 창에는 segmentation 결과와 생성된 point 수를 표시한다.

### 대상 파일
- 신규 파일 권장: `realsense_mask_pointcloud_demo.py`
- [`realsense_stream.py`](/home/ur5/handpose3d/realsense_stream.py)
- `segmentation_engine.py`
- [`depth_lifter.py`](/home/ur5/handpose3d/depth_lifter.py)

### 완료 기준
- 단일 카메라에서 특정 prompt object에 대한 point cloud를 안정적으로 생성할 수 있다.
- `.npy` 또는 `.ply`로 저장해 외부 뷰어에서 확인할 수 있다.

## Step 6. extrinsic 해석 및 좌표 변환 유틸 구현
### 작업
- 현재 extrinsic 파일이 어떤 의미의 변환인지 명확히 정리한다.
  - `world -> cam`
  - `cam -> world`
- 변환 함수를 추가한다.
  - `transform_points(points_xyz, R, t)`
  - `invert_extrinsic(R, t)`
  - `convert_cam1_to_cam0(points_xyz_cam1, extrinsics)`
- `T` 단위를 meter로 통일한다.

### 대상 파일
- [`utils.py`](/home/ur5/handpose3d/utils.py)
- 신규 파일 후보: `extrinsics.py`

### 완료 기준
- 임의의 기준 점 또는 calibration sanity check로 좌표 변환 방향이 검증된다.
- `cam1` cloud를 `cam0` 좌표계로 보냈을 때 물리적으로 비슷한 위치에 온다.

## Step 7. 두 카메라 동시 입력과 프레임 pairing
### 작업
- RealSense 2대를 동시에 초기화하고 각 프레임을 읽는다.
- 우선은 완전 동기화가 아닌 "최근접 시점" 기반으로 pairing한다.
- 프레임마다 아래를 독립 수행한다.
  - cam0 segmentation -> local cloud
  - cam1 segmentation -> local cloud
- timestamp 차이를 overlay/log로 남긴다.

### 대상 파일
- [`realsense_stream.py`](/home/ur5/handpose3d/realsense_stream.py)
- 신규 파일 권장: `dual_realsense_manager.py`
- `realsense_mask_pointcloud_demo.py`

### 완료 기준
- 두 카메라에서 같은 object를 동시에 분리해 각각 local point cloud를 만들 수 있다.
- timestamp mismatch가 심하면 바로 알 수 있다.

## Step 8. 두 point cloud의 좌표계 정렬
### 작업
- `cam0` local cloud는 그대로 유지한다.
- `cam1` local cloud를 extrinsic으로 `cam0` 좌표계로 변환한다.
- 변환 전/후를 각각 시각화해 정렬이 맞는지 확인한다.
- 정렬 검증용으로 정적인 박스, 병, 손 등 shape가 명확한 object를 사용한다.

### 대상 파일
- `extrinsics.py` 또는 [`utils.py`](/home/ur5/handpose3d/utils.py)
- `realsense_mask_pointcloud_demo.py`

### 완료 기준
- 같은 물체에 대해 cam0/cam1 cloud가 큰 오프셋 없이 겹친다.
- 축 뒤집힘, 단위 mismatch, 방향 오류가 제거된다.

## Step 9. merge 및 downsample
### 작업
- 두 cloud를 concat해서 merged cloud를 만든다.
- 후처리:
  - voxel downsample
  - statistical outlier removal 또는 radius outlier removal
  - point 수, centroid, bbox extent 로그화
- raw aligned cloud와 filtered merged cloud를 분리해서 비교한다.

### 대상 파일
- 신규 파일 후보: `pointcloud_merge.py`
- `realsense_mask_pointcloud_demo.py`

### 완료 기준
- merged cloud가 시각적으로 단일 object shape를 유지한다.
- 중복 점과 노이즈가 어느 정도 억제된다.
- raw -> voxel -> filtered 단계별 점 개수 변화를 확인할 수 있다.

## Step 9A. 정렬 후 경계 노이즈 억제
### 작업
- 정렬은 맞지만 edge noise가 남는 경우를 위한 후속 정제를 추가한다.
- 우선순위:
  - mask erosion 또는 largest connected component
  - depth edge 주변 median/patch 기반 depth 샘플링
  - reprojection overlay 기준으로 튀는 점 진단
  - reflective surface / occlusion mismatch 사례 수집
- 정렬 문제와 depth-mask 경계 노이즈를 구분해서 디버깅한다.

### 대상 파일
- `depth_lifter.py`
- `pointcloud_utils.py`
- `realsense_dual_mask_pointcloud_demo.py`

### 완료 기준
- 컵, 병 같은 작은 물체에서 테두리 바깥으로 튀는 점이 이전보다 줄어든다.
- extrinsic 오차와 depth/mask 경계 노이즈를 별도로 판단할 수 있다.

## Step 10. 저장 포맷과 디버그 산출물 정리
### 작업
- 프레임별 저장 포맷을 정한다.
  - raw mask png
  - depth npy
  - cam0 cloud ply
  - cam1 cloud ply
  - merged cloud ply
- object class, score, timestamp, serial을 메타데이터로 남긴다.
- 나중에 오프라인 재현이 가능하도록 저장 naming 규칙을 고정한다.

### 대상 파일
- `pointcloud_merge.py`
- 신규 파일 후보: `io_pointcloud.py`

### 완료 기준
- 특정 프레임의 입력과 결과를 다시 열어 원인 분석이 가능하다.
- 실시간 파이프라인과 오프라인 검증 파이프라인이 같은 포맷을 사용한다.

## Step 11. 시각화 도구 추가
### 작업
- 최소 2종의 뷰를 제공한다.
  - 2D overlay 뷰: segmentation + point count + FPS
  - 3D 뷰: cam0, transformed cam1, merged cloud
- 실시간성 우선이면 OpenCV + Open3D 조합이 적합하다.
- 초기 버전은 Open3D가 부담되면 `.ply` 저장 후 외부 뷰어 확인으로 대체할 수 있다.

### 대상 파일
- 신규 파일 후보: `pointcloud_viewer.py`
- `realsense_mask_pointcloud_demo.py`

### 완료 기준
- 사용자가 한 실행에서 segmentation 상태와 merge 결과를 함께 점검할 수 있다.

## Step 12. 통합 실행 스크립트 완성
### 작업
- 최종 실행 스크립트에 아래 옵션을 넣는다.
  - `--model`
  - `--prompt`
  - `--serials`
  - `--min-depth-m`
  - `--max-depth-m`
  - `--stride`
  - `--save-dir`
  - `--view-3d`
  - `--merge-frame`
- 실행 예시를 README에 추가한다.

### 대상 파일
- 신규 파일 권장: `realsense_dual_mask_merge.py`
- [`README.md`](/home/ur5/handpose3d/README.md)

### 완료 기준
- 한 명령으로 2대 카메라 입력부터 segmentation, cloud 생성, extrinsic merge, 저장/시각화까지 실행된다.

## 권장 구현 순서
1. Step 2
2. Step 3
3. Step 5
4. Step 6
5. Step 7
6. Step 8
7. Step 9
8. Step 10
9. Step 11
10. Step 12

## 각 단계별 검증 체크포인트
- Step 2: prompt class를 바꾸면 segmentation target이 실제로 바뀌는지 확인
- Step 3: mask 내부 점들만 3D로 올라오는지 확인
- Step 5: 단일 카메라 `.ply`가 object 형태를 유지하는지 확인
- Step 6: extrinsic 적용 후 scale과 방향이 맞는지 확인
- Step 8: 두 cloud가 같은 object 위에 겹치는지 확인
- Step 9: merge 후 노이즈가 줄고 밀도가 개선되는지 확인

## 가장 큰 기술 리스크
- extrinsic translation 단위 mismatch
- extrinsic 방향 해석 오류
- 두 카메라 프레임 시점 차이
- segmentation mask 누수로 인한 배경 point 유입
- depth hole과 reflective surface로 인한 sparse cloud

## 첫 구현 권장 마일스톤
- Milestone 1: 단일 카메라 `prompt -> mask -> point cloud -> ply 저장`
- Milestone 2: 두 카메라 `local cloud 생성`
- Milestone 3: `cam1 -> cam0` 변환 검증
- Milestone 4: `merged cloud` 실시간 시각화
