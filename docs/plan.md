# RealSense RGB Stereo Setup Plan

## 목표
- Intel RealSense 카메라 2대를 각각 `RGB 웹캠`처럼 사용한다.
- 현재 코드 구조를 유지한 채 `handpose3d.py`와 `python_stereo_camera_calibrate/calib.py`가 동작하도록 설정한다.
- 핵심 전제는 `깊이 스트림을 쓰지 않고`, 각 RealSense의 `color stream`만 OpenCV `cv.VideoCapture(...)`로 읽는 것이다.

## 전제
- 운영체제는 Linux라고 가정한다.
- RealSense 2대가 USB 3.x 포트에 연결되어 있다.
- 각 카메라의 `RGB(color)` 스트림이 `/dev/video*`로 노출된다.
- 두 카메라는 최종 설치 위치에 고정한 뒤 캘리브레이션한다.

## 현재 코드가 기대하는 입력 방식
- 런타임은 두 개의 카메라 인덱스를 받는다.
  - 예: `python handpose3d.py 0 1`
- 캘리브레이션도 같은 방식으로 두 개의 카메라 인덱스를 쓴다.
  - 설정 파일: `python_stereo_camera_calibrate/calibration_settings.yaml`
  - 키: `camera0`, `camera1`
- 따라서 RealSense를 쓰더라도 최종적으로는 `RGB용 /dev/video 인덱스 두 개`를 찾아 넣는 것이 핵심이다.

## 주의할 점
- RealSense 1대당 `/dev/video*` 노드가 여러 개 생길 수 있다.
  - color, depth, infrared, metadata 등이 분리되어 보일 수 있다.
- 현재 코드는 `VideoCapture(int_index)`만 사용하므로, 반드시 `color stream`에 해당하는 인덱스를 골라야 한다.
- 장치를 뽑았다 다시 연결하면 인덱스가 바뀔 수 있다.
  - 이 경우 `calibration_settings.yaml`과 실행 인자를 다시 맞춰야 한다.

## 1. 환경 준비
루트 README 기준 최소 환경:

```bash
conda create -n handpose3d python=3.8 -y
conda activate handpose3d
python -m pip install --upgrade pip
python -m pip install "mediapipe==0.10.8" opencv-python matplotlib numpy
python -m pip install scipy pyyaml
```

설명:
- `handpose3d.py` 실행에는 `mediapipe`, `opencv-python`, `matplotlib`, `numpy`가 필요하다.
- `python_stereo_camera_calibrate/calib.py` 실행에는 `opencv-python`, `scipy`, `pyyaml`, `numpy`가 필요하다.
- `Python 3.8`에서는 최신 `mediapipe`가 import 단계에서 실패할 수 있으므로 `0.10.8` 고정을 권장한다.

## 2. RealSense 장치 인덱스 확인
먼저 카메라가 어떤 `/dev/video*`로 보이는지 확인한다.

```bash
ls /dev/video*
```

가능하면 아래 명령으로 각 노드가 어느 장치의 어떤 스트림인지 확인한다.

```bash
v4l2-ctl --list-devices
```

판단 기준:
- 같은 RealSense 장치 아래 여러 `/dev/video*`가 나올 수 있다.
- 그중 `RGB / color`에 해당하는 노드를 찾아야 한다.
- 가장 확실한 방법은 후보 인덱스를 하나씩 OpenCV 미리보기로 확인하는 것이다.

예시 판단 절차:
1. `/dev/video4`, `/dev/video5`, `/dev/video6`, `/dev/video7`가 보인다.
2. `v4l2-ctl --list-devices`로 어떤 두 노드가 각 카메라의 color인지 추린다.
3. 실제 미리보기에서 손/컬러 영상이 나오는 인덱스 두 개만 사용한다.

## 3. 해상도 전략 정하기
현재 코드의 기본 가정:
- `handpose3d.py`는 `1280x720`으로 캡처를 시도한다.
- 그 다음 중앙 기준으로 `720x720` 크롭을 수행한다.
- README와 코드 주석상, `캘리브레이션 해상도`와 `실제 추론 해상도/크롭 정책`이 일치해야 한다.

권장 전략:
- 두 RealSense color stream을 모두 `1280x720`으로 맞춘다.
- 캘리브레이션도 `1280x720` 기준으로 수행한다.
- 추론 시 코드가 중앙 `720x720`으로 크롭하므로, 캘리브레이션 후 실제 사용 영상에서도 손이 중앙 영역에 들어오게 카메라를 배치한다.

중요:
- 다른 해상도를 쓰고 싶다면 캘리브레이션과 추론을 모두 같은 조건으로 다시 맞춰야 한다.
- 지금은 코드 수정 없이 가는 계획이므로 `1280x720` 유지가 가장 안전하다.

## 4. 캘리브레이션 설정 파일 수정
파일:
- `python_stereo_camera_calibrate/calibration_settings.yaml`

수정 원칙:
- `camera0`, `camera1`에 RealSense `RGB` 인덱스를 넣는다.
- `frame_width`, `frame_height`는 두 카메라 모두 동일한 해상도로 맞춘다.

예시:

```yaml
camera0: 4
camera1: 6
frame_width: 1280
frame_height: 720
mono_calibration_frames: 10
stereo_calibration_frames: 10
view_resize: 1
checkerboard_box_size_scale: 3.19
checkerboard_rows: 4
checkerboard_columns: 7
cooldown: 100
```

체크 포인트:
- `camera0`, `camera1`는 예시일 뿐이다. 실제 RGB 인덱스로 바꿔야 한다.
- 체커보드 `rows`, `columns`, `box_size_scale`는 실제 출력한 패턴과 반드시 일치해야 한다.

## 5. RealSense 2대 캘리브레이션 실행
캘리브레이션은 카메라가 최종 고정된 상태에서 진행한다.

```bash
cd /home/ur5/handpose3d/python_stereo_camera_calibrate
python calib.py calibration_settings.yaml
```

절차:
1. 각 카메라 단독 프레임 수집
2. 각 카메라 intrinsic 계산
3. 두 카메라 동시 프레임 수집
4. stereo extrinsic 계산
5. 결과 검증 화면 확인

캘리브레이션 품질 기준:
- intrinsic RMSE는 가능하면 `0.3 이하`
- stereo RMSE도 가능하면 `0.3 이하`, 실무적으로 `0.5 이하`

운영 팁:
- 체커보드는 평평해야 한다.
- 두 카메라 모두 패턴이 충분히 크게 보여야 한다.
- paired frame 수집 시 패턴을 흔들지 말아야 한다.

## 6. 캘리브레이션 결과를 런타임 파일명에 맞추기
여기가 현재 저장소에서 가장 중요하다.

캘리브레이션 스크립트 출력 파일명:
- `python_stereo_camera_calibrate/camera_parameters/camera0_intrinsics.dat`
- `python_stereo_camera_calibrate/camera_parameters/camera1_intrinsics.dat`
- `python_stereo_camera_calibrate/camera_parameters/camera0_rot_trans.dat`
- `python_stereo_camera_calibrate/camera_parameters/camera1_rot_trans.dat`

반면 런타임 `handpose3d.py`는 아래 파일명을 읽는다:
- `camera_parameters/c0.dat`
- `camera_parameters/c1.dat`
- `camera_parameters/rot_trans_c0.dat`
- `camera_parameters/rot_trans_c1.dat`

따라서 캘리브레이션 후 결과 파일을 루트의 `camera_parameters/`에 아래 이름으로 복사해야 한다.

```bash
cp /home/ur5/handpose3d/python_stereo_camera_calibrate/camera_parameters/camera0_intrinsics.dat /home/ur5/handpose3d/camera_parameters/c0.dat
cp /home/ur5/handpose3d/python_stereo_camera_calibrate/camera_parameters/camera1_intrinsics.dat /home/ur5/handpose3d/camera_parameters/c1.dat
cp /home/ur5/handpose3d/python_stereo_camera_calibrate/camera_parameters/camera0_rot_trans.dat /home/ur5/handpose3d/camera_parameters/rot_trans_c0.dat
cp /home/ur5/handpose3d/python_stereo_camera_calibrate/camera_parameters/camera1_rot_trans.dat /home/ur5/handpose3d/camera_parameters/rot_trans_c1.dat
```

의미:
- 저장소에 이미 들어 있는 샘플 `camera_parameters/*`는 데모용 값이다.
- RealSense 2대로 실제 쓰려면 반드시 새 캘리브레이션 결과로 교체해야 한다.

## 7. 추론 실행
루트 디렉터리에서 실행한다.

```bash
cd /home/ur5/handpose3d
python handpose3d.py <cam0_rgb_index> <cam1_rgb_index>
```

예시:

```bash
python handpose3d.py 4 6
```

성공 조건:
- 두 창 모두 컬러 영상이 정상적으로 떠야 한다.
- 손을 넣었을 때 두 창에 MediaPipe 랜드마크가 표시되어야 한다.
- 3D 결과는 내부적으로 `frame_p3ds`에 들어간다.

## 8. 초기 검증 체크리스트
- 두 입력이 모두 `RGB 컬러 영상`인지 확인
- 두 창 해상도가 동일한지 확인
- 손이 프레임 중앙 720x720 영역 안에 주로 들어오는지 확인
- 캘리브레이션 후 복사한 파일이 루트 `camera_parameters/`에 들어 있는지 확인
- 오래 실행할 경우 메모리 사용량이 증가하는지 확인

## 9. 문제 발생 시 우선 점검 순서
### 9-1. 검은 화면 또는 프레임 읽기 실패
- 잘못된 `/dev/video` 인덱스를 넣었을 가능성이 높다.
- color가 아니라 depth/IR 노드를 잡았을 수 있다.
- USB 대역폭 또는 전원 문제가 있을 수 있다.

### 9-2. 2D 손 검출은 되는데 3D가 이상함
- 캘리브레이션 파일이 샘플 값으로 남아 있을 수 있다.
- 캘리브레이션 해상도와 실제 실행 해상도가 다를 수 있다.
- 카메라 위치가 캘리브레이션 후 바뀌었을 수 있다.

### 9-3. MediaPipe가 불안정함
- 손이 너무 멀거나 너무 작을 수 있다.
- 중앙 크롭 때문에 손이 잘릴 수 있다.
- 두 카메라의 노출/화이트밸런스 차이가 너무 클 수 있다.

## 10. 운영 권장 사항
- RealSense는 되도록 같은 모델 2대를 쓴다.
- 두 카메라는 같은 높이, 비슷한 시야, 비슷한 노출 조건으로 맞춘다.
- USB 허브보다 메인보드의 독립 포트를 우선 사용한다.
- 장치 인덱스가 자주 바뀌면 실행 전에 매번 `v4l2-ctl --list-devices`로 확인한다.

## 11. 이번 계획의 범위 밖
이번 계획은 `코드 수정 없이` RealSense 2대를 RGB 웹캠처럼 사용하는 설정 문서다.

포함하지 않는 항목:
- `pyrealsense2` 기반 직접 입력 파이프라인
- depth stream 사용
- 장치 serial 기반 고정 매핑 자동화
- `handpose3d.py`의 메모리 누수성 버퍼 누적 개선

## 최종 실행 요약
1. conda 환경 생성 후 필요한 패키지 설치
2. `v4l2-ctl --list-devices`로 각 RealSense의 RGB 인덱스 확인
3. `python_stereo_camera_calibrate/calibration_settings.yaml`에 인덱스와 해상도 반영
4. `python_stereo_camera_calibrate/calib.py calibration_settings.yaml` 실행
5. 생성된 calibration 결과 파일을 루트 `camera_parameters/`의 `c0.dat`, `c1.dat`, `rot_trans_c0.dat`, `rot_trans_c1.dat`로 복사
6. `python handpose3d.py <cam0_rgb_index> <cam1_rgb_index>` 실행
