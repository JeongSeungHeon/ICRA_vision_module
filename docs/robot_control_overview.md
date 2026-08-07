# Robot Control 상세 구조

이 문서는 현재 기본 실행 경로인 `run_fitting_final.sh`와
`robot_control_rtde_fitting_final.py`를 기준으로 UR5 RTDE 제어, FPS,
live-follow, grasp/place 시퀀스를 정리한다.

## 1. 전체 실행 흐름

`run_fitting_final.sh`는 다음 주요 옵션으로 메인 프로그램을 실행한다.

- `--enable-follow`: RTDE 연결과 `RobotWorker` 활성화
- `--follow-z`: XY와 함께 Z축도 추종
- `--select-mode highest_score`: 가장 높은 score의 물체 선택
- `--3d-debug --save-image`: 3D 디버그 프레임과 원본 이미지 저장
- `--enable-pre-release-descend-before-open`: 물체를 놓기 전 추가 하강 활성화

전체 데이터 및 제어 흐름은 다음과 같다.

```text
듀얼 RealSense
  -> object segmentation / hand pose
  -> point-cloud fusion
  -> shape fitting
  -> grasp point 생성
  -> shared target 갱신
  -> 30 Hz servoL live-follow
  -> direct grasp trigger
  -> gripper close
  -> place 위치로 moveL
  -> gripper open
  -> backoff
  -> HOME moveJ
```

`run_visualize.sh`는 로봇이나 카메라를 실시간으로 구동하지 않는다. 저장된
3D debug `.npz` 파일을 Rerun으로 재생하는 오프라인 시각화 스크립트다.

## 2. 스레드와 제어권 구조

### 2.1 Perception 메인 루프

`robot_control_rtde_fitting_final.py`의 `main()`은 다음 작업을 순서대로
실행한다.

1. 동기화된 cam0/cam1 frame pair 취득
2. 각 카메라의 object segmentation
3. 각 카메라의 hand detection/tracking
4. 두 카메라의 object point cloud 병합
5. template shape fitting
6. active hand 선택
7. object/hand fusion 및 grasp point 계산
8. `FollowSharedState.update_target()`으로 로봇 목표 갱신

이 루프는 perception 결과만 생성하며, RTDE 명령을 직접 보내지 않는다.

### 2.2 RobotWorker

`RobotWorker`는 별도의 단일 background thread에서 다음을 담당한다.

- RTDE 연결 및 상태 읽기
- live-follow `servoL` 명령
- Robotiq gripper 명령
- grasp, return, place처럼 긴 blocking sequence
- HOME 복귀, reset, emergency stop, shutdown

메인 스레드는 `RobotRequest`를 queue에 넣고 Worker가 순서대로 처리한다.
따라서 follow와 grasp/place가 동시에 로봇 제어권을 잡지 않도록 직렬화한다.

주요 Worker 상태는 다음과 같다.

```text
IDLE
INITIALIZING
FOLLOWING
GRASPING
RETURNING
PLACING
RESETTING
DONE
ERROR
STOPPING
```

소스에 `robot_control_loop()` 함수도 남아 있지만 현재 `main()`에서는
사용하지 않는다. 실제 follow는 `RobotWorker._follow_once()`가 수행한다.

## 3. FPS와 제어 주기

이 시스템에는 서로 다른 의미의 FPS/Hz가 존재한다.

| 항목 | 현재 값 | 의미 |
|---|---:|---|
| 카메라 요청 FPS | 30 FPS | RealSense color/depth 스트림 설정 |
| perception 실측 FPS | 약 6~7 FPS | segmentation, hand, fitting을 포함한 전체 메인 루프 처리율 |
| live-follow | 30 Hz | Worker가 servo 목표를 계산하고 전송하는 주기 |
| RTDE `control_hz` | 125 Hz | `RtdeController` 내부 기본 주기 설정 |

최근 runtime profile 중 하나는 다음 결과를 보였다.

- 평균 perception FPS: 약 `6.55`
- 중앙값 FPS: 약 `6.83`
- 평균 전체 loop latency: 약 `155 ms`
- 평균 frame-pair read: 약 `38 ms`
- 평균 shape fitting: 약 `32 ms`

따라서 카메라가 30 FPS로 프레임을 공급하더라도 새로운 perception target은
대략 6~7 Hz로 만들어진다. RobotWorker는 그 사이 동일하거나 예측된 target을
향해 30 Hz로 reference pose를 조금씩 이동한다.

CLI `--fps`의 기본값이 30이므로 현재는 YAML의 카메라 FPS를
`build_dual_perception_pipeline()`에서 항상 30으로 덮어쓴다.

## 4. Live-follow 시작 조건

물체가 검출되었다고 로봇이 즉시 움직이지는 않는다. 다음 조건을 차례로
만족해야 한다.

1. 유효한 object 위치를 8프레임 수집
2. 초기 reference object 위치 고정
3. 물체가 reference에서 XY 50 mm 또는 Z 50 mm 이상 이동
4. 최소 3회의 유효 detection 확보
5. HOME 이동 후 현재 TCP orientation 확보
6. follow enabled이며 pause 요청이 없는 상태

물체의 안정된 초기 위치는 별도의 4프레임 buffer로 확인한다.

- X/Y range가 각각 15 mm 미만
- Z range가 18 mm 미만

조건을 만족하면 `home_object_xyz_mm`로 저장하며, 이 위치는 grasp 후 물체를
되돌려 놓을 delivery 기준점으로 사용한다.

## 5. Target 생성과 prediction

### 5.1 Perception target

최종 target은 다음 정보로 생성된다.

- 듀얼 카메라에서 병합된 object point cloud
- fitted template geometry
- 선택된 hand 위치 및 방향
- grasp candidate와 hand clearance
- base-frame calibration transform

유효한 grasp point에는 Z 방향으로 `+10 mm` 보정을 적용한다.

### 5.2 Dynamic X offset

로봇이 처음부터 grasp point로 직행하지 않도록 X offset을 거리에 따라
변경한다.

- EEF가 멀리 있을 때: grasp/object 기준 X `-320 mm`
- XY 거리 40 mm 이하: X offset `0 mm`
- 중간 거리: `-320 mm`에서 `0 mm`까지 선형 보간

이를 통해 멀리서는 물체 앞쪽의 안전한 위치를 따라가고 가까워질수록 실제
grasp point로 접근한다.

### 5.3 Target dropout

target source는 다음 중 하나다.

- `measured`: 정상 perception 측정값
- `hand_fallback`: object가 끊겼을 때 손과 물체의 상대 위치로 복원한 값
- `predicted`: Kalman predictor가 만든 단기 예측값

raw target timeout은 0.5초다. 측정값이 stale하면 prediction이 arm된 경우
최대 0.25초 동안 predicted target을 사용한다. 수동 follow 중지나 task state
변경 시 predictor는 reset된다.

## 6. Servo 속도와 접근 방식

30 Hz follow에서 기본 최대 step은 다음과 같다.

```text
MAX_XY_SPEED_MM_S = 250
MAX_Z_SPEED_MM_S  = 250
tick당 최대 step = 250 / 30 = 약 8.33 mm
```

XY 거리가 95 mm 미만인 close range에서는 더 부드럽게 제한한다.

- XY: tick당 5 mm, 축별 약 150 mm/s
- Z: tick당 2.2 mm, 약 66 mm/s

XY 거리가 65 mm 미만이고 X/Y 오차가 모두 15 mm보다 크면, 대각선 tip
collision을 줄이기 위해 오차가 큰 dominant axis 하나를 먼저 움직인다.

이 속도 제한은 벡터 norm이 아니라 축별 clamp다. 따라서 X와 Y가 동시에
최대 step으로 움직이면 실제 XY 벡터 속도는 250 mm/s보다 커질 수 있다.

이전 명령과 새 target 차이가 0.2 mm 미만이면 불필요한 `servoL` 전송을
건너뛴다.

## 7. Workspace 안전 범위

모든 Cartesian target은 `configs/handover.yaml`의 workspace 범위로 clamp된다.

```text
X: -0.20 ~ 1.10 m
Y: -0.80 ~ 0.80 m
Z:  0.00 ~ 0.80 m
```

CLI workspace override는 의도적으로 비활성화되어 있으며 YAML 설정만
사용한다.

## 8. RTDE Controller

`robot/rtde_controller.py`의 `RtdeController`는 다음 인터페이스를 사용한다.

- `RTDEControlInterface`
  - `servoL`
  - `moveL`
  - `moveJ`
  - `servoStop`, `stopL`, `stopJ`, `speedStop`
- `RTDEReceiveInterface`
  - actual TCP pose/speed/force
  - joint position/current
  - robot 상태 feedback
- `RTDEIOInterface`
  - digital output 방식의 gripper를 사용할 때 사용

현재 Robotiq gripper는 RTDE digital output이 아니라 UR의 Robotiq daemon
socket을 사용한다.

```text
robot IP: CLI 기본값 192.168.56.101
gripper port: 63352
```

### 8.1 명령 종류

| 단계 | RTDE 명령 |
|---|---|
| live-follow | `servoL` |
| return/place/backoff | asynchronous `moveL` |
| HOME 복귀 | asynchronous `moveJ` |
| follow stop | `servoStop`, 이후 fallback으로 `stopL`/`speedStop` |

주요 설정값은 다음과 같다.

```text
live follow rate:      30 Hz
RTDE control_hz:       125 Hz
watchdog timeout:      0.50 s
moveL speed:           0.35 m/s
moveL acceleration:    0.75 m/s^2
servo lookahead_time:  0.10 s
servo gain:            300
stop acceleration:     10.0 m/s^2
```

### 8.2 Orientation

standalone 실행에서는 YAML의 fixed RPY orientation을 직접 사용하지 않는다.

1. HOME joint pose로 이동
2. 실제 TCP pose를 RTDE로 읽음
3. 현재 orientation rotvec을 저장
4. follow, return, place 동안 해당 rotvec을 고정

즉 live-follow는 위치만 갱신하고 TCP orientation은 초기 HOME orientation을
유지한다.

## 9. 좌표계 mapping

perception/project base 좌표와 실제 UR RTDE base 좌표 사이에는 다음 position
sign mapping이 적용된다.

```text
project/base -> RTDE
X: -1
Y: -1
Z: +1
```

Controller가 RTDE에서 pose를 읽을 때도 같은 sign mapping을 적용하므로
메인 로직은 project base 좌표계만 사용한다. outgoing target에는 mapping을
다시 적용하여 RTDE 좌표로 보낸다.

HOME은 Cartesian position이 아닌 joint target이므로 이 sign mapping의 영향을
받지 않는다.

## 10. Direct grasp trigger

EEF와 최신 grasp point의 차이가 다음 범위에 들어오면 grasp/place 요청을
발행한다.

```text
|dx| <= 190 mm
|dy| <=  30 mm
|dz| <=  30 mm
```

메인 perception loop가 요청 전에 한 번 검사하고, RobotWorker가 실제 실행
직전에 최신 RTDE pose로 다시 검사한다. X tolerance 190 mm는 Y/Z에 비해
상당히 큰 값이므로 tool/gripper geometry와 함께 검토해야 한다.

grasp가 시작되면 Worker는 다음 순서로 follow 제어권을 회수한다.

1. `follow_pause_requested=True`
2. `follow_enabled=False`
3. RTDE stop 명령
4. 이전 follow reference 초기화
5. gripper close 시작

## 11. Robotiq gripper와 tactile grasp

현재 설정은 Robotiq socket mode와 AnySkin tactile sensor를 사용한다.

```text
open speed:                 250
open force:                 255
close speed:                255
close force:                255
position complete:          >= 200
close timeout:              2.0 s
tactile contact threshold:  35
contact 후 extra close:     20 position counts
```

gripper close 종료 조건은 다음과 같다.

- tactile norm이 threshold를 넘고 extra close target에 도달
- gripper position이 configured threshold 이상
- position 변화가 충분히 오래 멈추는 stall 감지
- TCP force-rise 감지
- timeout 또는 cancel

stall은 최소 0.12초가 지난 뒤 position 변화가 tolerance 1 이내인 상태가
3회 연속 관찰되면 발생한다.

현재 코드의 force delta threshold는 `100000 N`이므로 force-rise 종료 조건은
사실상 비활성에 가깝다. joint current와 `grasp_verified_force_current`는
상태와 로그에는 포함되지만 `execute_gripper_close()`의 직접적인 성공 종료
조건으로 사용되지는 않는다.

tactile이 활성화되어 있어도 tactile 접촉 전에 position threshold나 stall이
먼저 발생하면 grasp 성공으로 처리될 수 있다.

## 12. Place target 계산

grasp 성공 직후 현재 object와 EEF의 차이를 저장한다.

```text
grasp_offset_xyz_mm = latest_object_xyz_mm - current_eef_xyz_mm
```

place X는 초기 home object X에서 grasp offset X를 보정해서 계산한다. 현재
코드는 Y/Z grasp offset을 동일한 방식으로 모두 반영하지 않는다.

place Z는 최근 5프레임 buffer의 다음 median을 이용한다.

```text
place_z = grasp_z_median - template_bottom_z_median + 80 mm
```

각 buffer에서 최소 3개 이상의 유효 sample이 필요하다. sample이 부족하면
초기 home object Z를 fallback으로 사용한다.

## 13. Return 및 release 시퀀스

grasp 성공 후 sequence는 다음과 같다.

1. grasp offset 및 place Z 고정
2. `return_hover` 위치로 `moveL`
3. `return_place` 위치로 `moveL`
4. tactile held-object reference 측정
5. Z 방향 tactile release descent
6. release trigger 또는 최소 Z에서 정지
7. gripper open
8. optional post-release Z move
9. X 방향 55 mm backoff
10. TCP speed 또는 `isSteady()`로 정지 확인
11. HOME joint pose로 `moveJ`

현재 `HOVER_Z_OFFSET_MM=0`이므로 `return_hover`와 `return_place` target이
실질적으로 동일하다.

### 13.1 Tactile release

현재 tactile이 활성화되어 있으므로 고정 pre-release descend 거리는 무시되고
tactile continuous descent를 사용한다.

- place 위치에서 tactile reference norm 측정
- 현재 X/Y와 orientation을 유지하며 Z 최소 10 mm까지 `moveL`
- reference 대비 tactile delta가 30을 넘으면 stop
- trigger가 없으면 최소 Z 또는 timeout 위치에서 gripper open

### 13.2 Backoff와 HOME

release 후 X 방향으로 55 mm 물러난다. 이후 실제 TCP linear speed가
`0.002 m/s` 이하이거나 RTDE `isSteady()`가 true인지 확인하고 HOME `moveJ`로
전환한다.

정지 확인 timeout은 1초이며, 현재 설정은 확인 실패 시 경고만 출력하고
HOME 이동을 계속한다.

HOME joint pose는 다음과 같다.

```text
[0 deg, -135 deg, 135 deg, 0 deg, 90 deg, 0 deg]
```

- joint speed: `0.5 rad/s`
- joint acceleration: `0.5 rad/s^2`
- 허용 오차: 관절별 `1 deg`
- 기본 timeout: `10 s`

## 14. Stop, reset, shutdown

### Follow 비활성화

직전 tick에서 servo가 활성 상태였다면 stop 명령을 한 번 보내고 follow
reference를 초기화한다.

### `r` 키

- 현재 follow/grasp sequence cancel
- gripper open
- HOME 이동
- tactile, perception, target/predictor 상태 초기화
- 설정에 따라 follow 재시작

### `s` 키

- follow stop
- metadata/video/runtime profile 저장
- 로봇은 IDLE 상태 유지

### `q` 또는 ESC

- RobotWorker에 shutdown 요청
- safe stop
- RTDE와 gripper 연결 종료
- camera/tactile thread 종료

## 15. 운영 전 확인할 코드상 불일치

### 15.1 `servoL` loop time 불일치

Worker는 약 30 Hz로 `servoL`을 호출하지만 실제 측정된 `loop_dt`를
`RtdeController.step()`에 전달하지 않는다. 따라서 Controller는 YAML의
RTDE `control_hz=125`를 사용하여 다음 값을 `servoL`에 전달한다.

```text
servoL time = 1 / 125 = 0.008 s
실제 Python 호출 간격 = 1 / 30 = 약 0.033 s
```

Controller 주석은 실제 wall-clock period를 전달해야 한다고 설명하지만 현재
호출 경로는 이를 구현하지 않는다. 실제 로봇에서 우선 확인해야 할 항목이다.

### 15.2 Watchdog의 범위

watchdog은 독립 background watchdog이 아니다. `controller.step()`이 호출될
때 전달받은 command timestamp가 오래되었는지 검사한다. 모든 정상 명령은
보내기 직전에 새 timestamp를 생성하므로, RobotWorker 자체가 정지하거나
hang되면 watchdog이 별도로 0.5초 후 stop 명령을 보내지는 않는다.

target stale과 prediction timeout은 정상 Worker loop가 계속 실행되는 동안에는
follow를 비활성화하고 stop을 전송한다.

### 15.3 Mock fallback

`use_mock_fallback: true`이므로 RTDE 라이브러리가 없거나 실제 로봇 연결이
실패하면 프로그램이 mock mode로 계속 실행될 수 있다. 운전 전에 반드시
`RobotStatus.using_mock`과 초기 연결 로그를 확인해야 한다.

### 15.4 Post-release Z 부호

설정 주석은 post-release Z 이동을 robot base `+Z` 방향 상승으로 설명하지만
현재 값은 다음과 같다.

```yaml
post_release_z_offset_m: -0.03
```

코드는 이 값을 release Z에 그대로 더하므로 Z positive-up 기준으로 30 mm
하강한다. 주석, 의도, 부호 중 어느 것이 맞는지 확인해야 한다.

### 15.5 기타 확인 사항

- tactile enabled 상태에서는 고정 `pre_release_descend_m` 설정이 무시된다.
- `object_stopped` 상태는 계산되지만 현재 follow/grasp gating에는 사용되지 않는다.
- X/Y 최대 속도는 벡터 norm이 아닌 축별 제한이다.
- `HOVER_Z_OFFSET_MM=0`이라 hover 단계가 별도 안전 높이를 만들지 않는다.
- connection 실패가 mock으로 전환되면 초기화가 성공한 것처럼 진행될 수 있다.

## 16. 주요 파일

- `run_fitting_final.sh`: 현재 live 실행 옵션
- `run_visualize.sh`: 저장된 3D debug 결과 재생
- `robot_control_rtde_fitting_final.py`: perception, shared state, Worker, grasp/place 전체 orchestration
- `robot/rtde_controller.py`: RTDE control/receive/IO와 frame mapping
- `robot/robotiq_gripper_controller.py`: Robotiq daemon socket 상위 제어
- `robot/robotiq_gripper.py`: Robotiq socket protocol
- `system/dual_sensor_hub.py`: 듀얼 RealSense frame pair 공급
- `configs/handover.yaml`: camera, safety, live-follow, RTDE, tactile, gripper 설정
- `output/runtime_profile/`: 실측 FPS 및 stage latency 기록
