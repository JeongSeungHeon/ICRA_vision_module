# Tracking Prediction Implementation Plan

## 목표
- `FOLLOW` 단계에서 target point가 잠깐 끊겨도 로봇 follow가 바로 멈추지 않도록 한다.
- raw target measurement가 없는 짧은 구간은 Kalman filter prediction으로 메운다.
- prediction이 너무 오래 지속되거나 상태가 바뀌면 즉시 중단하고 기존 안전 정지 로직으로 돌아간다.

## 확정한 구현 기준
- prediction은 `task_state == "FOLLOW"` 구간에만 사용한다.
- prediction 허용 시간은 최대 `0.25초`로 제한한다.
- 1차 구현은 `x, y`만 prediction하고 `z`는 기존 값 유지 방식으로 둔다.
- `PREGRASP`, `GRASPED`, `DONE`, `reset`에서는 predictor를 즉시 초기화한다.

## 왜 이렇게 가는가
- 현재 코드는 `latest_target_xyz_mm is None` 또는 target timeout이 나면 follow를 비활성화하고 `safe_stop_rtde()`로 멈춘다.
- 이 동작은 안전하지만, 짧은 perception dropout에도 바로 정지하는 문제가 있다.
- 반대로 prediction을 너무 오래 허용하면 실제 물체가 사라졌는데도 로봇이 계속 쫓을 수 있어 위험하다.
- 그래서 `FOLLOW` 중 짧은 공백만 보수적으로 메우는 구조가 가장 안전하다.

## 현재 코드 기준 문제 지점

### 1. follow 정지 조건
- [`robot_control_rtde.py`](/home/ur5/ICRA_vision_module/robot_control_rtde.py#L1017) 부근에서 `active` 여부를 판단한다.
- 아래 조건 중 하나라도 만족하면 follow가 멈춘다.
  - `latest_target_xyz_mm is None`
  - `valid_detection_streak < min_valid_count`
  - `motion_triggered == False`
  - `time.perf_counter() - latest_target_t > target_timeout_s`

### 2. 로그에서 확인된 NaN 패턴
- `FOLLOW` 중 짧은 `NaN` 구간은 실제 measurement dropout이다.
- `GRASPED` 이후 긴 `NaN` 구간은 follow 대상이 더 이상 없어지는 정상 동작 구간이다.
- 따라서 CSV 전체의 `NaN`를 모두 prediction으로 메우면 안 된다.

## 전체 구현 전략

### 핵심 아이디어
- raw measurement 경로는 그대로 유지한다.
- raw measurement가 들어오면 predictor를 `update`한다.
- raw measurement가 잠깐 끊기면 predictor를 `predict`해서 가상의 target을 만든다.
- control loop는 raw target이 없더라도 predictor output이 유효하면 그 값을 따라간다.
- prediction age가 `0.25초`를 넘으면 기존처럼 정지한다.

### 1차 구현 범위
- `x, y`만 constant-velocity Kalman filter 적용
- `z`는 마지막 유효 target의 `z`를 hold
- logging에 measured/predicted 구분 추가
- debug overlay/console에 source 표시 추가

### 2차 확장 후보
- `z`까지 prediction
- innovation gating 강화
- acceleration-aware model
- predictor debug plot/analysis script 추가

## Step 1. predictor 모듈을 분리한다

### 작업
- 신규 파일 [`perception/target_predictor.py`](/home/ur5/ICRA_vision_module/perception/target_predictor.py) 를 만든다.
- Kalman filter 구현을 `robot_control_rtde.py`에서 분리해 독립 모듈로 둔다.

### 이유
- 상태 추적 로직은 제어 루프와 분리하는 편이 테스트와 디버깅이 쉽다.
- 이후 다른 카메라 백엔드나 단일 카메라 모드에도 재사용 가능하다.

### 제안 API
```python
class TargetPredictor:
    def reset(self) -> None: ...
    def has_state(self) -> bool: ...
    def update(self, measured_xyz_mm: np.ndarray, timestamp_perf: float) -> None: ...
    def predict(self, timestamp_perf: float) -> PredictedTarget | None: ...
```

### 출력 구조 예시
```python
@dataclass
class PredictedTarget:
    xyz_mm: np.ndarray
    source: str  # "measured" or "predicted"
    prediction_age_s: float
    dt_s: float
    velocity_xy_mm_s: np.ndarray
    valid: bool
```

### 완료 기준
- 모듈 단독으로 `update -> predict -> reset` 흐름이 동작한다.

## Step 2. 상태 모델을 constant velocity로 정의한다

### 상태 벡터
- `state = [x, y, vx, vy]`
- `measurement = [x, y]`

### 이유
- 1차 목표는 짧은 dropout bridge라서 `x, y`만 먼저 안정화하는 것이 안전하다.
- `z`는 depth noise 영향이 더 크므로 처음부터 같이 예측하지 않는다.

### 상태 전이
- 시간 간격 `dt`마다
  - `x = x + vx * dt`
  - `y = y + vy * dt`
  - `vx, vy`는 유지

### measurement 반영
- raw target이 들어오면 `x, y`를 observation으로 update한다.
- `z`는 Kalman state에 넣지 않고 마지막 유효 `z`를 별도로 저장한다.

### 완료 기준
- 짧은 missing interval 동안 `xy` prediction이 연속적으로 생성된다.

## Step 3. 파라미터를 코드 상수 또는 config로 노출한다

### 1차 권장값
- `prediction_max_horizon_s = 0.25`
- `process_noise_xy = 보수적 중간값`
- `measurement_noise_xy = 로그 분산 기준 중간값`
- `max_velocity_xy_mm_s = 기존 follow max speed보다 약간 큰 값`
- `reinit_jump_threshold_mm = 큰 measurement jump를 재초기화할 기준`

### 구현 위치
- 1차는 [`robot_control_rtde.py`](/home/ur5/ICRA_vision_module/robot_control_rtde.py) argument로 추가
- 안정화 후 [`configs/handover.yaml`](/home/ur5/ICRA_vision_module/configs/handover.yaml) `robot.live_follow.prediction` 블록으로 이동 가능

### 추천 인자 예시
- `--enable-target-prediction`
- `--prediction-max-horizon-s`
- `--prediction-process-noise`
- `--prediction-measurement-noise`
- `--prediction-max-xy-speed-mm-s`
- `--prediction-reinit-jump-mm`

### 완료 기준
- prediction 관련 값이 하드코딩 한 군데에만 흩어지지 않는다.

## Step 4. FollowSharedState에 raw/predicted 상태를 분리해서 보관한다

### 작업
- 기존 `latest_target_xyz_mm`는 raw measurement 의미로 유지한다.
- 아래 필드를 추가한다.
  - `predicted_target_xyz_mm`
  - `predicted_target_t`
  - `target_source`
  - `prediction_age_s`
  - `last_measured_target_xyz_mm`

### snapshot 확장
- `get_snapshot()`에 위 필드를 포함한다.

### 이유
- 현재는 제어용 target과 측정 target이 하나로 섞여 있다.
- raw와 predicted를 분리해야 디버깅과 안전 조건이 명확해진다.

### 완료 기준
- control loop와 logger가 raw/predicted를 구분해서 읽을 수 있다.

## Step 5. 메인 perception 루프에서 predictor를 update한다

### 작업
- `shared_state.update_target(...)` 직전에 또는 내부에서 predictor update를 수행한다.
- raw grasp target이 유효하면 predictor를 update한다.
- raw grasp target이 없더라도 object가 보이는 동안은 predictor를 유지하되, update는 하지 않는다.

### 적용 위치
- [`robot_control_rtde.py`](/home/ur5/ICRA_vision_module/robot_control_rtde.py#L1857) 부근

### 세부 규칙
- `grasp_point_base is not None`이면 predictor `update`
- `grasp_point_base is None`이면 predictor는 `predict`만 가능
- `object_point_base is None`이거나 상태 전이가 발생하면 predictor `reset`

### 완료 기준
- measurement 유무에 따라 predictor 상태가 올바르게 갱신된다.

## Step 6. control loop가 predicted target을 사용할 수 있게 바꾼다

### 작업
- `robot_control_loop()`에서 `latest_target_xyz_mm`만 보지 않고, 최종 제어 대상 `control_target_xyz_mm`를 선택하도록 바꾼다.

### 우선순위
1. fresh measured target
2. valid predicted target within horizon
3. none -> 기존처럼 stop

### 적용 위치
- [`robot_control_rtde.py`](/home/ur5/ICRA_vision_module/robot_control_rtde.py#L1005) 부근

### active 조건 수정
- 현재의 `latest_target_xyz_mm is None` 조건을
  - `control_target_xyz_mm is None`
  로 바꾼다.
- prediction 중일 때는 `target_source == "predicted"`를 허용한다.
- 단, `prediction_age_s > 0.25`이면 inactive로 전환한다.

### z 처리
- `x, y`는 measured/predicted target 사용
- `z`는 마지막 measured target의 `z` 또는 기존 `fixed_z_mm` 정책 유지

### 완료 기준
- `FOLLOW` 중 1~7프레임 정도의 target dropout에도 servo가 바로 끊기지 않는다.

## Step 7. 안전한 reset 조건을 명확히 넣는다

### predictor reset이 필요한 시점
- `reset_system_to_start_state()`
- `execute_pregrasp_x_only()` 진입 직전
- `save_grasp_offset()`에서 `GRASPED` 전환 시점
- `execute_return_and_place()` 완료 후
- `follow_enabled == False`로 수동 정지했을 때

### 이유
- predictor가 살아 있으면 다음 상태에서 과거 target을 다시 쓰는 위험이 있다.

### 완료 기준
- task state가 바뀔 때 과거 prediction state가 남지 않는다.

## Step 8. innovation gating과 reinitialize 기준을 넣는다

### 작업
- 새 measurement가 들어왔을 때 prediction과 measurement 차이가 너무 크면 두 가지 중 하나를 수행한다.
  - 필터 재초기화
  - update를 거부하고 raw target만 즉시 채택

### 추천 기준
- `||measurement_xy - predicted_xy|| > reinit_jump_threshold_mm` 이면 reinitialize

### 이유
- dropout 뒤 measurement가 크게 바뀐 상황에서 예전 속도 상태를 계속 믿으면 안 된다.

### 완료 기준
- 가끔 발생하는 큰 튐에도 predictor가 오히려 불안정을 만들지 않는다.

## Step 9. 로깅을 measured vs predicted로 확장한다

### 작업
- [`robot_control_rtde.py`](/home/ur5/ICRA_vision_module/robot_control_rtde.py#L867) 부근의 CSV logger를 확장한다.

### 추가 컬럼
- `raw_target_x_mm`
- `raw_target_y_mm`
- `raw_target_z_mm`
- `control_target_x_mm`
- `control_target_y_mm`
- `control_target_z_mm`
- `target_source`
- `prediction_age_s`
- `measurement_age_s`

### 규칙
- raw measurement가 없으면 raw 컬럼은 `NaN`
- prediction을 사용하는 동안 control target 컬럼은 값 유지
- prediction도 불가능하면 control target 컬럼도 `NaN`

### 완료 기준
- 로그만 보고도 "측정 끊김", "prediction 사용", "최종 stop"을 구분할 수 있다.

## Step 10. 디버그 오버레이와 콘솔 상태를 보강한다

### 작업
- preview overlay에 아래 상태를 추가한다.
  - `target_source=measured/predicted/none`
  - `prediction_age_s`
  - `predictor=armed/reset`

### 이유
- CSV를 열지 않고도 현장에서 즉시 동작 상태를 볼 수 있어야 한다.

### 완료 기준
- follow 중 measurement가 끊겼을 때 prediction 사용 여부가 화면에 보인다.

## Step 11. 검증 순서를 짧은 단계로 나눈다

### Phase A. 오프라인 sanity check
- 기존 CSV를 읽어 간단한 replay 스크립트로 prediction bridge가 어떻게 생기는지 확인한다.
- 실제 로봇 없이 measured gap에서 predicted trajectory가 과도하게 튀지 않는지 본다.

### Phase B. 카메라만 연결한 online test
- 로봇 명령 없이 live target stream에 prediction을 붙여 로그만 수집한다.
- `FOLLOW` 중 raw dropout 때 predicted target이 잘 이어지는지 확인한다.

### Phase C. 저속 robot follow test
- 저속 조건에서 prediction 허용 시간 `0.25초`로 테스트한다.
- 짧은 dropout에 stop 빈도가 줄어드는지 확인한다.

### Phase D. failure case test
- 물체를 갑자기 가리기
- 손을 빠르게 빼기
- object detection을 일부러 끊기게 만들기
- 이때도 prediction이 과도하게 오래 유지되지 않는지 확인한다.

## Step 12. 수용 기준을 미리 정의한다

### 성공 기준
- `FOLLOW` 중 짧은 dropout에서 즉시 stop 빈도가 눈에 띄게 감소한다.
- prediction 구간 종료 후 raw measurement 재획득 시 자연스럽게 복귀한다.
- `PREGRASP`, `GRASPED`, `DONE`에서 과거 target을 다시 사용하지 않는다.
- log에서 `target_source` 전환이 기대한 대로 보인다.

### 실패 기준
- prediction 중 XY가 급격히 튄다.
- dropout이 길 때도 robot이 계속 쫓는다.
- task state 전환 후에도 predictor state가 남는다.
- raw measurement 재획득 시 overshoot가 커진다.

## 구현 순서 요약
1. `TargetPredictor` 모듈 추가
2. prediction 관련 CLI/config 파라미터 추가
3. `FollowSharedState`에 raw/predicted target 필드 추가
4. perception 루프에서 predictor update/predict 연결
5. `robot_control_loop`가 control target을 measured/predicted 중 선택하도록 수정
6. state transition/reset 시 predictor 초기화
7. CSV logger 확장
8. overlay/console debug 추가
9. 오프라인 replay 검증
10. 실기 저속 테스트 후 파라미터 조정

## 구현 시 주의할 점
- prediction은 stop을 없애는 기능이 아니라, 짧은 measurement dropout을 흡수하는 기능으로만 써야 한다.
- `GRASPED` 이후 긴 `NaN`는 정상 시나리오이므로 prediction 대상이 아니다.
- `z` prediction은 1차 구현에서 제외한다.
- 기존 `max_step_mm`, `target_timeout_s`, `motion_triggered` 로직과 충돌하지 않게 control gating 순서를 명확히 해야 한다.
