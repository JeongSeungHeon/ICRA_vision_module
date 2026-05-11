# RealSense YOLOE Tracking

RealSense 컬러/깊이 스트림에 `yoloe-26l-seg` 세그멘테이션 모델을 붙여서 실시간 tracking ID, segmentation mask, 대략적인 객체 depth, 이동 궤적을 한 번에 보여주는 스크립트입니다.

## Files

- `realsense_yoloe_tracking.py`: 메인 실행 스크립트
- `requirements.txt`: 최소 Python 의존성

## Recommended Environment

이 머신에서는 `handpose3d` conda 환경에 필요한 패키지가 이미 있습니다.

```bash
conda activate handpose3d
cd /home/ur5/yolo-seg
```

## Run

가중치 경로를 따로 주지 않으면 아래 위치들에서 `yoloe-26l-seg.pt`를 자동으로 찾습니다.

- `/home/ur5/yolo-seg/yoloe-26l-seg.pt`
- `/home/ur5/ICRA_vision_module/yoloe-26l-seg.pt`
- `/home/ur5/ICRA_vision_module _codex/yoloe-26l-seg.pt`
- `/home/ur5/Fast-FoundationStereo/yoloe-26l-seg.pt`
- `/home/ur5/rayst3r/yoloe-26l-seg.pt`

기본 실행:

```bash
python realsense_yoloe_tracking.py
```

특정 프롬프트만 추적:

```bash
python realsense_yoloe_tracking.py --prompt cup
```

depth 창까지 같이 보기:

```bash
python realsense_yoloe_tracking.py --prompt cup --show-depth
```

영상 저장:

```bash
python realsense_yoloe_tracking.py --prompt cup --save-video --output outputs/cup_tracking.mp4
```

카메라 목록 확인:

```bash
python realsense_yoloe_tracking.py --list-cameras
```

모델/설정만 빠르게 점검:

```bash
python realsense_yoloe_tracking.py --prompt cup --dry-run
```

## Notes

- 기본 tracking backend는 `--tracker-backend custom` 입니다. 이 모드는 추가 패키지 없이 동작합니다.
- Ultralytics ByteTrack을 쓰고 싶다면 `--tracker-backend ultralytics --tracker bytetrack.yaml` 로 실행하고, 환경에 `lap>=0.5.12` 가 설치되어 있어야 합니다.
- `--prompt` 를 쓰는 경우 필요한 `mobileclip2_b.ts` 가 로컬에 있으면 현재 폴더로 자동 연결해서 오프라인에서도 동작하도록 처리했습니다.
- 실행 중 `s` 키를 누르면 현재 프레임의 segmentation 마스크 적용 이미지가 `outputs/masks/` 아래에 `.png`로 저장됩니다.
- 화면에는 mask, box, track ID, confidence, 추정 depth, 최근 궤적이 같이 표시됩니다.
- RealSense 권한이나 장치 인식 문제가 있으면 `realsense-viewer` 또는 `rs-enumerate-devices`로 먼저 카메라 상태를 확인하는 편이 좋습니다.
