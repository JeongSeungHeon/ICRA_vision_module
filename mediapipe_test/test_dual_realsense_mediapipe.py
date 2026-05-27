import argparse
from pathlib import Path
import cv2
import time
import os
from datetime import datetime

import mediapipe as mp
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import yaml
except ImportError:
    yaml = None


# =========================
# Config
# =========================
CAM0_INDEX = 0
CAM1_INDEX = 1
# OpenCV fallback용 index입니다. RealSense는 기본적으로 serial로 엽니다.

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

SAVE_DIR = "debug_frames"
VIDEO_CODEC = "VP80"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "handover.yaml"


# =========================
# MediaPipe setup
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


class OpenCVCamera:
    def __init__(self, index, name, width, height, fps):
        self.index = int(index)
        self.name = name
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open {name} at OpenCV index {self.index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"[INFO] {name} opened via OpenCV")
        print(f"       index={self.index}, size={actual_w}x{actual_h}, fps={actual_fps}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class RealSenseColorCamera:
    def __init__(self, serial, name, width, height, fps):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed in this Python environment")

        self.serial = str(serial) if serial else None
        self.name = name
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.serial:
            self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as exc:
            target = self.serial or "first available device"
            raise RuntimeError(f"Failed to start {name} RealSense color stream ({target}): {exc}") from exc

        device = self.profile.get_device()
        self.serial = device.get_info(rs.camera_info.serial_number)
        device_name = device.get_info(rs.camera_info.name)
        print(f"[INFO] {name} opened via RealSense")
        print(f"       serial={self.serial}, device={device_name}, size={width}x{height}, fps={fps}")

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
        except Exception as exc:
            print(f"[WARN] Failed to wait for {self.name} RealSense frame: {exc}")
            return False, None

        if not color_frame:
            return False, None
        return True, np.asanyarray(color_frame.get_data())

    def release(self):
        self.pipeline.stop()


def _env_int(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Dual RealSense/OpenCV MediaPipe hand debug viewer.")
    parser.add_argument(
        "--backend",
        choices=("auto", "realsense", "opencv"),
        default=os.environ.get("CAMERA_BACKEND", "auto"),
        help="Camera backend. auto tries RealSense first, then OpenCV.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config with camera serials.")
    parser.add_argument("--cam0-serial", default=os.environ.get("CAM0_SERIAL"), help="RealSense serial for cam0.")
    parser.add_argument("--cam1-serial", default=os.environ.get("CAM1_SERIAL"), help="RealSense serial for cam1.")
    parser.add_argument("--cam0-index", type=int, default=_env_int("CAM0_INDEX", CAM0_INDEX), help="OpenCV index for cam0.")
    parser.add_argument("--cam1-index", type=int, default=_env_int("CAM1_INDEX", CAM1_INDEX), help="OpenCV index for cam1.")
    parser.add_argument("--width", type=int, default=None, help="Color frame width.")
    parser.add_argument("--height", type=int, default=None, help="Color frame height.")
    parser.add_argument("--fps", type=int, default=None, help="Color frame FPS.")
    return parser.parse_args()


def load_camera_config(config_path):
    path = Path(config_path)
    if yaml is None or not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config.get("cameras", {}) or {}


def list_realsense_devices():
    if rs is None:
        return []

    devices = []
    context = rs.context()
    for device in context.query_devices():
        devices.append(
            {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
            }
        )
    return devices


def _camera_value(camera_cfg, camera_name, key, default=None):
    return (camera_cfg.get(camera_name, {}) or {}).get(key, default)


def resolve_runtime_config(args):
    camera_cfg = load_camera_config(args.config)
    system_fps = FPS

    width = args.width or int(_camera_value(camera_cfg, "cam0", "width", FRAME_WIDTH))
    height = args.height or int(_camera_value(camera_cfg, "cam0", "height", FRAME_HEIGHT))
    fps = args.fps or int(_camera_value(camera_cfg, "cam0", "fps", system_fps))

    serial0 = args.cam0_serial or _camera_value(camera_cfg, "cam0", "serial")
    serial1 = args.cam1_serial or _camera_value(camera_cfg, "cam1", "serial")
    return width, height, fps, serial0, serial1


def open_opencv_pair(args, width, height, fps):
    cap0 = None
    try:
        cap0 = OpenCVCamera(args.cam0_index, "cam0", width, height, fps)
        cap1 = OpenCVCamera(args.cam1_index, "cam1", width, height, fps)
    except Exception:
        if cap0 is not None:
            cap0.release()
        raise
    return cap0, cap1


def open_realsense_pair(serial0, serial1, width, height, fps):
    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError("No RealSense devices were detected")

    print("[INFO] Detected RealSense devices:")
    for device in devices:
        print(f"       serial={device['serial']}, name={device['name']}")

    available_serials = [device["serial"] for device in devices]
    if not serial0 and len(available_serials) >= 1:
        serial0 = available_serials[0]
    if not serial1 and len(available_serials) >= 2:
        serial1 = next((serial for serial in available_serials if serial != serial0), None)

    if not serial0 or not serial1:
        raise RuntimeError("Two RealSense serials are required for the dual-camera test")

    missing = [serial for serial in (str(serial0), str(serial1)) if serial not in available_serials]
    if missing:
        print(f"[WARN] Configured RealSense serial(s) not currently detected: {missing}")

    cap0 = None
    try:
        cap0 = RealSenseColorCamera(serial0, "cam0", width, height, fps)
        cap1 = RealSenseColorCamera(serial1, "cam1", width, height, fps)
    except Exception:
        if cap0 is not None:
            cap0.release()
        raise
    return cap0, cap1


def open_camera_pair(args, width, height, fps, serial0, serial1):
    if args.backend in ("auto", "realsense"):
        try:
            return open_realsense_pair(serial0, serial1, width, height, fps)
        except Exception as exc:
            print(f"[ERROR] RealSense backend failed: {exc}")
            if args.backend == "realsense":
                return None, None
            print("[INFO] Falling back to OpenCV camera indices.")

    if args.backend in ("auto", "opencv"):
        try:
            return open_opencv_pair(args, width, height, fps)
        except Exception as exc:
            print(f"[ERROR] OpenCV backend failed: {exc}")
            return None, None

    return None, None


def open_combined_video_writer(combined_frame, fps):
    height, width = combined_frame.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(SAVE_DIR, f"{timestamp}_cam0_cam1_combined.webm")
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")

    print(f"[REC] Saving combined video to {video_path}")
    return writer, video_path


def write_realtime_video_frame(writer, frame, record_start_time, recorded_frame_count, record_fps):
    elapsed = max(0.0, time.monotonic() - record_start_time)
    target_frame_count = max(1, int(elapsed * record_fps))

    if recorded_frame_count >= target_frame_count:
        return recorded_frame_count

    while recorded_frame_count < target_frame_count:
        writer.write(frame)
        recorded_frame_count += 1

    return recorded_frame_count


def process_frame(frame, hands, camera_name, fps_value):
    """
    frame: BGR image from OpenCV
    hands: MediaPipe Hands object
    """

    # OpenCV BGR -> MediaPipe RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    results = hands.process(rgb)

    rgb.flags.writeable = True

    debug_frame = frame.copy()

    detected_info = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = handedness.classification[0].label
            score = handedness.classification[0].score

            detected_info.append((label, score))

            # landmark drawing
            mp_drawing.draw_landmarks(
                debug_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            # 손목 landmark 기준으로 label 표시
            wrist = hand_landmarks.landmark[0]
            h, w, _ = debug_frame.shape
            x = int(wrist.x * w)
            y = int(wrist.y * h)

            cv2.putText(
                debug_frame,
                f"{label} {score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # 상단 디버그 정보
    cv2.putText(
        debug_frame,
        camera_name,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        debug_frame,
        f"FPS: {fps_value:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if len(detected_info) == 0:
        cv2.putText(
            debug_frame,
            "No hand detected",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        y0 = 95
        for i, (label, score) in enumerate(detected_info):
            cv2.putText(
                debug_frame,
                f"Hand {i}: {label}, score={score:.2f}",
                (10, y0 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return debug_frame, detected_info


def main():
    args = parse_args()
    width, height, fps, serial0, serial1 = resolve_runtime_config(args)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"[INFO] backend={args.backend}, size={width}x{height}, fps={fps}")
    if serial0 or serial1:
        print(f"[INFO] configured RealSense serials: cam0={serial0}, cam1={serial1}")

    cap0, cap1 = open_camera_pair(args, width, height, fps, serial0, serial1)
    if cap0 is None or cap1 is None:
        print("[ERROR] One or both cameras failed to open.")
        print("[HINT] For RealSense, check pyrealsense2, USB connection, and serials in configs/handover.yaml.")
        print("[HINT] Override serials with CAM0_SERIAL/CAM1_SERIAL or --cam0-serial/--cam1-serial.")
        print("[HINT] For OpenCV fallback, check /dev/video* and CAM0_INDEX/CAM1_INDEX.")
        return

    hands0 = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    hands1 = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    prev_time = time.time()
    fps_value = 0.0
    video_writer = None
    video_path = None
    record_start_time = None
    recorded_frame_count = 0

    print("[INFO] Press q to quit.")
    print("[INFO] Recording cam0+cam1 combined view as WebM.")

    try:
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0:
                print("[WARN] Failed to read cam0 frame")
                continue

            if not ret1:
                print("[WARN] Failed to read cam1 frame")
                continue

            now = time.time()
            dt = now - prev_time
            prev_time = now

            if dt > 0:
                fps_value = 1.0 / dt

            debug0, info0 = process_frame(
                frame0,
                hands0,
                camera_name="cam0",
                fps_value=fps_value,
            )

            debug1, info1 = process_frame(
                frame1,
                hands1,
                camera_name="cam1",
                fps_value=fps_value,
            )

            # 두 화면 가로로 붙여서 보기
            combined = cv2.hconcat([debug0, debug1])
            if video_writer is None:
                video_writer, video_path = open_combined_video_writer(combined, fps)
                record_start_time = time.monotonic()
            recorded_frame_count = write_realtime_video_frame(
                video_writer,
                combined,
                record_start_time,
                recorded_frame_count,
                fps,
            )

            cv2.imshow("Dual RealSense MediaPipe Hands", combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        if video_writer is not None:
            video_writer.release()
            duration_sec = recorded_frame_count / float(fps) if fps > 0 else 0.0
            print(f"[SAVE] Video saved: {video_path} ({duration_sec:.1f}s, {recorded_frame_count} frames)")
        cap0.release()
        cap1.release()
        hands0.close()
        hands1.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
