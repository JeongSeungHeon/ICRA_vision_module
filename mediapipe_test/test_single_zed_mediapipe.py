import argparse
from pathlib import Path
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp

try:
    import pyzed.sl as sl
except ImportError:
    sl = None

try:
    import yaml
except ImportError:
    yaml = None


# =========================
# Config
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]

FPS = 30
ZED_RESOLUTION = "HD720"
ZED_DEPTH_MODE = "NONE"

MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

SAVE_DIR = "debug_frames"
VIDEO_CODEC = "VP80"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "handover_zed_single.yaml"


# =========================
# MediaPipe setup
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


class SingleZedColorCamera:
    def __init__(self, serial, resolution, fps, depth_mode):
        if sl is None:
            raise RuntimeError("pyzed.sl is not installed in this Python environment")

        self.name = "zed_cam0"
        self.serial = None if serial in (None, "", "null") else str(serial)
        self.resolution = resolution
        self.fps = int(fps)
        self.depth_mode = depth_mode
        self.zed = sl.Camera()
        self.image = sl.Mat()

        init_params = sl.InitParameters()
        init_params.camera_resolution = _enum_value(sl.RESOLUTION, resolution, ZED_RESOLUTION)
        init_params.camera_fps = self.fps
        init_params.depth_mode = _enum_value(sl.DEPTH_MODE, depth_mode, ZED_DEPTH_MODE)

        if self.serial is not None:
            try:
                init_params.set_from_serial_number(int(self.serial))
            except Exception as exc:
                raise RuntimeError(f"Invalid ZED serial number `{self.serial}`.") from exc

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")

        self.serial = self._read_serial_number()

        print("[INFO] ZED camera opened")
        print(
            f"       serial={self.serial}, resolution={resolution}, "
            f"fps={self.fps}, depth_mode={depth_mode}"
        )

    def _read_serial_number(self):
        try:
            camera_info = self.zed.get_camera_information()
            serial_number = getattr(camera_info, "serial_number", None)
            if serial_number is None and hasattr(camera_info, "camera_configuration"):
                serial_number = getattr(camera_info.camera_configuration, "serial_number", None)
            return "zed" if serial_number is None else str(serial_number)
        except Exception:
            return self.serial or "zed"

    def read(self):
        if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
            return False, None

        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        frame_bgra = self.image.get_data()
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        return True, frame_bgr

    def release(self):
        self.zed.close()
        print("[INFO] ZED camera closed.")


def _enum_value(enum_group, name, default_name):
    candidate = default_name if name is None else str(name).strip().upper().replace("-", "_").replace(" ", "_")
    if hasattr(enum_group, candidate):
        return getattr(enum_group, candidate)
    fallback = str(default_name).strip().upper().replace("-", "_").replace(" ", "_")
    if hasattr(enum_group, fallback):
        return getattr(enum_group, fallback)
    raise ValueError(f"Unsupported enum value `{name}` for {enum_group}.")


def _env_int(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Single ZED MediaPipe hand debug viewer.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config for the ZED camera.")
    parser.add_argument("--serial", default=os.environ.get("ZED_SERIAL"), help="ZED serial number.")
    parser.add_argument(
        "--resolution",
        default=os.environ.get("ZED_RESOLUTION"),
        help="ZED resolution enum name, for example HD720, HD1080, or VGA.",
    )
    parser.add_argument("--fps", type=int, default=None, help="ZED camera FPS.")
    parser.add_argument(
        "--depth-mode",
        default=os.environ.get("ZED_DEPTH_MODE"),
        help="ZED depth mode enum name, for example NEURAL, PERFORMANCE, or NONE.",
    )
    parser.add_argument(
        "--max-num-hands",
        type=int,
        default=None,
        help="Maximum number of hands for MediaPipe.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=None,
        help="MediaPipe minimum detection confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=None,
        help="MediaPipe minimum tracking confidence.",
    )
    parser.add_argument("--save-dir", default=SAVE_DIR, help="Directory for recorded debug videos.")
    parser.add_argument("--no-record", action="store_true", help="Disable WebM recording.")
    return parser.parse_args()


def load_config(config_path):
    path = Path(config_path)
    if yaml is None or not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _nested_value(config, path, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def resolve_runtime_config(args):
    config = load_config(args.config)
    camera_cfg = _nested_value(config, ("cameras", "cam0"), {}) or {}
    hand_detector_cfg = _nested_value(config, ("perception", "hand", "detector"), {}) or {}
    system_fps = _nested_value(config, ("system", "sensor_fps"), FPS)

    serial = args.serial or camera_cfg.get("serial")
    resolution = args.resolution or camera_cfg.get("resolution", ZED_RESOLUTION)
    fps = args.fps or int(camera_cfg.get("fps", system_fps))
    depth_mode = args.depth_mode or ZED_DEPTH_MODE

    max_num_hands = (
        args.max_num_hands
        if args.max_num_hands is not None
        else _env_int("MAX_NUM_HANDS", int(hand_detector_cfg.get("max_num_hands", MAX_NUM_HANDS)))
    )
    min_detection_confidence = (
        args.min_detection_confidence
        if args.min_detection_confidence is not None
        else float(hand_detector_cfg.get("min_detection_confidence", MIN_DETECTION_CONFIDENCE))
    )
    min_tracking_confidence = (
        args.min_tracking_confidence
        if args.min_tracking_confidence is not None
        else float(hand_detector_cfg.get("min_tracking_confidence", MIN_TRACKING_CONFIDENCE))
    )

    return (
        serial,
        str(resolution),
        int(fps),
        str(depth_mode),
        max_num_hands,
        float(min_detection_confidence),
        float(min_tracking_confidence),
    )


def print_available_zed_devices():
    if sl is None:
        print("[WARN] pyzed.sl is not installed, so ZED devices cannot be listed.")
        return

    if not hasattr(sl.Camera, "get_device_list"):
        print("[WARN] This ZED SDK version does not support Camera.get_device_list().")
        return

    devices = sl.Camera.get_device_list()
    serials = [str(getattr(device, "serial_number", "")) for device in devices]
    serials = [serial for serial in serials if serial]

    if not serials:
        print("[WARN] No ZED devices were detected by the ZED SDK.")
        return

    print("[INFO] Detected ZED serials:")
    for serial in serials:
        print(f"       serial={serial}")


def open_video_writer(frame, fps, save_dir):
    height, width = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(save_dir, f"{timestamp}_zed_mediapipe.webm")
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")

    print(f"[REC] Saving ZED MediaPipe video to {video_path}")
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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    results = hands.process(rgb)

    rgb.flags.writeable = True
    debug_frame = frame.copy()
    detected_info = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            score = handedness.classification[0].score
            detected_info.append((label, score))

            mp_drawing.draw_landmarks(
                debug_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

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
    (
        serial,
        resolution,
        fps,
        depth_mode,
        max_num_hands,
        min_detection_confidence,
        min_tracking_confidence,
    ) = resolve_runtime_config(args)

    os.makedirs(args.save_dir, exist_ok=True)

    print(
        f"[INFO] single ZED, serial={serial}, resolution={resolution}, "
        f"fps={fps}, depth_mode={depth_mode}"
    )
    print_available_zed_devices()

    try:
        camera = SingleZedColorCamera(serial, resolution, fps, depth_mode)
    except Exception as exc:
        print(f"[ERROR] Failed to open ZED camera: {exc}")
        print("[HINT] Check ZED SDK installation, USB connection, and --serial / ZED_SERIAL.")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    prev_time = time.time()
    fps_value = 0.0
    video_writer = None
    video_path = None
    record_start_time = None
    recorded_frame_count = 0

    print("[INFO] Press q to quit.")
    if args.no_record:
        print("[INFO] Recording disabled.")
    else:
        print("[INFO] Recording single ZED MediaPipe view as WebM.")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("[WARN] Failed to read ZED frame")
                continue

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps_value = 1.0 / dt

            debug_frame, _ = process_frame(
                frame,
                hands,
                camera_name="zed_cam0",
                fps_value=fps_value,
            )

            if not args.no_record:
                if video_writer is None:
                    video_writer, video_path = open_video_writer(debug_frame, fps, args.save_dir)
                    record_start_time = time.monotonic()
                recorded_frame_count = write_realtime_video_frame(
                    video_writer,
                    debug_frame,
                    record_start_time,
                    recorded_frame_count,
                    fps,
                )

            cv2.imshow("Single ZED MediaPipe Hands", debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        if video_writer is not None:
            video_writer.release()
            duration_sec = recorded_frame_count / float(fps) if fps > 0 else 0.0
            print(f"[SAVE] Video saved: {video_path} ({duration_sec:.1f}s, {recorded_frame_count} frames)")
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
