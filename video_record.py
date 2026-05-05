"""Reusable web-assisted handover video recorder service."""

from __future__ import annotations

import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
import re

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from werkzeug.serving import make_server

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


SAVE_DIR = Path("recordings")
PENDING_DIRNAME = "pending"

DEFAULT_SERIAL = "231522072349"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30

CUP_OPTIONS = ["white cup", "red cup", "beer cup", "wine glass"]
FULLNESS_OPTIONS = ["empty", "filled"]
GRASP_OPTIONS = ["bottom", "top", "natural"]
HANDOVER_OPTIONS = ["left", "right", "center"]

FILLING_AMOUNT_MAP = {
    ("white cup", "empty"): 0,
    ("white cup", "filled"): 125,
    ("red cup", "empty"): 0,
    ("red cup", "filled"): 400,
    ("beer cup", "empty"): 0,
    ("beer cup", "filled"): 450,
    ("wine glass", "empty"): 0,
    ("wine glass", "filled"): 300,
}

HTML = """
<!doctype html>
<html>
<head>
    <title>Handover Recorder</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #111;
            color: white;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 12px;
        }
        .main {
            display: flex;
            gap: 20px;
            align-items: flex-start;
            justify-content: center;
            flex-wrap: wrap;
        }
        .video-panel {
            text-align: center;
        }
        img {
            width: 900px;
            max-width: 95vw;
            border: 2px solid #444;
            border-radius: 10px;
            margin-top: 10px;
        }
        .side-panel {
            width: 420px;
            max-width: 95vw;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 16px;
        }
        .group {
            margin-bottom: 18px;
        }
        .group-title {
            font-size: 17px;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .option-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .option-btn, .action-btn {
            border: none;
            border-radius: 10px;
            cursor: pointer;
            padding: 10px 14px;
            font-size: 15px;
        }
        .option-btn {
            background: #2b2b2b;
            color: white;
            border: 1px solid #444;
        }
        .option-btn.selected {
            background: #1976d2;
            border-color: #42a5f5;
        }
        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }
        .start { background: #2e7d32; color: white; }
        .finish { background: #6d4c41; color: white; }
        .save { background: #1565c0; color: white; }
        .discard { background: #c62828; color: white; }
        .status {
            margin-top: 16px;
            font-size: 16px;
            line-height: 1.5;
            background: #151515;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 12px;
        }
        .hint {
            margin-top: 12px;
            color: #bbb;
            font-size: 14px;
        }
        .selected-box {
            margin-top: 12px;
            background: #151515;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 12px;
            line-height: 1.7;
        }
        .warn {
            color: #ffb74d;
            margin-top: 10px;
            min-height: 22px;
        }
    </style>
</head>
<body>
    <h1>Handover Recorder</h1>

    <div class="main">
        <div class="video-panel">
            <img src="/video_feed" />
        </div>

        <div class="side-panel">
            <div class="group">
                <div class="group-title">Cup</div>
                <div class="option-grid" id="cup-options"></div>
            </div>

            <div class="group">
                <div class="group-title">Fullness</div>
                <div class="option-grid" id="fullness-options"></div>
            </div>

            <div class="group">
                <div class="group-title">Grasp Type</div>
                <div class="option-grid" id="grasp-options"></div>
            </div>

            <div class="group">
                <div class="group-title">Handover Location</div>
                <div class="option-grid" id="handover-options"></div>
            </div>

            <div class="actions">
                <button class="action-btn start" onclick="postAction('/start')">Start (c)</button>
                <button class="action-btn finish" onclick="postAction('/finish')">Finish (f)</button>
                <button class="action-btn save" onclick="postAction('/save')">Save (s)</button>
                <button class="action-btn discard" onclick="postAction('/discard')">Discard (d)</button>
            </div>

            <div class="selected-box" id="selected-box">
                current selection
            </div>

            <div class="status" id="status">status: idle</div>
            <div class="hint">
                task start in RTDE: recording starts automatically<br>
                RTDE key s: stop recording and move to pending save state<br>
                web save: choose config and finalize the file name
            </div>
            <div class="warn" id="warn"></div>
        </div>
    </div>

    <script>
        const OPTIONS = {
            cup: ["white cup", "red cup", "beer cup", "wine glass"],
            fullness: ["empty", "filled"],
            grasp_type: ["bottom", "top", "natural"],
            handover_location: ["left", "right", "center"]
        };

        let state = null;

        function pretty(v) {
            return v ?? "-";
        }

        function createOptionButtons(containerId, field, items) {
            const container = document.getElementById(containerId);
            container.innerHTML = "";

            items.forEach(item => {
                const btn = document.createElement("button");
                btn.className = "option-btn";
                btn.innerText = item;
                btn.onclick = () => setSelection(field, item);
                container.appendChild(btn);
            });
        }

        function renderSelectionButtons() {
            const mapping = {
                cup: "cup-options",
                fullness: "fullness-options",
                grasp_type: "grasp-options",
                handover_location: "handover-options",
            };

            for (const [field, containerId] of Object.entries(mapping)) {
                const buttons = document.querySelectorAll(`#${containerId} .option-btn`);
                buttons.forEach(btn => {
                    const selected = state?.selected_config?.[field] === btn.innerText;
                    btn.classList.toggle("selected", selected);
                });
            }
        }

        function renderInfo() {
            if (!state) return;

            const cfg = state.selected_config;
            const amount = state.filling_amount_ml;
            const configId = state.configuration_id;
            const currentFile = state.current_file || "-";
            const pendingFile = state.pending_file || "-";
            const predictedFile = state.predicted_file || "-";

            document.getElementById("selected-box").innerHTML = `
                <b>Selected Configuration</b><br>
                mode: ${state.mode}<br>
                configuration id: ${configId ?? "-"}<br>
                cup: ${pretty(cfg.cup)}<br>
                fullness: ${pretty(cfg.fullness)}<br>
                filling amount: ${amount !== null ? amount + " ml" : "-"}<br>
                grasp type: ${pretty(cfg.grasp_type)}<br>
                handover location: ${pretty(cfg.handover_location)}<br>
                pending file: ${pendingFile}<br>
                predicted final file: ${predictedFile}
            `;

            document.getElementById("status").innerText =
                `status: ${state.status} | mode=${state.mode} | recording=${state.recording} | pending=${state.pending_save} | frames=${state.frame_count} | file=${currentFile}`;

            const warn = document.getElementById("warn");
            if (state.pending_save && !state.selection_complete) {
                warn.innerText = "Select all 4 options before final Save.";
            } else if (state.recording) {
                warn.innerText = "Recording in progress. Use RTDE s or web Finish before Save.";
            } else {
                warn.innerText = "";
            }
        }

        async function refreshStatus() {
            const res = await fetch('/status');
            state = await res.json();
            renderSelectionButtons();
            renderInfo();
        }

        async function setSelection(field, value) {
            const res = await fetch('/set_selection', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ field, value })
            });
            state = await res.json();
            renderSelectionButtons();
            renderInfo();
        }

        async function postAction(url) {
            const res = await fetch(url, { method: 'POST' });
            const data = await res.json();
            document.getElementById('status').innerText = data.message;
            await refreshStatus();
        }

        document.addEventListener('keydown', function(event) {
            const tag = document.activeElement.tagName.toLowerCase();
            if (tag === 'input' || tag === 'textarea') return;

            if (event.key === 'c') postAction('/start');
            if (event.key === 'f') postAction('/finish');
            if (event.key === 's') postAction('/save');
            if (event.key === 'd') postAction('/discard');
        });

        createOptionButtons("cup-options", "cup", OPTIONS.cup);
        createOptionButtons("fullness-options", "fullness", OPTIONS.fullness);
        createOptionButtons("grasp-options", "grasp_type", OPTIONS.grasp_type);
        createOptionButtons("handover-options", "handover_location", OPTIONS.handover_location);

        setInterval(refreshStatus, 1000);
        refreshStatus();
    </script>
</body>
</html>
"""


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def build_config_rows():
    rows = []
    config_id = 1

    for cup in CUP_OPTIONS:
        for fullness in FULLNESS_OPTIONS:
            for grasp in GRASP_OPTIONS:
                for handover in HANDOVER_OPTIONS:
                    rows.append(
                        {
                            "configuration_id": config_id,
                            "cup": cup,
                            "fullness": fullness,
                            "filling_amount_ml": FILLING_AMOUNT_MAP[(cup, fullness)],
                            "grasp_type": grasp,
                            "handover_location": handover,
                        }
                    )
                    config_id += 1
    return rows


CONFIG_ROWS = build_config_rows()


def config_complete(cfg: dict) -> bool:
    return all(cfg.get(key) is not None for key in ("cup", "fullness", "grasp_type", "handover_location"))


def get_filling_amount(cfg: dict):
    cup = cfg.get("cup")
    fullness = cfg.get("fullness")
    return FILLING_AMOUNT_MAP.get((cup, fullness), None)


def get_configuration_row(cfg: dict):
    if not config_complete(cfg):
        return None

    for row in CONFIG_ROWS:
        if (
            row["cup"] == cfg["cup"]
            and row["fullness"] == cfg["fullness"]
            and row["grasp_type"] == cfg["grasp_type"]
            and row["handover_location"] == cfg["handover_location"]
        ):
            return row
    return None


def get_configuration_id(cfg: dict):
    row = get_configuration_row(cfg)
    return row["configuration_id"] if row is not None else None


def build_base_filename(cfg: dict, *, timestamp: datetime | None = None) -> str:
    cup = slugify(cfg["cup"])
    fullness = slugify(cfg["fullness"])
    grasp = slugify(cfg["grasp_type"])
    handover = slugify(cfg["handover_location"])
    config_id = get_configuration_id(cfg)
    timestamp = datetime.now() if timestamp is None else timestamp
    timestamp_text = timestamp.strftime("%Y%m%d_%H%M%S")

    if config_id is None:
        return f"{cup}_{fullness}_{grasp}_{handover}_{timestamp_text}"

    return f"cfg_{config_id:03d}_{cup}_{fullness}_{grasp}_{handover}_{timestamp_text}"


def get_unique_video_path(save_dir: Path, cfg: dict, *, timestamp: datetime | None = None, suffix: str | None = None) -> Path:
    base = build_base_filename(cfg, timestamp=timestamp)
    suffix_text = "" if not suffix else f"_{slugify(str(suffix))}"
    path = save_dir / f"{base}{suffix_text}.mp4"
    if not path.exists():
        return path

    idx = 1
    while True:
        candidate = save_dir / f"{base}{suffix_text}_{idx:03d}.mp4"
        if not candidate.exists():
            return candidate
        idx += 1


def get_unique_pending_path(pending_dir: Path, timestamp: datetime | None = None, suffix: str | None = None) -> Path:
    timestamp = datetime.now() if timestamp is None else timestamp
    base = f"task_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    suffix_text = "" if not suffix else f"_{slugify(str(suffix))}"
    path = pending_dir / f"{base}{suffix_text}.mp4"
    if not path.exists():
        return path

    idx = 1
    while True:
        candidate = pending_dir / f"{base}{suffix_text}_{idx:03d}.mp4"
        if not candidate.exists():
            return candidate
        idx += 1


def open_writer(path: Path, *, fps: int, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    return writer if writer.isOpened() else None


def format_record_clock(elapsed_s):
    if elapsed_s is None:
        return None
    elapsed_s = max(float(elapsed_s), 0.0)
    minutes = int(elapsed_s // 60.0)
    seconds = int(elapsed_s % 60.0)
    tenths = int((elapsed_s - int(elapsed_s)) * 10.0)
    return f"REC {minutes:02d}:{seconds:02d}.{tenths:d}"


def draw_record_clock_overlay(image_bgr, elapsed_s):
    clock_text = format_record_clock(elapsed_s)
    if not clock_text:
        return image_bgr

    origin = (18, 42)
    cv2.putText(image_bgr, clock_text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image_bgr, clock_text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 255), 2, cv2.LINE_AA)
    return image_bgr


def draw_config_id_overlay(image_bgr, config_id):
    if config_id is None:
        return image_bgr

    text = f"CFG {int(config_id):03d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    outline_thickness = 5
    text_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, text_thickness)
    x = max(image_bgr.shape[1] - text_width - 18, 12)
    y = max(text_height + 18, 24)

    cv2.putText(image_bgr, text, (x, y), font, scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(image_bgr, text, (x, y), font, scale, (255, 220, 60), text_thickness, cv2.LINE_AA)
    return image_bgr


class HandoverVideoRecorderService:
    def __init__(
        self,
        *,
        serial: str = DEFAULT_SERIAL,
        save_dir: str | Path = SAVE_DIR,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        metadata_recorder=None,
    ) -> None:
        self.serial = str(serial)
        self.save_dir = Path(save_dir)
        self.pending_dir = self.save_dir / PENDING_DIRNAME
        self.host = str(host)
        self.port = int(port)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.metadata_recorder = metadata_recorder

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._frame_ready = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._server = None
        self._server_thread = None
        self._pipeline = None
        self._latest_frame = None
        self._latest_frame_perf = None
        self._writer = None

        self._recording = False
        self._pending_save = False
        self._current_path = None
        self._last_saved_path = None
        self._task_start_timestamp_iso = None
        self._task_started_at = None
        self._pending_start_task_timestamp_iso = None
        self._pending_start_task_started_at = None
        self._frame_count = 0
        self._capture_error_count = 0
        self._last_status = "idle"

        self._selected_config = {
            "cup": None,
            "fullness": None,
            "grasp_type": None,
            "handover_location": None,
        }

        self._app = Flask(__name__)
        self._register_routes()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _require_realsense(self):
        if rs is None:
            raise RuntimeError("pyrealsense2 is required for video recording.")

    def _start_camera(self):
        if self._pipeline is not None:
            return

        self._require_realsense()
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        pipeline.start(config)
        self._pipeline = pipeline

    def start_server(self) -> str:
        with self._lock:
            if self._server_thread is not None:
                return self.url
            self._start_camera()
            self._stop_event.clear()
            self._capture_thread = threading.Thread(target=self._capture_loop, name="handover-recorder-capture", daemon=True)
            self._capture_thread.start()
            self._server = make_server(self.host, self.port, self._app, threaded=True)
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                name="handover-recorder-web",
                daemon=True,
            )
            self._server_thread.start()
            self._last_status = f"web recorder ready: {self.url}"
            url = self.url

        frame_ready = self._wait_for_frame(timeout_s=2.0)
        with self._lock:
            if frame_ready:
                self._last_status = f"web recorder ready with live frame: {url}"
            else:
                self._last_status = f"web recorder ready but waiting for first frame: {url}"
        return url

    def stop(self):
        with self._lock:
            self._stop_event.set()
            self._close_writer_locked()
            server = self._server
            capture_thread = self._capture_thread
            server_thread = self._server_thread
            pipeline = self._pipeline
            self._server = None
            self._server_thread = None
            self._capture_thread = None
            self._pipeline = None

        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass

        if server_thread is not None:
            server_thread.join(timeout=1.0)
        if capture_thread is not None:
            capture_thread.join(timeout=1.0)
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                pipeline = self._pipeline
                if pipeline is None:
                    time.sleep(0.05)
                    continue
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            except Exception as exc:
                with self._lock:
                    if not self._stop_event.is_set():
                        self._capture_error_count += 1
                        self._last_status = f"camera read failed: {exc}"
                time.sleep(0.05)
                continue

            with self._lock:
                display_frame = frame.copy()
                task_started_at = self._task_started_at
                if task_started_at is None:
                    task_started_at = self._pending_start_task_started_at
                if task_started_at is not None:
                    elapsed_s = max((datetime.now() - task_started_at).total_seconds(), 0.0)
                    display_frame = draw_record_clock_overlay(display_frame, elapsed_s)

                self._latest_frame = display_frame
                self._latest_frame_perf = time.perf_counter()
                self._frame_ready.notify_all()
                if (
                    self._pending_start_task_started_at is not None
                    and not self._recording
                    and not self._pending_save
                    and self._writer is None
                ):
                    ok, message = self._start_recording_locked(
                        task_start_timestamp_iso=self._pending_start_task_timestamp_iso,
                        task_started_at=self._pending_start_task_started_at,
                        initial_frame=display_frame,
                    )
                    self._last_status = message
                    if not ok:
                        self._pending_start_task_timestamp_iso = None
                        self._pending_start_task_started_at = None
                if self._recording and self._writer is not None:
                    self._writer.write(display_frame)
                    self._frame_count += 1

    def _wait_for_frame(self, timeout_s=2.0, *, min_frame_perf=None):
        deadline = time.perf_counter() + max(float(timeout_s), 0.0)
        with self._frame_ready:
            while not self._stop_event.is_set():
                frame_ready = self._latest_frame is not None
                perf_ready = min_frame_perf is None or (
                    self._latest_frame_perf is not None and float(self._latest_frame_perf) >= float(min_frame_perf)
                )
                if frame_ready and perf_ready:
                    return True
                remaining_s = deadline - time.perf_counter()
                if remaining_s <= 0.0:
                    break
                self._frame_ready.wait(timeout=remaining_s)
            return False

    def refresh_camera_stream(self, timeout_s=2.0):
        restart_perf = time.perf_counter()
        with self._lock:
            old_pipeline = self._pipeline
            self._pipeline = None
            self._latest_frame = None
            self._latest_frame_perf = None
            self._capture_error_count = 0

        if old_pipeline is not None:
            try:
                old_pipeline.stop()
            except Exception:
                pass

        with self._lock:
            self._start_camera()
            self._last_status = "camera stream refreshed; waiting for live frame"

        frame_ready = self._wait_for_frame(timeout_s=timeout_s, min_frame_perf=restart_perf)
        with self._lock:
            if frame_ready:
                self._last_status = "camera stream refreshed with live frame"
            else:
                self._last_status = "camera stream refresh timed out waiting for live frame"
        return frame_ready

    def _close_writer_locked(self):
        if self._writer is not None:
            try:
                self._writer.release()
            finally:
                self._writer = None

    def _render_pending_video_with_config_overlay(self, pending_path: Path, final_path: Path, config_id):
        capture = cv2.VideoCapture(str(pending_path))
        if not capture.isOpened():
            raise RuntimeError(f"failed to open pending video: {pending_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 1e-6 else float(self.fps)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = width if width > 0 else int(self.width)
        height = height if height > 0 else int(self.height)

        writer = open_writer(final_path, fps=int(round(fps)), width=width, height=height)
        if writer is None:
            capture.release()
            raise RuntimeError(f"failed to open final writer: {final_path}")

        wrote_frames = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                output_frame = draw_config_id_overlay(frame, config_id)
                writer.write(output_frame)
                wrote_frames += 1
        finally:
            capture.release()
            writer.release()

        if wrote_frames <= 0:
            if final_path.exists():
                final_path.unlink()
            raise RuntimeError("pending video contained no readable frames")

    def _infer_mode_locked(self) -> str:
        if self._recording:
            return "recording"
        if self._pending_start_task_started_at is not None:
            return "waiting_for_frame"
        if self._pending_save:
            return "pending_save"
        return "idle"

    def _start_recording_locked(self, *, task_start_timestamp_iso, task_started_at, initial_frame=None):
        pending_path = get_unique_pending_path(self.pending_dir, task_started_at)
        writer = open_writer(
            pending_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        if writer is None:
            return False, "failed to open writer"

        self._writer = writer
        self._recording = True
        self._pending_save = False
        self._current_path = pending_path
        self._frame_count = 0
        self._task_started_at = task_started_at
        self._task_start_timestamp_iso = task_start_timestamp_iso
        self._pending_start_task_timestamp_iso = None
        self._pending_start_task_started_at = None
        if initial_frame is not None:
            self._writer.write(initial_frame)
            self._frame_count = 1
        return True, f"recording in progress: {pending_path.name}"

    def get_configuration_id(self):
        with self._lock:
            return get_configuration_id(self._selected_config)

    def set_selection(self, field: str, value: str):
        valid_map = {
            "cup": CUP_OPTIONS,
            "fullness": FULLNESS_OPTIONS,
            "grasp_type": GRASP_OPTIONS,
            "handover_location": HANDOVER_OPTIONS,
        }
        with self._lock:
            if field not in valid_map:
                self._last_status = "invalid field"
                return False
            if value not in valid_map[field]:
                self._last_status = "invalid value"
                return False
            self._selected_config[field] = value
            self._last_status = f"selected {field}: {value}"
            return True

    def get_status(self):
        with self._lock:
            mode = self._infer_mode_locked()
            selection = dict(self._selected_config)
            selection_complete = config_complete(selection)
            filling_amount = get_filling_amount(selection)
            config_id = get_configuration_id(selection)
            predicted_file = "-"
            if selection_complete and self._task_started_at is not None:
                predicted_file = f"{build_base_filename(selection, timestamp=self._task_started_at)}.mp4"

            last_frame_age_ms = None
            if self._latest_frame_perf is not None:
                last_frame_age_ms = int(max((time.perf_counter() - self._latest_frame_perf) * 1000.0, 0.0))

            return {
                "mode": mode,
                "recording": self._recording,
                "pending_save": self._pending_save,
                "pending_start": self._pending_start_task_started_at is not None,
                "frame_count": int(self._frame_count),
                "current_file": self._current_path.name if self._current_path is not None else "-",
                "pending_file": self._current_path.name if self._pending_save and self._current_path is not None else "-",
                "last_saved_file": self._last_saved_path.name if self._last_saved_path is not None else "-",
                "status": self._last_status,
                "selected_config": selection,
                "selection_complete": selection_complete,
                "filling_amount_ml": filling_amount,
                "configuration_id": config_id,
                "predicted_file": predicted_file,
                "task_start_timestamp_iso": self._task_start_timestamp_iso,
                "has_live_frame": self._latest_frame is not None,
                "last_frame_age_ms": last_frame_age_ms,
                "capture_error_count": int(self._capture_error_count),
                "url": self.url,
            }

    def start_recording_for_task(self, *, task_start_timestamp_iso: str | None = None):
        with self._lock:
            if self._pending_save:
                self._last_status = "resolve pending recording before starting a new task"
                return False, self._last_status
            if self._recording:
                self._last_status = "already recording"
                return False, self._last_status
            if self._pending_start_task_started_at is not None:
                self._last_status = "recording start already armed; waiting for live frame"
                return True, self._last_status

            task_started_at = datetime.now()
            initial_frame = None
            if self._latest_frame is not None and self._latest_frame_perf is not None:
                frame_age_s = max(time.perf_counter() - self._latest_frame_perf, 0.0)
                if frame_age_s <= 1.0:
                    initial_frame = self._latest_frame.copy()

            if initial_frame is not None:
                ok, message = self._start_recording_locked(
                    task_start_timestamp_iso=task_start_timestamp_iso,
                    task_started_at=task_started_at,
                    initial_frame=initial_frame,
                )
                self._last_status = message
                return ok, self._last_status

            self._pending_start_task_timestamp_iso = task_start_timestamp_iso
            self._pending_start_task_started_at = task_started_at
            self._last_status = "recording armed: waiting for live frame"
            return True, self._last_status

    def finish_recording_to_pending(self):
        with self._lock:
            if self._pending_save:
                self._last_status = "ready to save pending recording"
                return True, self._last_status
            if not self._recording:
                if self._pending_start_task_started_at is not None:
                    self._last_status = "recording has not started yet; still waiting for live frame"
                    return False, self._last_status
                self._last_status = "no active recording to finish"
                return False, self._last_status

            if self._frame_count <= 0 and self._latest_frame is not None and self._writer is not None:
                self._writer.write(self._latest_frame)
                self._frame_count = 1

            self._recording = False
            self._pending_save = True
            pending_name = self._current_path.name if self._current_path is not None else "-"
            saved_frames = int(self._frame_count)
            self._close_writer_locked()
            self._last_status = f"ready to save pending recording: {pending_name} ({saved_frames} frames)"
            return True, self._last_status

    def save_pending_recording(self):
        with self._lock:
            if self._recording:
                self._last_status = "finish recording before final save"
                return False, self._last_status
            if not self._pending_save or self._current_path is None:
                self._last_status = "no pending recording to save"
                return False, self._last_status
            if not config_complete(self._selected_config):
                self._last_status = "select all options before final save"
                return False, self._last_status

            pending_path = self._current_path
            selected_config = dict(self._selected_config)
            task_started_at = self._task_started_at or datetime.now()
            task_start_timestamp_iso = self._task_start_timestamp_iso
            config_id = get_configuration_id(selected_config)
            final_path = get_unique_video_path(self.save_dir, selected_config, timestamp=task_started_at)
            pending_size = pending_path.stat().st_size if pending_path.exists() else 0
            if self._frame_count <= 0 or pending_size <= 512:
                self._last_status = (
                    "cannot finalize saved video: pending recording has no frames. "
                    "Wait for the live preview before starting or finish later."
                )
                return False, self._last_status

        try:
            self._render_pending_video_with_config_overlay(pending_path, final_path, config_id)
        except Exception as exc:
            if final_path.exists():
                final_path.unlink()
            with self._lock:
                self._last_status = f"failed to finalize saved video: {exc}"
            return False, self._last_status

        if pending_path.exists():
            pending_path.unlink()
        if self.metadata_recorder is not None and config_id is not None:
            try:
                self.metadata_recorder.attach_config_id(
                    config_id,
                    task_start_timestamp_iso=task_start_timestamp_iso,
                )
            except Exception as exc:
                with self._lock:
                    self._last_status = f"saved video, metadata update failed: {exc}"
                    self._pending_save = False
                    self._current_path = None
                    self._last_saved_path = final_path
                    self._frame_count = 0
                return True, self._last_status

        with self._lock:
            self._pending_save = False
            self._current_path = None
            self._last_saved_path = final_path
            self._frame_count = 0
            self._last_status = f"saved: {final_path.name}"
            return True, self._last_status

    def discard_pending_recording(self):
        with self._lock:
            path = self._current_path
            was_recording = self._recording
            had_pending = self._pending_save
            had_pending_start = self._pending_start_task_started_at is not None
            if not was_recording and not had_pending and not had_pending_start:
                self._last_status = "nothing to discard"
                return False, self._last_status
            self._recording = False
            self._pending_save = False
            self._close_writer_locked()
            self._current_path = None
            self._frame_count = 0
            self._last_saved_path = None
            self._task_start_timestamp_iso = None
            self._task_started_at = None
            self._pending_start_task_timestamp_iso = None
            self._pending_start_task_started_at = None

        if path is not None and path.exists():
            path.unlink()

        with self._lock:
            self._last_status = "recording discarded"
            return True, self._last_status

    def open_browser(self):
        url = self.url
        opened = False
        try:
            opened = bool(webbrowser.open(url, new=1))
        except Exception:
            opened = False
        return opened, url

    def _make_visual_frame(self):
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            mode = self._infer_mode_locked()
            frame_count = int(self._frame_count)
            display_path = self._current_path.name if self._current_path is not None else "-"
            cup = self._selected_config["cup"] or "-"
            fullness = self._selected_config["fullness"] or "-"
            grasp = self._selected_config["grasp_type"] or "-"
            handover = self._selected_config["handover_location"] or "-"
            filling_amount = get_filling_amount(self._selected_config)
            filling_text = f"{filling_amount}ml" if filling_amount is not None else "-"
            config_id = get_configuration_id(self._selected_config)
            config_text = f"config_id={config_id}" if config_id is not None else "config_id=-"

        if frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Waiting for camera frame...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        if mode == "recording":
            mode_text = "REC"
            mode_color = (0, 0, 255)
        elif mode == "pending_save":
            mode_text = "PENDING SAVE"
            mode_color = (0, 200, 255)
        else:
            mode_text = "IDLE"
            mode_color = (0, 255, 0)

        cv2.putText(frame, mode_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 2)
        cv2.putText(frame, f"frames={frame_count}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"file: {display_path}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"cup={cup}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"fullness={fullness} ({filling_text})", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"grasp={grasp}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"handover={handover}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, config_text, (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame

    def _generate_stream(self):
        sleep_s = 1.0 / max(float(self.fps), 1.0)
        while not self._stop_event.is_set():
            frame = self._make_visual_frame()
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                time.sleep(sleep_s)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(sleep_s)

    def _register_routes(self):
        app = self._app

        @app.route("/")
        def index():
            return render_template_string(HTML)

        @app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_stream(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/status")
        def status():
            return jsonify(self.get_status())

        @app.route("/set_selection", methods=["POST"])
        def set_selection():
            data = request.get_json(silent=True) or {}
            field = data.get("field")
            value = data.get("value")
            self.set_selection(field, value)
            return jsonify(self.get_status())

        @app.route("/start", methods=["POST"])
        def start_recording():
            _, message = self.start_recording_for_task()
            return jsonify({"message": message})

        @app.route("/finish", methods=["POST"])
        def finish_recording():
            _, message = self.finish_recording_to_pending()
            return jsonify({"message": message})

        @app.route("/save", methods=["POST"])
        def save_recording():
            _, message = self.save_pending_recording()
            return jsonify({"message": message})

        @app.route("/discard", methods=["POST"])
        def discard_recording():
            _, message = self.discard_pending_recording()
            return jsonify({"message": message})


def main():
    recorder = HandoverVideoRecorderService()
    url = recorder.start_server()
    print(f"Open: {url}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()
